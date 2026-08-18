"""
The main chatbot endpoint — this is what the React Native app calls on every message.

Endpoints:
  POST /api/chat          — generate a reply from the AI companion
  GET  /api/chat/health   — check if the model is loaded and ready
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from chatbot.guardrails import Guardrail
from chatbot.guardrails.demo_scenario import DEMO_ON, DEMO_TEMPERATURE, inject_secret
from chatbot.inference import LevelUpChat
from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Runtime guardrail ("AI firewall") wrapped around the model. Inspects input
# before generation and output after. Default state via LEVELUP_GUARDRAILS;
# a per-request `safe_mode` flag (from the app's Firewall toggle) overrides it.
GUARDRAILS_ON = os.getenv("LEVELUP_GUARDRAILS", "on").lower() not in ("off", "false", "0")
_guard = Guardrail()


# Request / Response models

class ChatMessage(BaseModel):
    role:    str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    build:   str = Field(..., description="Build name: TITAN | ORACLE | PHANTOM | SAGE | MUSE | EMPIRE | GG")
    message: str = Field(..., description="The user's latest message", min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default=[], description="Prior conversation turns (max 10)")

    # If the app provides user stats (XP, streak, badge), we build a richer system
    # prompt that references those numbers. Without this, we use the standard fine-tune prompt.
    user_context: dict | None = Field(
        default=None,
        description="UserState context dict from xp_engine (optional)"
    )

    # Generation parameters (optional overrides)
    max_tokens:  int   = Field(default=512,  ge=50,  le=1024)
    temperature: float = Field(default=0.7,  ge=0.1, le=1.5)

    # Audio response (ElevenLabs TTS)
    audio: bool = Field(
        default=False,
        description="If true, return base64-encoded MP3 audio alongside the text reply",
    )

    # Guardrail "Firewall" toggle from the app UI. None = use the server default
    # (LEVELUP_GUARDRAILS); True/False = force the guardrail on/off for this request.
    safe_mode: bool | None = Field(
        default=None,
        description="Per-request guardrail override: true = firewall on, false = off",
    )


class ChatResponse(BaseModel):
    reply:      str
    build:      str
    latency_ms: int
    tokens_out: int | None = None
    audio_b64:  str | None = None   # base64-encoded MP3 when audio=true is requested

    # Guardrail telemetry — lets the app (and the demo) show what the firewall did.
    flagged:   bool = False            # true if any rule matched (input or output)
    blocked:   bool = False            # true if the input was blocked before generation
    guardrail: dict | None = None   # {"input": {...}, "output": {...}} scan details


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    builds:      list[str]


# Dependency: get the loaded model

def get_model(request: Request) -> LevelUpChat:
    """Retrieve the LevelUpChat instance stored in app state."""
    model = getattr(request.app.state, "chat_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")
    return model


# Endpoints

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    model = getattr(request.app.state, "chat_model", None)
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        builds=list(FINETUNE_SYSTEM_PROMPTS.keys()),
    )


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, model: LevelUpChat = Depends(get_model)):
    """
    Generate a reply from the LevelUp AI companion.

    The build field selects which persona to use.
    Pass history to maintain multi-turn conversation context.
    """
    build = req.build.upper()
    if build not in FINETUNE_SYSTEM_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown build '{build}'. Choose from: {list(FINETUNE_SYSTEM_PROMPTS.keys())}"
        )

    # If user stats were provided, build the rich dynamic system prompt.
    # If anything fails (missing models, bad context) we fall back to the
    # standard fine-tune prompt — the chat always returns something.
    system_prompt: str | None = None
    if req.user_context:
        try:
            # Imported lazily: the sentiment model is optional, so a missing
            # dependency degrades to the standard prompt instead of a 500.
            from chatbot.sentiment import predict as sentiment_predict
            from chatbot.system_prompt import build_system_prompt, detect_mode
            try:
                sentiment = sentiment_predict(req.message)
            except Exception:
                sentiment = None
            mode = detect_mode(req.message, sentiment)
            system_prompt = build_system_prompt(req.user_context, sentiment=sentiment, mode=mode)
        except Exception:
            system_prompt = None

    # Keep only the last 10 turns so we don't overflow the context window
    history = [{"role": m.role, "content": m.content} for m in req.history[-10:]]

    # Effective firewall state: the app's toggle wins; otherwise the server default.
    guardrails_active = GUARDRAILS_ON if req.safe_mode is None else bool(req.safe_mode)

    # Demo scenario: plant confidential data in the model's context so an
    # injection has something real to leak (Case 1 vs Case 2). Off by default;
    # enabled only when the server runs with LEVELUP_DEMO=on.
    gen_temperature = req.temperature
    if DEMO_ON:
        base_prompt = system_prompt or FINETUNE_SYSTEM_PROMPTS.get(build, FINETUNE_SYSTEM_PROMPTS["ORACLE"])
        system_prompt = inject_secret(base_prompt)
        gen_temperature = DEMO_TEMPERATURE  # deterministic leak on stage

    # Guardrail: INPUT scan
    # Inspect the untrusted surface (message + user turns + ingested context)
    # BEFORE the model sees it. A blocked input never reaches generation.
    guardrail_report: dict | None = None
    if guardrails_active:
        in_verdict = _guard.check_input(
            req.message, history=history, user_context=req.user_context,
        )
        guardrail_report = {"input": in_verdict.as_dict()}
        if in_verdict.blocked:
            return ChatResponse(
                reply=_guard.safe_refusal(build),
                build=build,
                latency_ms=0,
                flagged=True,
                blocked=True,
                guardrail=guardrail_report,
            )

    t0 = time.time()
    reply = model.generate(
        build=build,
        user_message=req.message,
        history=history if history else None,
        system_override=system_prompt,
        max_new_tokens=req.max_tokens,
        temperature=gen_temperature,
    )
    latency = int((time.time() - t0) * 1000)

    # Guardrail: OUTPUT scan
    # Catch a successful injection by its symptoms: prompt leakage, identity
    # break, or an exfiltration channel (EchoLeak-style auto-loading image).
    flagged = bool(guardrail_report and guardrail_report["input"]["findings"])
    blocked_out = False
    if guardrails_active:
        out_verdict = _guard.check_output(reply)
        reply = out_verdict.safe_text or reply
        guardrail_report = (guardrail_report or {})
        guardrail_report["output"] = out_verdict.as_dict()
        flagged = flagged or out_verdict.flagged
        blocked_out = out_verdict.blocked

    # Optional TTS audio (of the sanitised reply)
    audio_b64: str | None = None
    if req.audio:
        try:
            import base64

            from voice.tts import speak_async
            audio_bytes = await speak_async(reply, build=build)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception:
            # Audio is a nice-to-have; the text reply still goes out.
            logger.warning("TTS failed - returning reply without audio", exc_info=True)

    return ChatResponse(
        reply=reply,
        build=build,
        latency_ms=latency,
        audio_b64=audio_b64,
        flagged=flagged,
        blocked=blocked_out,
        guardrail=guardrail_report,
    )
