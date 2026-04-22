"""
api/routes/chat.py
==================
FastAPI router for the LevelUp AI chatbot endpoint.

Endpoints:
  POST /api/chat          -- generate a reply from the AI companion
  GET  /api/chat/health   -- check if model is loaded and ready
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from chatbot.inference import get_chat_model, LevelUpChat
from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Request / Response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    build:   str = Field(..., description="Build name: TITAN | ORACLE | PHANTOM | SAGE | MUSE | EMPIRE | GG")
    message: str = Field(..., description="The user's latest message", min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default=[], description="Prior conversation turns (max 10)")

    # Runtime context injected from the user's profile (optional)
    # If provided, the full dynamic system prompt is used instead of the fine-tune prompt
    user_context: Optional[dict] = Field(
        default=None,
        description="UserState context dict from xp_engine (optional)"
    )

    # Generation parameters (optional overrides)
    max_tokens:  int   = Field(default=512,  ge=50,  le=1024)
    temperature: float = Field(default=0.7,  ge=0.1, le=1.5)


class ChatResponse(BaseModel):
    reply:      str
    build:      str
    latency_ms: int
    tokens_out: Optional[int] = None


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    builds:      list[str]


# ── Dependency: get the loaded model ──────────────────────────────────────────

def get_model(request: Request) -> LevelUpChat:
    """Retrieve the LevelUpChat instance stored in app state."""
    model = getattr(request.app.state, "chat_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")
    return model


# ── Endpoints ──────────────────────────────────────────────────────────────────

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

    # Build the system prompt
    system_prompt: Optional[str] = None
    if req.user_context:
        # Use the rich runtime prompt (includes user stats, streak, recent activity)
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from chatbot.system_prompt import build_system_prompt, detect_mode
            from chatbot.sentiment import predict as sentiment_predict   # optional
            try:
                sentiment = sentiment_predict(req.message)
            except Exception:
                sentiment = None
            mode = detect_mode(req.message, sentiment)
            system_prompt = build_system_prompt(req.user_context, sentiment=sentiment, mode=mode)
        except Exception:
            system_prompt = None  # fall back to fine-tune prompt

    # Convert history to the format expected by inference
    history = [{"role": m.role, "content": m.content} for m in req.history[-10:]]

    t0 = time.time()
    reply = model.generate(
        build=build,
        user_message=req.message,
        history=history if history else None,
        system_override=system_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    latency = int((time.time() - t0) * 1000)

    return ChatResponse(
        reply=reply,
        build=build,
        latency_ms=latency,
    )
