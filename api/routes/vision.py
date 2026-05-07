"""
api/routes/vision.py
====================
FastAPI router for LevelUp AI multimodal (image + text) endpoint.

Endpoints:
  POST /api/chat/vision        -- send an image + text, get a reply
  GET  /api/chat/vision/health -- check if vision model is loaded
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from chatbot.vision_inference import get_vision_model, VisionChat
from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS

router = APIRouter(prefix="/api/chat/vision", tags=["vision"])


# ── Response models ────────────────────────────────────────────────────────────

class VisionResponse(BaseModel):
    reply:      str
    build:      str
    latency_ms: int
    has_image:  bool
    audio_b64:  Optional[str] = None   # base64-encoded MP3 if audio=true


class VisionHealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    builds:       list[str]


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_model(request: Request) -> VisionChat:
    model = getattr(request.app.state, "vision_model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Vision model not loaded. Set LEVELUP_LOAD_VISION=true and restart.",
        )
    return model


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=VisionHealthResponse)
async def vision_health(request: Request):
    model = getattr(request.app.state, "vision_model", None)
    return VisionHealthResponse(
        status="ok" if model is not None else "not_loaded",
        model_loaded=model is not None,
        model_name=model.model_name if model else "Qwen/Qwen2.5-VL-7B-Instruct",
        builds=list(FINETUNE_SYSTEM_PROMPTS.keys()),
    )


@router.post("")
async def vision_chat(
    # Required fields (multipart form)
    build:   str = Form(..., description="Build name: TITAN | ORACLE | PHANTOM | SAGE | MUSE | EMPIRE | GG"),
    message: str = Form(..., description="The user's text question about the image"),

    # Optional image upload
    image:   Optional[UploadFile] = File(default=None, description="Image file (JPEG, PNG, WebP, GIF)"),

    # Optional generation params
    max_tokens:  int   = Form(default=512),
    temperature: float = Form(default=0.7),

    # Optional: return audio with the reply
    audio:   bool  = Form(default=False, description="If true, include base64-encoded MP3 audio"),
    audio_build: Optional[str] = Form(default=None, description="Build to use for TTS voice (defaults to build)"),

    model: VisionChat = Depends(get_model),
):
    """
    Multimodal chat endpoint.

    Send a form-data POST with:
    - `build`   : persona name
    - `message` : your question
    - `image`   : (optional) an image file
    - `audio`   : set to true to get an MP3 audio response back

    Returns JSON with `reply`, `build`, `latency_ms`, `has_image`,
    and optionally `audio_b64` (base64-encoded MP3).
    """
    build = build.upper()
    if build not in FINETUNE_SYSTEM_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown build '{build}'. Choose from: {list(FINETUNE_SYSTEM_PROMPTS.keys())}",
        )

    # Read image bytes if provided
    image_bytes: Optional[bytes] = None
    image_ext:   str = ".jpg"
    if image is not None:
        image_bytes = await image.read()
        if image.content_type:
            ct = image.content_type.lower()
            if "png" in ct:
                image_ext = ".png"
            elif "webp" in ct:
                image_ext = ".webp"
            elif "gif" in ct:
                image_ext = ".gif"

    t0 = time.time()
    reply = model.generate(
        build=build,
        user_message=message,
        image_input=image_bytes,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    latency = int((time.time() - t0) * 1000)

    # Optional TTS
    audio_b64: Optional[str] = None
    if audio:
        try:
            import base64
            from voice.tts import speak_async
            tts_build = (audio_build or build).upper()
            audio_bytes_out = await speak_async(reply, build=tts_build)
            audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
        except Exception as e:
            # TTS is best-effort — don't fail the whole request
            print(f"[TTS] Warning: could not generate audio: {e}")

    return VisionResponse(
        reply=reply,
        build=build,
        latency_ms=latency,
        has_image=image_bytes is not None,
        audio_b64=audio_b64,
    )
