"""
FastAPI entry point for the LevelUp AI backend.

On startup it loads the fine-tuned LoRA model (or falls back to Ollama if
the GPU model can't load), then mounts all the route handlers.

Run locally:
    uvicorn api.main:app --reload --port 8000

Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
    Keep workers=1 — the GPU model is a singleton and can't be forked safely.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.logging_config import setup_logging
from api.routes.activity import router as activity_router
from api.routes.chat import router as chat_router
from api.routes.profile import router as profile_router
from api.routes.sentiment import router as sentiment_router
from api.routes.vision import router as vision_router

setup_logging()
logger = logging.getLogger(__name__)

# Read config from environment so we can swap models without touching code
ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = os.getenv("LEVELUP_ADAPTER", str(ROOT / "models" / "final" / "levelup-qlora-cloud"))
BASE_MODEL = os.getenv("LEVELUP_BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct")
LOAD_MODEL = os.getenv("LEVELUP_LOAD_MODEL", "true").lower() == "true"

# Vision model is off by default — it needs 14GB+ VRAM (Qwen2.5-VL-7B)
VISION_MODEL = os.getenv("LEVELUP_VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
LOAD_VISION = os.getenv("LEVELUP_LOAD_VISION", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LLM at startup; release on shutdown."""
    if LOAD_MODEL:
        try:
            from chatbot.inference import LevelUpChat

            logger.info("Loading LoRA model from %s", ADAPTER_PATH)
            model = LevelUpChat(adapter_path=ADAPTER_PATH, base_model=BASE_MODEL)
            model.load()
            app.state.chat_model = model
            logger.info("LoRA model ready")
        except Exception:
            # Not fatal: we fall through to the Ollama backend below.
            logger.warning("Could not load LoRA model", exc_info=True)
            app.state.chat_model = None
    else:
        logger.info("LEVELUP_LOAD_MODEL=false - skipping LoRA model load")
        app.state.chat_model = None

    # If the LoRA adapter failed (no GPU, no torch, etc.), try Ollama as a fallback
    if app.state.chat_model is None:
        try:
            from chatbot.ollama_inference import get_ollama_model

            ollama = get_ollama_model()
            if ollama is not None:
                app.state.chat_model = ollama
                logger.info("Using Ollama (llama3.2) as local backend")
            else:
                logger.warning("Ollama not available - /api/chat will return 503")
        except Exception:
            logger.warning("Ollama fallback failed", exc_info=True)

    if LOAD_VISION:
        try:
            from chatbot.vision_inference import VisionChat

            logger.info("Loading vision model %s", VISION_MODEL)
            vision = VisionChat(model_name=VISION_MODEL)
            vision.load()
            app.state.vision_model = vision
            logger.info("Vision model ready")
        except Exception:
            logger.warning("Could not load vision model", exc_info=True)
            app.state.vision_model = None
    else:
        logger.info("LEVELUP_LOAD_VISION=false - skipping vision model load")
        app.state.vision_model = None

    yield  # server is running

    if getattr(app.state, "chat_model", None):
        del app.state.chat_model
        logger.info("Model unloaded")

    if getattr(app.state, "vision_model", None):
        app.state.vision_model.unload()


# App

app = FastAPI(
    title="LevelUp AI API",
    description=(
        "Backend for LevelUp — the RPG self-improvement app. "
        "Fine-tuned LLaMA 3 chatbot + XP engine endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS for now so the React Native app can reach the API during development.
# Restrict allow_origins to specific domains before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers

app.include_router(chat_router)
app.include_router(vision_router)
app.include_router(activity_router)
app.include_router(sentiment_router)
app.include_router(profile_router)


# Root endpoint

@app.get("/")
async def root():
    return {
        "app":     "LevelUp AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "chat":    "/api/chat",
    }


@app.get("/health")
async def health():
    model_loaded  = getattr(app.state, "chat_model",  None) is not None
    vision_loaded = getattr(app.state, "vision_model", None) is not None
    return {
        "status":        "ok",
        "model_loaded":  model_loaded,
        "vision_loaded": vision_loaded,
    }


@app.get("/demo", include_in_schema=False)
async def serve_demo():
    """Serve the LevelUp AI demo frontend."""
    demo_path = ROOT / "frontend" / "index.html"
    if demo_path.exists():
        return FileResponse(str(demo_path), media_type="text/html")
    return {"error": "Frontend not found. Run from project root."}


@app.get("/app", include_in_schema=False)
async def serve_app():
    """Serve the LevelUp AI full RPG app (Stitch design)."""
    app_path = ROOT / "frontend" / "app.html"
    if app_path.exists():
        return FileResponse(str(app_path), media_type="text/html")
    return {"error": "App not found. Run from project root."}


# Serve static frontend files (CSS, JS, images if any)
_frontend_dir = ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(_frontend_dir)), name="frontend")


@app.get("/api/voices", tags=["voice"])
async def voice_profiles():
    """
    Return the ElevenLabs voice profile for every build.
    Shows voice ID, stability, similarity, style, and description.
    Useful for confirming which voice + settings each persona uses.
    """
    try:
        from voice.tts import list_profiles, tts_available
        return {
            "tts_available": tts_available(),
            "model":         os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
            "profiles":      list_profiles(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/tts", tags=["voice"])
async def text_to_speech(request: Request):
    """
    Convert AI response text to speech using ElevenLabs.
    Uses the build-specific voice profile (TITAN = deep intense, GG = hype, etc.)

    Request body:
        { "text": "...", "build": "TITAN" }

    Returns:
        audio/mpeg bytes (MP3) ready to play in <audio> or Web Audio API.

    503 if ELEVENLABS_API_KEY not set in .env
    """
    body = await request.json()
    text  = body.get("text", "").strip()
    build = body.get("build", "ORACLE").upper()

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        from voice.tts import speak_async, tts_available
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"TTS module error: {e}") from e

    if not tts_available():
        raise HTTPException(
            status_code=503,
            detail="TTS unavailable: add ELEVENLABS_API_KEY to .env and pip install elevenlabs"
        )

    try:
        audio_bytes = await speak_async(text, build=build)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e


@app.post("/api/stt", tags=["voice"])
async def speech_to_text(file: UploadFile = File(...)):
    """
    Transcribe voice input to text using local OpenAI Whisper.
    Accepts any audio format (WebM, MP3, WAV, M4A, OGG).

    Returns:
        { "text": "transcribed text" }

    503 if openai-whisper not installed.
    """
    try:
        from voice.stt import stt_available, transcribe_bytes_async
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"STT module error: {e}") from e

    if not stt_available():
        raise HTTPException(
            status_code=503,
            detail="STT unavailable: run 'pip install openai-whisper' in the venv"
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    ext = Path(file.filename).suffix if (file.filename and "." in file.filename) else ".webm"

    try:
        text = await transcribe_bytes_async(audio_bytes, file_ext=ext)
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}") from e


# Dev server

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,             # don't reload — model load is expensive
        workers=1,
    )
