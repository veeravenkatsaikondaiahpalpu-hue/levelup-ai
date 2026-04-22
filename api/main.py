"""
api/main.py
===========
FastAPI application entry point for LevelUp AI backend.

Starts the fine-tuned chatbot model on startup and mounts all routers.

Run locally:
    uvicorn api.main:app --reload --port 8000

Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
    (workers=1 because the GPU model is a singleton — do NOT fork)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Config from environment ────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
ADAPTER_PATH  = os.getenv("LEVELUP_ADAPTER",    str(ROOT / "models" / "final" / "levelup-qlora-all"))
BASE_MODEL    = os.getenv("LEVELUP_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LOAD_MODEL    = os.getenv("LEVELUP_LOAD_MODEL", "true").lower() == "true"


# ── Lifespan: load model on startup ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LLM at startup; release on shutdown."""
    if LOAD_MODEL:
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            from chatbot.inference import LevelUpChat
            print(f"\n[LevelUp API] Loading model from {ADAPTER_PATH} ...")
            model = LevelUpChat(adapter_path=ADAPTER_PATH, base_model=BASE_MODEL)
            model.load()
            app.state.chat_model = model
            print("[LevelUp API] Model ready.\n")
        except Exception as e:
            print(f"[LevelUp API] WARNING: Could not load model: {e}")
            print("[LevelUp API] Chat endpoint will return 503 until model is available.\n")
            app.state.chat_model = None
    else:
        print("[LevelUp API] LEVELUP_LOAD_MODEL=false — skipping model load (dev mode)")
        app.state.chat_model = None

    yield  # server is running

    # Cleanup on shutdown
    if getattr(app.state, "chat_model", None):
        del app.state.chat_model
        print("[LevelUp API] Model unloaded.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LevelUp AI API",
    description=(
        "Backend for LevelUp — the RPG self-improvement app. "
        "Fine-tuned LLaMA 3 chatbot + XP engine endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React Native app to reach the API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ────────────────────────────────────────────────────────────────────

from api.routes.chat import router as chat_router
app.include_router(chat_router)

# Placeholder routers (implement as needed)
# from api.routes.activity import router as activity_router
# from api.routes.profile  import router as profile_router
# app.include_router(activity_router)
# app.include_router(profile_router)


# ── Root endpoint ──────────────────────────────────────────────────────────────

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
    model_loaded = getattr(app.state, "chat_model", None) is not None
    return {
        "status":       "ok",
        "model_loaded": model_loaded,
    }


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,             # don't reload — model load is expensive
        workers=1,
    )
