"""
spaces/backend/app.py
=====================
HuggingFace Spaces entry point for LevelUp AI backend.

Runs as a Gradio Space (required for ZeroGPU support).
All /api/* REST routes are mounted on Gradio's internal FastAPI app.

Environment variables (set in Space Settings → Secrets):
  HF_TOKEN              — HuggingFace token to download LLaMA 3.2
  LEVELUP_ADAPTER       — HF repo ID of LoRA adapter
                          e.g. "your-username/levelup-qlora"
  ELEVENLABS_API_KEY    — optional, enables per-build AI voices
  WHISPER_MODEL         — optional, base / small / medium (default: base)
"""

from __future__ import annotations
import os, sys, json, tempfile
from pathlib import Path

import gradio as gr
import spaces                          # HuggingFace ZeroGPU
from fastapi import Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN      = os.getenv("HF_TOKEN", "")
ADAPTER_REPO  = os.getenv("LEVELUP_ADAPTER", "")   # e.g. "veera/levelup-qlora"
BASE_MODEL    = os.getenv("LEVELUP_BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct")
ELEVENLABS    = os.getenv("ELEVENLABS_API_KEY", "")
WHISPER_SIZE  = os.getenv("WHISPER_MODEL", "base")

# ── Model (lazy-loaded on first GPU call) ─────────────────────────────────────

_chat_model = None

def _load_chat_model():
    global _chat_model
    if _chat_model is not None:
        return _chat_model
    if not ADAPTER_REPO:
        raise RuntimeError(
            "LEVELUP_ADAPTER not set. "
            "Add your HF adapter repo ID in Space Settings → Variables."
        )
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_REPO,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        FastLanguageModel.for_inference(model)
        _chat_model = (model, tokenizer)
        print("[LevelUp] Chat model ready.")
    except Exception as e:
        print(f"[LevelUp] WARNING: Could not load chat model: {e}")
        _chat_model = None
    return _chat_model


# ── ZeroGPU inference ─────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def _run_inference(system_prompt: str, history: list[dict], user_msg: str) -> str:
    """Runs on ZeroGPU — allocated on demand, released after 60 s."""
    pair = _load_chat_model()
    if pair is None:
        return "Model not available. Check Space logs."
    model, tokenizer = pair

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])           # keep last 10 turns
    messages.append({"role": "user", "content": user_msg})

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── System prompts ────────────────────────────────────────────────────────────

BUILD_PROMPTS = {
    "TITAN":   "You are TITAN, an intense warrior strength coach. You speak with raw power and drive. Keep responses punchy and motivating. You help with lifting, nutrition, and physical training.",
    "ORACLE":  "You are ORACLE, a sharp analytical mentor. You explain complex topics clearly with precision. You help with programming, learning, science, and logical problem-solving.",
    "PHANTOM": "You are PHANTOM, a fluid precision coach. You are calm but dynamic. You help with martial arts, parkour, agility training, and movement.",
    "SAGE":    "You are SAGE, a calm mindful guide. You speak slowly and grounded. You help with wellness, meditation, mental health, and balance.",
    "MUSE":    "You are MUSE, a warm creative director. You are expressive and encouraging. You help with art, music, writing, and design.",
    "EMPIRE":  "You are EMPIRE, an authoritative startup co-founder. You speak like a seasoned business leader. You help with finance, productivity, and entrepreneurship.",
    "GG":      "You are GG, a hyped esports coach with max energy. You use gaming language and keep the hype high. You help with competitive gaming, streaming, and improving game skills.",
}


# ── Anomaly & Sentiment (CPU — no GPU needed) ─────────────────────────────────

def _load_anomaly_models():
    try:
        import joblib
        base = Path(__file__).parent / "anomaly_detection"
        lr   = joblib.load(base / "logistic_regression_model.pkl")
        dt   = joblib.load(base / "decision_tree_model.pkl")
        return lr, dt
    except Exception as e:
        print(f"[LevelUp] Anomaly models not loaded: {e}")
        return None, None

def _load_sentiment_model():
    try:
        from chatbot.sentiment import SentimentAnalyzer
        return SentimentAnalyzer()
    except Exception as e:
        print(f"[LevelUp] Sentiment model not loaded: {e}")
        return None

_anomaly_lr, _anomaly_dt = _load_anomaly_models()
_sentiment_model = _load_sentiment_model()


# ── Gradio UI (minimal — just shows API info) ─────────────────────────────────

with gr.Blocks(title="LevelUp AI — Backend API") as demo:
    gr.Markdown("""
# ⚔️ LevelUp AI — Backend API

This Space exposes the REST API for the **LevelUp AI** RPG self-improvement app.

Use the [Frontend Space](https://huggingface.co/spaces) to interact with the app.

### Available Endpoints
| Route | Method | Description |
|---|---|---|
| `/api/chat` | POST | Fine-tuned LLM chatbot |
| `/api/activity/check` | POST | Anomaly detection |
| `/api/sentiment/analyze` | POST | Sentiment analysis |
| `/api/tts` | POST | Text-to-speech (ElevenLabs) |
| `/api/stt` | POST | Speech-to-text (Whisper) |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger docs |

### Quick Test
""")
    with gr.Row():
        build_dd  = gr.Dropdown(list(BUILD_PROMPTS.keys()), value="ORACLE", label="Build")
        msg_box   = gr.Textbox(label="Message", placeholder="Ask your AI coach something...")
    reply_box = gr.Textbox(label="Reply", interactive=False)
    test_btn  = gr.Button("Send", variant="primary")

    def quick_chat(build, message):
        if not message.strip():
            return "Please enter a message."
        try:
            sys_prompt = BUILD_PROMPTS.get(build.upper(), BUILD_PROMPTS["ORACLE"])
            return _run_inference(sys_prompt, [], message)
        except Exception as e:
            return f"Error: {e}"

    test_btn.click(quick_chat, inputs=[build_dd, msg_box], outputs=reply_box)


# ── CORS ──────────────────────────────────────────────────────────────────────

demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /health ───────────────────────────────────────────────────────────────────

@demo.app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": _chat_model is not None,
        "adapter_repo": ADAPTER_REPO or "not set",
    }


# ── /api/chat ─────────────────────────────────────────────────────────────────

@demo.app.post("/api/chat")
async def chat_endpoint(request: Request):
    body    = await request.json()
    message = (body.get("message") or "").strip()
    build   = (body.get("build") or "ORACLE").upper()
    history = body.get("history", [])

    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    sys_prompt = BUILD_PROMPTS.get(build, BUILD_PROMPTS["ORACLE"])
    try:
        reply = _run_inference(sys_prompt, history, message)
        return JSONResponse({"reply": reply, "build": build})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/activity/check ───────────────────────────────────────────────────────

@demo.app.post("/api/activity/check")
async def activity_check(request: Request):
    if _anomaly_lr is None:
        return JSONResponse({"anomaly": False, "note": "Models not loaded"})
    body     = await request.json()
    features = body.get("features", [])
    import numpy as np
    X = np.array(features).reshape(1, -1)
    lr_pred = int(_anomaly_lr.predict(X)[0])
    dt_pred = int(_anomaly_dt.predict(X)[0])
    anomaly = bool(lr_pred or dt_pred)
    return JSONResponse({
        "anomaly":            anomaly,
        "logistic_regression": lr_pred,
        "decision_tree":       dt_pred,
    })


# ── /api/sentiment/analyze ────────────────────────────────────────────────────

@demo.app.post("/api/sentiment/analyze")
async def sentiment_endpoint(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if _sentiment_model is None:
        return JSONResponse({"label": "neutral", "score": 0.5, "note": "Model not loaded"})
    try:
        result = _sentiment_model.analyze(text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/tts ─────────────────────────────────────────────────────────────────

@demo.app.post("/api/tts")
async def tts_endpoint(request: Request):
    body  = await request.json()
    text  = (body.get("text") or "").strip()
    build = (body.get("build") or "ORACLE").upper()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if not ELEVENLABS:
        raise HTTPException(status_code=503, detail="TTS unavailable: ELEVENLABS_API_KEY not set")
    try:
        from voice.tts import speak_async
        audio_bytes = await speak_async(text, build=build)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/stt ─────────────────────────────────────────────────────────────────

@demo.app.post("/api/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    try:
        import whisper as _whisper   # noqa: F401
    except ImportError:
        raise HTTPException(status_code=503, detail="STT unavailable: openai-whisper not installed")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    ext = Path(file.filename).suffix if (file.filename and "." in file.filename) else ".webm"
    try:
        from voice.stt import transcribe_bytes_async
        text = await transcribe_bytes_async(audio_bytes, file_ext=ext)
        return JSONResponse({"text": text.strip()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
