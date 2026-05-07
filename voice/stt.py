"""
voice/stt.py
============
Speech-to-Text using OpenAI Whisper (local, runs on CPU or GPU).

Converts uploaded audio files (MP3, WAV, M4A, WebM, OGG) to text so users
can speak to LevelUp instead of typing.

Usage:
    from voice.stt import transcribe_bytes, transcribe_file

    # From raw audio bytes (e.g. uploaded via FastAPI)
    text = transcribe_bytes(audio_bytes, file_ext=".webm")

    # From a file path
    text = transcribe_file("recording.mp3")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


# ── Whisper model size ────────────────────────────────────────────────────────
# "base" is fast and accurate enough for voice commands.
# Upgrade to "small" or "medium" if accuracy needs improvement.
_WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

# Cached model instance (lazy-loaded)
_whisper_model = None


def _get_model():
    """Load Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. "
                "Run: pip install openai-whisper"
            )
        print(f"[STT] Loading Whisper {_WHISPER_MODEL_SIZE} model ...")
        _whisper_model = whisper.load_model(_WHISPER_MODEL_SIZE)
        print("[STT] Whisper ready.")
    return _whisper_model


# ── Core transcription ────────────────────────────────────────────────────────

def transcribe_bytes(
    audio_bytes: bytes,
    file_ext:    str = ".webm",
    language:    Optional[str] = None,
) -> str:
    """
    Transcribe raw audio bytes to text.

    Args:
        audio_bytes : Raw audio data (any format Whisper supports)
        file_ext    : File extension hint (e.g. ".mp3", ".wav", ".webm", ".m4a")
        language    : ISO-639-1 language code hint (e.g. "en"). None = auto-detect.

    Returns:
        Transcribed text string.
    """
    # Write bytes to a temp file so Whisper's ffmpeg pipeline can read it
    suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return _transcribe_path(tmp_path, language=language)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def transcribe_file(
    file_path: str | Path,
    language:  Optional[str] = None,
) -> str:
    """
    Transcribe an audio file to text.

    Args:
        file_path : Path to audio file
        language  : ISO-639-1 language hint (None = auto-detect)

    Returns:
        Transcribed text string.
    """
    return _transcribe_path(str(file_path), language=language)


def _transcribe_path(path: str, language: Optional[str] = None) -> str:
    """Internal: run Whisper on a file path."""
    model = _get_model()
    options: dict = {"fp16": False}   # fp16=False ensures CPU compatibility
    if language:
        options["language"] = language
    result = model.transcribe(path, **options)
    return result["text"].strip()


# ── Async wrapper (for FastAPI) ───────────────────────────────────────────────

async def transcribe_bytes_async(
    audio_bytes: bytes,
    file_ext:    str = ".webm",
    language:    Optional[str] = None,
) -> str:
    """
    Async wrapper — runs Whisper in a thread pool so it doesn't block
    the FastAPI event loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: transcribe_bytes(audio_bytes, file_ext=file_ext, language=language),
    )


# ── Availability check ────────────────────────────────────────────────────────

def stt_available() -> bool:
    """Returns True if openai-whisper is installed."""
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m voice.stt <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Transcribing: {audio_path}")
    result = transcribe_file(audio_path)
    print(f"\nTranscript:\n{result}")
