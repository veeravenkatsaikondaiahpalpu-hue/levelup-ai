"""
voice/
======
Voice I/O for LevelUp AI.

  tts.py  — ElevenLabs Text-to-Speech (build-specific voices)
  stt.py  — OpenAI Whisper Speech-to-Text (local, no API key needed)
"""

from voice.tts import text_to_audio_bytes, speak_async, tts_available
from voice.stt import transcribe_bytes, transcribe_file, stt_available

__all__ = [
    "text_to_audio_bytes",
    "speak_async",
    "tts_available",
    "transcribe_bytes",
    "transcribe_file",
    "stt_available",
]
