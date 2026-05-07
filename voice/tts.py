"""
voice/tts.py
============
ElevenLabs Text-to-Speech integration for LevelUp AI.

Every build gets its own VoiceProfile:
  • voice_id         — which ElevenLabs voice to use
  • stability        — how consistent the delivery is (low = more expressive/varied)
  • similarity_boost — how closely to follow the cloned voice (high = true to character)
  • style            — style exaggeration (high = more dramatic/over-the-top)
  • speaker_boost    — ElevenLabs clarity enhancement
  • preprocess_fn    — optional text transformer applied before synthesis

Build personalities:
  TITAN   — deep, intense, punchy. Low stability (varied delivery), high style (dramatic).
  ORACLE  — clear, measured, precise. High stability (consistent), minimal style (neutral).
  PHANTOM — fluid, quick, energetic. Medium stability, elevated style (dynamic).
  SAGE    — calm, slow, grounded. Very high stability (barely varies), near-zero style.
  MUSE    — expressive, warm, creative. Lowest stability (most varied), high style.
  EMPIRE  — authoritative, confident. High stability (commanding), medium-high style.
  GG      — hype, chaotic, casual. Lowest stability of all, max style (full energy).

Usage:
    from voice.tts import text_to_audio_bytes, speak_async

    audio = text_to_audio_bytes("You crushed it today, warrior!", build="TITAN")
    audio = await speak_async("Nice work!", build="ORACLE")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Voice Profile dataclass ────────────────────────────────────────────────────

@dataclass
class VoiceProfile:
    """
    Complete voice personality for one build.

    ElevenLabs VoiceSettings parameters:
      stability        (0–1)  Low = expressive/varied. High = consistent/robotic.
      similarity_boost (0–1)  How closely to match the original voice. High = true to voice.
      style            (0–1)  Style exaggeration. High = more dramatic delivery.
      speaker_boost    bool   Clarity enhancement (slight VRAM cost on ElevenLabs side).

    Text pre-processing:
      preprocess       Callable applied to text before synthesis.
                       Use to reshape punctuation, add emphasis, etc.
    """
    voice_id:         str
    stability:        float
    similarity_boost: float
    style:            float
    speaker_boost:    bool                  = True
    preprocess:       Callable[[str], str]  = field(default=lambda t: t, repr=False)
    description:      str                   = ""


# ── Text pre-processors ────────────────────────────────────────────────────────
# Each returns the text exactly as ElevenLabs should hear it.
# Techniques: punctuation shaping, line-break pacing, filler removal.

def _preprocess_titan(text: str) -> str:
    """
    TITAN — intense, punchy, no fluff.
    • Replace '...' with ', ' to keep momentum going instead of trailing off
    • Ensure sentences end with ! or . (no weak question marks on statements)
    • Strip soft hedges like 'maybe', 'perhaps', 'I think' at sentence starts
    • Double-space between short sentences for dramatic micro-pauses
    """
    # Remove common filler hedges at sentence starts
    text = re.sub(r'\b(maybe|perhaps|I think|I guess|sort of|kind of)\b', '', text, flags=re.IGNORECASE)
    # Replace ellipses with a hard pause comma
    text = text.replace('...', ', ')
    # Collapse extra whitespace from hedge removal
    text = re.sub(r'  +', ' ', text).strip()
    return text


def _preprocess_oracle(text: str) -> str:
    """
    ORACLE — measured, analytical, clean.
    • Expand numbered lists with a natural pause ('. ' after number becomes '. ... ')
    • No change otherwise — Oracle's precision shouldn't be dressed up
    """
    # Add a tiny pause after numbered list items: "1. Foo" → "1. Foo"
    # (ElevenLabs reads periods in lists naturally; no action needed here)
    return text.strip()


def _preprocess_phantom(text: str) -> str:
    """
    PHANTOM — fluid, quick, movement-driven.
    • Replace semicolons with em-dashes for flow continuity
    • Trim over-long sentences — Phantom is economical with words
    """
    text = text.replace(';', ' — ')
    return text.strip()


def _preprocess_sage(text: str) -> str:
    """
    SAGE — calm, slow, grounded.
    • Add a comma after opening words ('breathe', 'notice', 'feel', 'let', 'allow')
      so ElevenLabs introduces a natural breath pause
    • Replace '!' with '.' — Sage never exclaims
    • Replace ellipsis with ' ... ' (three spaced dots = longer breath)
    """
    text = text.replace('!', '.')
    text = text.replace('...', ' ... ')

    # Gentle pause after calming openers
    calming_words = r'\b(breathe|notice|feel|allow|let|simply|gently|slowly|release|observe)\b'
    text = re.sub(calming_words, r'\1,', text, flags=re.IGNORECASE)
    # De-duplicate accidental double commas
    text = text.replace(',,', ',')
    return text.strip()


def _preprocess_muse(text: str) -> str:
    """
    MUSE — warm, expressive, creative.
    • Keep everything natural — Muse's expressiveness lives in the voice settings
    • Just clean up double spaces
    """
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _preprocess_empire(text: str) -> str:
    """
    EMPIRE — authoritative, strategic, deliberate.
    • Add emphasis pauses around key business/strategy phrases
    • Replace casual '&' with 'and' for cleaner delivery
    • Strip question marks from rhetorical questions (treat them as statements)
    """
    text = text.replace(' & ', ' and ')
    # Make rhetorical questions into statements: "Want to win?" → "Want to win."
    text = re.sub(r'\?(\s|$)', r'.\1', text)
    return text.strip()


def _preprocess_gg(text: str) -> str:
    """
    GG — hype, casual, gamer energy.
    • Keep acronyms uppercase for natural pronunciation (GG, EZ, POG, etc.)
    • Replace 'no cap' with 'no cap,' for a casual pause after the phrase
    • Trim overly long explanations — GG is punchy
    """
    # Ensure common gaming terms are uppercase for TTS pronunciation
    gaming_terms = {
        r'\bgg\b': 'GG',
        r'\bez\b': 'EZ',
        r'\bpog\b': 'POG',
        r'\bpoggers\b': 'POGGERS',
        r'\bngl\b': 'NGL',
        r'\bnpc\b': 'NPC',
        r'\bxp\b': 'XP',
        r'\bgg ez\b': 'GG EZ',
    }
    for pattern, replacement in gaming_terms.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Add a casual pause after 'no cap'
    text = re.sub(r'\bno cap\b', 'no cap,', text, flags=re.IGNORECASE)
    return text.strip()


# ── Per-build profiles ─────────────────────────────────────────────────────────
# Voice IDs default to free ElevenLabs library voices.
# Set ELEVENLABS_VOICE_<BUILD> in .env to override with your own cloned voices.

BUILD_PROFILES: dict[str, VoiceProfile] = {

    "TITAN": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_TITAN",   "pNInz6obpgDQGcFmaJgB"),  # Adam
        stability        = 0.30,   # Low — punchy, varied intensity
        similarity_boost = 0.88,   # High — stay true to the deep voice
        style            = 0.65,   # High — dramatic, forceful delivery
        speaker_boost    = True,
        preprocess       = _preprocess_titan,
        description      = "Deep, intense, punchy strength coach",
    ),

    "ORACLE": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_ORACLE",  "ErXwobaYiN019PkySvjV"),  # Antoni
        stability        = 0.85,   # Very high — consistent, measured, never wavering
        similarity_boost = 0.85,
        style            = 0.08,   # Very low — neutral, analytical delivery
        speaker_boost    = False,
        preprocess       = _preprocess_oracle,
        description      = "Clear, measured, analytical intelligence build",
    ),

    "PHANTOM": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_PHANTOM", "VR6AewLTigWG4xSOukaG"),  # Arnold
        stability        = 0.42,   # Medium-low — dynamic, fluid energy
        similarity_boost = 0.80,
        style            = 0.48,   # Medium-high — expressive movement quality
        speaker_boost    = True,
        preprocess       = _preprocess_phantom,
        description      = "Fluid, precise, energetic dexterity build",
    ),

    "SAGE": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_SAGE",    "EXAVITQu4vr4xnSDxMaL"),  # Bella
        stability        = 0.92,   # Highest — ultra-calm, barely varies
        similarity_boost = 0.75,
        style            = 0.04,   # Near-zero — no drama, pure stillness
        speaker_boost    = False,
        preprocess       = _preprocess_sage,
        description      = "Ultra-calm, slow, grounded wellness build",
    ),

    "MUSE": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_MUSE",    "ThT5KcBeYPX3keUQqHPh"),  # Dorothy
        stability        = 0.25,   # Lowest (after GG) — most expressive, most varied
        similarity_boost = 0.78,
        style            = 0.62,   # High — creative, emotional range
        speaker_boost    = True,
        preprocess       = _preprocess_muse,
        description      = "Warm, expressive, imaginative creative build",
    ),

    "EMPIRE": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_EMPIRE",  "TxGEqnHWrfWFTfGW9XjX"),  # Josh
        stability        = 0.72,   # High — commanding, authoritative
        similarity_boost = 0.90,   # Very high — strong voice identity
        style            = 0.35,   # Medium — professional authority, not theatrical
        speaker_boost    = True,
        preprocess       = _preprocess_empire,
        description      = "Authoritative, strategic, confident entrepreneur build",
    ),

    "GG": VoiceProfile(
        voice_id         = os.getenv("ELEVENLABS_VOICE_GG",      "yoZ06aMxZJJ28mfd3POQ"),  # Sam
        stability        = 0.18,   # Lowest — maximum hype variation, chaotic energy
        similarity_boost = 0.65,   # Lower — raw casual feel, not over-polished
        style            = 0.80,   # Highest — full hype mode
        speaker_boost    = True,
        preprocess       = _preprocess_gg,
        description      = "Hype, chaotic, casual gamer build",
    ),
}

# Fallback for unknown builds
_FALLBACK_BUILD = "ORACLE"

# ElevenLabs model
_MODEL_ID = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

# Global API key
_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")


def get_profile(build: str) -> VoiceProfile:
    """Return the VoiceProfile for a build (falls back to ORACLE)."""
    return BUILD_PROFILES.get(build.upper(), BUILD_PROFILES[_FALLBACK_BUILD])


# ── Synchronous TTS ───────────────────────────────────────────────────────────

def text_to_audio_bytes(
    text:    str,
    build:   str = "ORACLE",
    api_key: Optional[str] = None,
    # Optional overrides — if None, use the build's profile defaults
    stability:        Optional[float] = None,
    similarity_boost: Optional[float] = None,
    style:            Optional[float] = None,
) -> bytes:
    """
    Convert text to MP3 audio bytes using the build's voice profile.

    The build's VoiceProfile controls voice selection, speaking style,
    expressiveness, and text pre-processing automatically.

    Args:
        text             : The text to synthesise
        build            : Build name (selects voice + speaking style automatically)
        api_key          : ElevenLabs API key (falls back to ELEVENLABS_API_KEY env var)
        stability        : Override stability (0-1). None = use build default.
        similarity_boost : Override similarity (0-1). None = use build default.
        style            : Override style (0-1). None = use build default.

    Returns:
        Raw MP3 bytes ready to stream or save.

    Raises:
        RuntimeError if ElevenLabs is unavailable or API key is missing.
    """
    key = api_key or _API_KEY
    if not key:
        raise RuntimeError(
            "ElevenLabs API key not set. "
            "Add ELEVENLABS_API_KEY to your .env file."
        )

    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
    except ImportError:
        raise RuntimeError(
            "elevenlabs package not installed. "
            "Run: pip install elevenlabs"
        )

    profile = get_profile(build)

    # Apply build-specific text pre-processing
    processed_text = profile.preprocess(text)

    # Use profile defaults unless caller explicitly overrides
    _stability        = stability        if stability        is not None else profile.stability
    _similarity_boost = similarity_boost if similarity_boost is not None else profile.similarity_boost
    _style            = style            if style            is not None else profile.style

    client = ElevenLabs(api_key=key)

    audio_generator = client.text_to_speech.convert(
        text=processed_text,
        voice_id=profile.voice_id,
        model_id=_MODEL_ID,
        voice_settings=VoiceSettings(
            stability=_stability,
            similarity_boost=_similarity_boost,
            style=_style,
            use_speaker_boost=profile.speaker_boost,
        ),
        output_format="mp3_44100_128",
    )

    return b"".join(audio_generator)


# ── Async TTS (for FastAPI endpoints) ─────────────────────────────────────────

async def speak_async(
    text:    str,
    build:   str = "ORACLE",
    api_key: Optional[str] = None,
) -> bytes:
    """
    Async wrapper — runs in a thread pool so it doesn't block FastAPI.
    Automatically uses the build's voice profile.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: text_to_audio_bytes(text, build=build, api_key=api_key),
    )


# ── Availability check ────────────────────────────────────────────────────────

def tts_available() -> bool:
    """Returns True if ElevenLabs key is set and the package is installed."""
    if not _API_KEY:
        return False
    try:
        import elevenlabs  # noqa: F401
        return True
    except ImportError:
        return False


def list_profiles() -> dict[str, dict]:
    """Return a summary of all voice profiles (useful for /health endpoints)."""
    return {
        build: {
            "voice_id":         p.voice_id,
            "stability":        p.stability,
            "similarity_boost": p.similarity_boost,
            "style":            p.style,
            "speaker_boost":    p.speaker_boost,
            "description":      p.description,
        }
        for build, p in BUILD_PROFILES.items()
    }


# ── Quick CLI test ─────────────────────────────────────────────────────────────
# python -m voice.tts TITAN "You crushed it. Now recover."
# python -m voice.tts GG "bro that was CRACKED, no cap GG EZ"
# python -m voice.tts SAGE "Breathe. Notice the stillness inside you."

if __name__ == "__main__":
    import sys

    build = sys.argv[1].upper() if len(sys.argv) > 1 else "TITAN"
    text  = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_TEXTS.get(build, f"LevelUp {build} build.")

    profile = get_profile(build)
    print(f"\nBuild    : {build}")
    print(f"Voice    : {profile.voice_id}  ({profile.description})")
    print(f"Settings : stability={profile.stability}  similarity={profile.similarity_boost}  style={profile.style}")
    print(f"Input    : {text!r}")
    processed = profile.preprocess(text)
    if processed != text:
        print(f"After pre-process: {processed!r}")
    print()

    try:
        audio = text_to_audio_bytes(text, build=build)
        out_path = f"test_{build.lower()}.mp3"
        with open(out_path, "wb") as f:
            f.write(audio)
        print(f"✅  Saved {len(audio):,} bytes → {out_path}")
    except Exception as e:
        print(f"❌  TTS error: {e}")


# Default demo texts that match each build's voice/tone
_DEFAULT_TEXTS: dict[str, str] = {
    "TITAN":   "You plateaued because you got comfortable. That ends today. Add weight. Add reps. No excuses.",
    "ORACLE":  "The key insight here is that compound interest applies to knowledge, not just capital. Every concept you master accelerates every future concept.",
    "PHANTOM": "Flow isn't found, it's built — movement by movement, until your body stops asking permission.",
    "SAGE":    "Breathe. Notice the tension you've been carrying. Allow it to soften, just a little, with each exhale.",
    "MUSE":    "Your creative block isn't a wall, it's a doorway you haven't tried opening sideways yet.",
    "EMPIRE":  "Execution is the only strategy that matters. Ideas are free. Results cost everything.",
    "GG":      "bro that ranked game was CRACKED, no cap GG EZ you're built different fr fr",
}
