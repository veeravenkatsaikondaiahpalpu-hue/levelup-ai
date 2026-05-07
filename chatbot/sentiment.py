"""
chatbot/sentiment.py
=====================
Thin bridge between the chatbot layer and the trained sentiment classifiers.

This module is imported by api/routes/chat.py to detect the emotional tone
of a user's message BEFORE generating the AI reply — so the system prompt
can be adjusted to support_mode or hype_mode accordingly.

Usage::

    from chatbot.sentiment import predict, predict_full, get_flag

    label = predict("I feel like giving up today")
    # -> "struggling"

    flag = get_flag("I crushed my workout! New PR!")
    # -> "hype_mode"

    full = predict_full("I've been so exhausted lately")
    # -> {
    #      "label": "struggling",
    #      "confidence": 0.84,
    #      "flag": "support_mode",
    #      "scores": {"neutral": 0.08, "motivated": 0.08, "struggling": 0.84}
    #    }

Labels:
    "neutral"    -- no strong emotion detected -> standard coaching tone
    "motivated"  -- high energy, positive      -> hype_mode (celebrate + push)
    "struggling" -- low, negative, overwhelmed -> support_mode (empathy first)

These map directly to the [ALERT] blocks in chatbot/system_prompt.py and to
the chatbot tone modes: support_mode / hype_mode / None.
"""

from __future__ import annotations

from typing import Optional

# ── Lazy singleton ─────────────────────────────────────────────────────────────
# We delay importing SentimentAnalyzer until first call so the chatbot process
# doesn't fail to start if the sentiment models haven't been trained yet.

_analyzer = None


def _get_analyzer():
    """Return the module-level SentimentAnalyzer singleton (lazy-loaded)."""
    global _analyzer
    if _analyzer is None:
        from sentiment.models import get_analyzer
        _analyzer = get_analyzer()
    return _analyzer


# ── Public API ────────────────────────────────────────────────────────────────

def predict(text: str) -> str:
    """
    Classify the emotional tone of a user message.

    Args:
        text: Raw user message string.

    Returns:
        One of: ``"neutral"`` | ``"motivated"`` | ``"struggling"``

    Raises:
        Exception: If models are not trained yet. The caller (chat.py) wraps
                   this in a try/except and falls back to sentiment=None.

    Example::

        label = predict("I've been skipping workouts, feeling really behind")
        # -> "struggling"
    """
    result = _get_analyzer().predict(text)
    return result["label"]


def predict_full(text: str) -> dict:
    """
    Full sentiment analysis result with label, confidence, scores, and flag.

    Args:
        text: Raw user message string.

    Returns::

        {
            "label":      "neutral" | "motivated" | "struggling",
            "confidence": float,           # 0.0 – 1.0
            "flag":       "support_mode" | "hype_mode" | None,
            "scores": {
                "neutral":    float,
                "motivated":  float,
                "struggling": float,
            }
        }

    Example::

        r = predict_full("Just finished my session. Nothing special.")
        # -> {"label": "neutral", "confidence": 0.83, "flag": None, ...}
    """
    result = _get_analyzer().predict(text)
    return {
        "label":      result["label"],
        "confidence": result["confidence"],
        "flag":       result.get("flag"),
        "scores":     result["scores"],
    }


def get_flag(text: str) -> Optional[str]:
    """
    Return only the chatbot tone flag for a message.

    Args:
        text: Raw user message string.

    Returns:
        ``"support_mode"`` if user is struggling (conf >= 0.45),
        ``"hype_mode"`` if user is motivated (conf >= 0.50),
        ``None`` for neutral or low-confidence cases.

    Example::

        flag = get_flag("I feel exhausted and burned out")
        # -> "support_mode"

        flag = get_flag("Beast mode! New personal record!")
        # -> "hype_mode"

        flag = get_flag("Logged 3 sets of squats")
        # -> None
    """
    result = _get_analyzer().predict(text)
    return result.get("flag")


def is_struggling(text: str) -> bool:
    """Quick boolean check — True if user's message signals distress."""
    return predict(text) == "struggling"


def is_motivated(text: str) -> bool:
    """Quick boolean check — True if user's message signals high energy."""
    return predict(text) == "motivated"
