"""
Thin bridge between the chatbot and our trained sentiment classifiers.

chat.py calls this BEFORE generating the AI reply so the system prompt can be
nudged into support_mode (user is struggling) or hype_mode (user is pumped).
It's a best-effort layer — if the models aren't trained yet the caller falls
back gracefully to sentiment=None and uses the default coaching tone.

Quick usage:
    from chatbot.sentiment import predict, predict_full, get_flag

    label = predict("I feel like giving up today")       # -> "struggling"
    flag  = get_flag("I crushed my workout! New PR!")    # -> "hype_mode"

    full  = predict_full("I've been so exhausted lately")
    # -> {"label": "struggling", "confidence": 0.84, "flag": "support_mode", ...}

Labels and what they mean for the chatbot:
    "neutral"    — no strong emotion, keep the standard coaching tone
    "motivated"  — high energy / positive → hype_mode (celebrate and push harder)
    "struggling" — low / negative / overwhelmed → support_mode (empathy first)
"""

from __future__ import annotations

# The SentimentAnalyzer is loaded lazily on first use — this way the chatbot
# process starts up even if the sentiment models haven't been trained yet.
_analyzer = None


def _get_analyzer():
    """Return the shared SentimentAnalyzer instance, loading it on first call."""
    global _analyzer
    if _analyzer is None:
        from sentiment.models import get_analyzer
        _analyzer = get_analyzer()
    return _analyzer


# Public API

def predict(text: str) -> str:
    """
    Classify the emotional tone of a user message into one of three labels.

    Returns "neutral" | "motivated" | "struggling".
    Raises if the sentiment models haven't been trained yet — chat.py wraps
    this in a try/except and falls back to sentiment=None safely.
    """
    result = _get_analyzer().predict(text)
    return result["label"]


def predict_full(text: str) -> dict:
    """
    Full sentiment result — label, confidence score, per-class probabilities, and tone flag.

    Returns a dict like:
        {
            "label":      "struggling",
            "confidence": 0.84,
            "flag":       "support_mode",
            "scores":     {"neutral": 0.08, "motivated": 0.08, "struggling": 0.84}
        }
    """
    result = _get_analyzer().predict(text)
    return {
        "label":      result["label"],
        "confidence": result["confidence"],
        "flag":       result.get("flag"),
        "scores":     result["scores"],
    }


def get_flag(text: str) -> str | None:
    """
    Return just the chatbot tone flag for a message — the part the chat route actually uses.

    Returns "support_mode" when the user seems to be struggling (confidence >= 0.45),
    "hype_mode" when they're motivated (confidence >= 0.50), or None for neutral / unsure.
    """
    result = _get_analyzer().predict(text)
    return result.get("flag")


def is_struggling(text: str) -> bool:
    """True if the classifier thinks the user is in a low / distressed state."""
    return predict(text) == "struggling"


def is_motivated(text: str) -> bool:
    """True if the classifier thinks the user is fired up and high-energy."""
    return predict(text) == "motivated"
