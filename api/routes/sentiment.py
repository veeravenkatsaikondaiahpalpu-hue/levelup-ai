"""
api/routes/sentiment.py
========================
Sentiment analysis endpoint for LevelUp AI.

POST /api/sentiment/analyze
    Body  : {"text": "..."}
    Return: SentimentResult (label, confidence, scores, flag, all_models)

POST /api/sentiment/batch
    Body  : {"texts": ["...", "...", ...]}
    Return: list of SentimentResult

GET /api/sentiment/health
    Return: model load status

The "flag" field is what the chatbot layer uses:
    "support_mode" -> switch to empathetic, supportive tone
    "hype_mode"    -> switch to energetic, celebratory tone
    null           -> default coaching tone
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="User message to analyse")

    model_config = {"json_schema_extra": {"example": {
        "text": "I've been really struggling to stay consistent with my workouts lately"
    }}}


class BatchInput(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=50,
                              description="List of texts to analyse (max 50)")


class ModelPrediction(BaseModel):
    label:      str
    confidence: float


class AllModelScores(BaseModel):
    logistic_regression: Optional[ModelPrediction] = None
    linear_svc:          Optional[ModelPrediction] = None
    xgboost:             Optional[ModelPrediction] = None
    lightgbm:            Optional[ModelPrediction] = None
    voting_ensemble:     Optional[ModelPrediction] = None
    stacking_ensemble:   Optional[ModelPrediction] = None


class SentimentResult(BaseModel):
    label:      str   = Field(description="neutral | motivated | struggling")
    confidence: float = Field(description="Primary model's probability for predicted label")
    scores:     dict  = Field(description="Per-class probabilities {neutral, motivated, struggling}")
    flag:       Optional[str] = Field(
        None,
        description="support_mode | hype_mode | null — chatbot tone hint"
    )
    all_models: Optional[AllModelScores] = None
    latency_ms: float


# ── Lazy analyzer loader ──────────────────────────────────────────────────────

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from sentiment.models import SentimentAnalyzer
            _analyzer = SentimentAnalyzer()
            _analyzer.load()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Sentiment models not found. "
                    f"Run `python -m sentiment.train` first. ({e})"
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not load sentiment analyzer: {e}",
            )
    return _analyzer


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_result(raw: dict, latency_ms: float) -> SentimentResult:
    all_m = raw.get("all_models", {})
    return SentimentResult(
        label      = raw["label"],
        confidence = raw["confidence"],
        scores     = raw["scores"],
        flag       = raw.get("flag"),
        all_models = AllModelScores(
            **{k: ModelPrediction(**v) for k, v in all_m.items()
               if k in AllModelScores.model_fields}
        ),
        latency_ms = round(latency_ms, 2),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=SentimentResult,
    summary="Analyse sentiment of a user message",
)
async def analyze_sentiment(body: TextInput):
    """
    Classify the emotional tone of a user's message into one of three labels:

    | Label       | Meaning                              | Chatbot response     |
    |-------------|--------------------------------------|----------------------|
    | `motivated` | Energised, positive, driven          | Hype / celebrate     |
    | `neutral`   | Matter-of-fact, no strong emotion    | Standard coaching    |
    | `struggling`| Low, negative, overwhelmed or stuck  | Supportive / empathy |

    ### Fields
    - `flag` — `"support_mode"` or `"hype_mode"` when confidence is high enough,
      otherwise `null`. The app uses this to switch the chatbot's persona tone.
    - `all_models` — predictions from each individual classifier for transparency.

    ### Notes
    The classifier was trained on ~2,500 motivational quotes with TF-IDF +
    Logistic Regression (primary), LinearSVC, and Gradient Boosting.
    The rare `struggling` class was oversampled to 12 % of training data
    to improve recall on the most safety-critical label.
    """
    t0       = time.perf_counter()
    analyzer = _get_analyzer()
    raw      = analyzer.predict(body.text)
    ms       = (time.perf_counter() - t0) * 1000
    return _to_result(raw, ms)


@router.post(
    "/batch",
    response_model=list[SentimentResult],
    summary="Analyse sentiment for up to 50 texts at once",
)
async def analyze_batch(body: BatchInput):
    """
    Batch variant of `/analyze`. Sends up to 50 texts in one request.
    Useful for pre-processing chat history or analysing session logs.
    """
    t0       = time.perf_counter()
    analyzer = _get_analyzer()
    raws     = analyzer.predict_batch(body.texts)
    total_ms = (time.perf_counter() - t0) * 1000
    per_ms   = total_ms / len(raws)
    return [_to_result(r, per_ms) for r in raws]


@router.get("/health", summary="Sentiment analyzer readiness check")
async def sentiment_health():
    """Returns whether the sentiment models are loaded and ready."""
    try:
        a      = _get_analyzer()
        loaded = True
        models = {
            "logistic_regression": a.lr       is not None,
            "linear_svc":          a.svc      is not None,
            "xgboost":             a.xgb      is not None,
            "lightgbm":            a.lgb      is not None,
            "voting_ensemble":     a.voting   is not None,
            "stacking_ensemble":   a.stacking is not None,
        }
        # DualVectorizer exposes word_vec and char_vec sub-vectorizers
        if hasattr(a.vec, "word_vec"):
            vocab = len(a.vec.word_vec.vocabulary_) + len(a.vec.char_vec.vocabulary_)
        elif a.vec is not None:
            vocab = len(a.vec.vocabulary_)
        else:
            vocab = 0
    except HTTPException:
        loaded, models, vocab = False, {}, 0

    return {
        "status":     "ok" if loaded else "unavailable",
        "loaded":     loaded,
        "models":     models,
        "vocab_size": vocab,
        "labels":     ["neutral", "motivated", "struggling"],
        "flags":      ["support_mode", "hype_mode", None],
    }


@router.get("/demo", summary="Demo predictions for each sentiment class")
async def sentiment_demo():
    """
    Returns example predictions for all three sentiment classes.
    Good for testing the endpoint without a real app.
    """
    examples = {
        "motivated":  "I crushed my workout today and beat my PR! Feeling unstoppable!",
        "neutral":    "Completed my study session. Covered chapters 4 through 6.",
        "struggling": "I feel like no matter what I do I keep falling behind. It's exhausting.",
    }

    try:
        analyzer = _get_analyzer()
        t0 = time.perf_counter()
        results = {}
        for label, text in examples.items():
            raw = analyzer.predict(text)
            results[label] = {
                "text":       text,
                "prediction": raw["label"],
                "confidence": raw["confidence"],
                "flag":       raw["flag"],
                "scores":     raw["scores"],
            }
        ms = (time.perf_counter() - t0) * 1000
        return {"status": "ok", "latency_ms": round(ms, 2), "predictions": results}

    except HTTPException as e:
        return {"status": "models_not_loaded", "detail": e.detail, "examples": examples}
