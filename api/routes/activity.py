"""
api/routes/activity.py
=======================
Activity anomaly detection endpoint.

POST /api/activity/check
    Body  : ActivityLog (8 numeric features matching the training schema)
    Return: AnomalyResult (is_anomaly, confidence, anomaly_type, per-model scores, latency)

GET /api/activity/health
    Return: model load status and feature list

GET /api/activity/demo
    Return: example predictions for each anomaly type (no model required, uses raw rules)
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/activity", tags=["activity"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ActivityLog(BaseModel):
    """One day's aggregated activity statistics for a single user."""
    activities_per_day:    int   = Field(..., ge=0,        description="Number of sessions logged today")
    daily_xp_total:        float = Field(..., ge=0,        description="Total XP earned today")
    streak_gap_days:       int   = Field(..., ge=0,        description="Days elapsed since last activity")
    intensity_switch_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of consecutive sessions that changed intensity")
    avg_session_duration:  float = Field(..., ge=0,        description="Mean session duration (minutes)")
    max_session_duration:  float = Field(..., ge=0,        description="Longest single session today (minutes)")
    sessions_at_cap_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of days where daily XP hit the cap")
    xp_per_minute:         float = Field(..., ge=0,        description="XP earned per active minute")

    model_config = {"json_schema_extra": {"example": {
        "activities_per_day":    12,
        "daily_xp_total":        720.0,
        "streak_gap_days":       0,
        "intensity_switch_rate": 1.0,
        "avg_session_duration":  20.0,
        "max_session_duration":  30.0,
        "sessions_at_cap_ratio": 1.0,
        "xp_per_minute":         2.4,
    }}}


class ModelScores(BaseModel):
    logistic_regression: Optional[float] = Field(None, description="Anomaly probability from Logistic Regression")
    decision_tree:       Optional[float] = Field(None, description="Anomaly probability from Decision Tree")
    random_forest:       Optional[float] = Field(None, description="Anomaly probability from Random Forest")


class AnomalyResult(BaseModel):
    is_anomaly:   bool
    confidence:   float          = Field(description="Primary model's anomaly probability (0–1)")
    anomaly_type: Optional[str]  = Field(None, description="xp_grinding | intensity_spoofing | impossible_streak | null")
    scores:       ModelScores
    latency_ms:   float


# ── Lazy detector loader ──────────────────────────────────────────────────────

_detector = None


def _get_detector():
    """
    Lazy-load the AnomalyDetector singleton.
    Raises HTTP 503 if models have not been trained yet.
    """
    global _detector
    if _detector is None:
        try:
            from anomaly_detection.models import AnomalyDetector
            _detector = AnomalyDetector()
            _detector.load()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Anomaly detection models not found. "
                    f"Run `python -m anomaly_detection.train` first. ({e})"
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not load anomaly detector: {e}",
            )
    return _detector


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/check",
    response_model=AnomalyResult,
    summary="Detect XP cheating / anomalous activity",
)
async def check_activity(log: ActivityLog):
    """
    Analyse one day's activity statistics and flag potential XP abuse.

    ### Anomaly types detected
    | Type | Description |
    |---|---|
    | `xp_grinding` | Suspiciously many sessions (≥8/day) or single session >300 min |
    | `intensity_spoofing` | Rapid intensity toggling (rate ≥0.85) to inflate XP multiplier |
    | `impossible_streak` | Streak counter advanced despite a 3+ day gap in logs |

    ### Models used
    Primary: **Random Forest** (150 trees, balanced class weights).
    Secondary scores from Logistic Regression and Decision Tree are also returned
    for transparency / comparison.

    ### Returns
    - `is_anomaly` — `true` if the primary model predicts cheating
    - `confidence` — anomaly probability [0, 1]
    - `anomaly_type` — one of the three types above, or `null` for normal
    - `scores` — per-model probabilities for comparison
    - `latency_ms` — server-side inference time
    """
    t0       = time.perf_counter()
    detector = _get_detector()
    result   = detector.predict(log.model_dump())
    ms       = (time.perf_counter() - t0) * 1000

    return AnomalyResult(
        is_anomaly   = result["is_anomaly"],
        confidence   = result["confidence"],
        anomaly_type = result["anomaly_type"],
        scores       = ModelScores(**{
            k: result["scores"].get(k) for k in
            ["logistic_regression", "decision_tree", "random_forest"]
        }),
        latency_ms   = round(ms, 2),
    )


@router.get("/health", summary="Anomaly detector readiness check")
async def activity_health():
    """
    Returns whether the anomaly detection models are loaded and ready,
    plus the list of expected input features.
    """
    try:
        det    = _get_detector()
        loaded = True
        models = {
            "logistic_regression": det.lr is not None,
            "decision_tree":       det.dt is not None,
            "random_forest":       det.rf is not None,
        }
    except HTTPException:
        loaded = False
        models = {}

    return {
        "status":   "ok" if loaded else "unavailable",
        "loaded":   loaded,
        "models":   models,
        "features": [
            "activities_per_day",    "daily_xp_total",
            "streak_gap_days",       "intensity_switch_rate",
            "avg_session_duration",  "max_session_duration",
            "sessions_at_cap_ratio", "xp_per_minute",
        ],
    }


@router.get("/demo", summary="Demo predictions for each anomaly type")
async def activity_demo():
    """
    Returns example anomaly detector outputs for all three cheat types
    plus a normal user — useful for testing without a real app.
    """
    examples = {
        "normal_user": {
            "activities_per_day": 2, "daily_xp_total": 140.0,
            "streak_gap_days": 1, "intensity_switch_rate": 0.0,
            "avg_session_duration": 45.0, "max_session_duration": 60.0,
            "sessions_at_cap_ratio": 0.0, "xp_per_minute": 1.2,
        },
        "xp_grinder": {
            "activities_per_day": 12, "daily_xp_total": 780.0,
            "streak_gap_days": 0, "intensity_switch_rate": 0.4,
            "avg_session_duration": 22.0, "max_session_duration": 35.0,
            "sessions_at_cap_ratio": 1.0, "xp_per_minute": 2.95,
        },
        "intensity_spoofer": {
            "activities_per_day": 6, "daily_xp_total": 360.0,
            "streak_gap_days": 0, "intensity_switch_rate": 1.0,
            "avg_session_duration": 18.0, "max_session_duration": 25.0,
            "sessions_at_cap_ratio": 0.5, "xp_per_minute": 3.33,
        },
        "impossible_streak": {
            "activities_per_day": 1, "daily_xp_total": 15.0,
            "streak_gap_days": 5, "intensity_switch_rate": 0.0,
            "avg_session_duration": 10.0, "max_session_duration": 10.0,
            "sessions_at_cap_ratio": 0.0, "xp_per_minute": 1.0,
        },
    }

    try:
        detector = _get_detector()
        results  = {}
        for label, feats in examples.items():
            res = detector.predict(feats)
            results[label] = {
                "input":        feats,
                "is_anomaly":   res["is_anomaly"],
                "confidence":   res["confidence"],
                "anomaly_type": res["anomaly_type"],
                "scores":       res["scores"],
            }
        return {"status": "ok", "predictions": results}
    except HTTPException as e:
        # Models not trained yet — return the example inputs without predictions
        return {
            "status":  "models_not_loaded",
            "detail":  e.detail,
            "examples": examples,
        }
