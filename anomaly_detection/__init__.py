"""
anomaly_detection/
==================
XP-cheating / activity anomaly detection for LevelUp AI.

Three classifiers trained on synthetic activity logs:
  • Logistic Regression (interpretable baseline)
  • Decision Tree       (rule-based, visualisable)
  • Random Forest       (primary — best F1)

Quick usage::

    from anomaly_detection.models import get_detector
    det = get_detector()
    result = det.predict({
        "activities_per_day": 12,
        "daily_xp_total": 720.0,
        "streak_gap_days": 0,
        "intensity_switch_rate": 1.0,
        "avg_session_duration": 20.0,
        "max_session_duration": 30,
        "sessions_at_cap_ratio": 1.0,
        "xp_per_minute": 2.4,
    })
    # → {"is_anomaly": True, "confidence": 0.97, "anomaly_type": "intensity_spoofing", ...}

Training::

    python -m anomaly_detection.train

Evaluation::

    python -m anomaly_detection.evaluate
"""

from anomaly_detection.models    import AnomalyDetector, get_detector
from anomaly_detection.data_prep import FEATURES, ANOMALY_TYPES

__all__ = [
    "AnomalyDetector",
    "get_detector",
    "FEATURES",
    "ANOMALY_TYPES",
]
