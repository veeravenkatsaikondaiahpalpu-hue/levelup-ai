"""
sentiment/
===========
User-message sentiment analysis for LevelUp AI.

Six classifiers trained on 24,000 balanced samples (dual TF-IDF features):
  - Logistic Regression  (interpretable baseline)
  - Linear SVC           (max-margin, calibrated)
  - XGBoost              (proper gradient boosting)
  - LightGBM             (Microsoft boosting — fast, strong)
  - Voting Ensemble      (soft-vote: LR + SVC + LGB)
  - Stacking Ensemble    (primary — LR + SVC + LGB base -> LR meta)

Labels:
  neutral    — matter-of-fact, mixed, or no strong emotion
  motivated  — energised, positive, driven  -> chatbot enters hype_mode
  struggling — low, negative, overwhelmed   -> chatbot enters support_mode

Quick usage::

    from sentiment.models import get_analyzer
    a = get_analyzer()
    result = a.predict("I've been struggling to stay consistent lately")
    # -> {"label": "struggling", "confidence": 0.87, "flag": "support_mode", ...}

Training::

    python -m sentiment.train

Evaluation::

    python -m sentiment.evaluate
"""

from sentiment.models    import SentimentAnalyzer, get_analyzer
from sentiment.data_prep import LABELS, LABEL_TO_INT, INT_TO_LABEL

__all__ = [
    "SentimentAnalyzer",
    "get_analyzer",
    "LABELS",
    "LABEL_TO_INT",
    "INT_TO_LABEL",
]
