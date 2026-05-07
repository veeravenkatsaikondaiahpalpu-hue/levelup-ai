"""
anomaly_detection/models.py
============================
Train and serve three anomaly detectors on the LevelUp activity log dataset:
  • Logistic Regression (interpretable baseline)
  • Decision Tree       (rule-based, visualisable)
  • Random Forest       (primary — best accuracy)

All classifiers use class_weight="balanced" to handle the ~20 % anomaly rate.

Training
--------
    from anomaly_detection.models import train
    results = train("data/raw/anomaly_train.csv", "data/raw/anomaly_test.csv")

Inference
---------
    from anomaly_detection.models import get_detector
    det = get_detector()          # lazy singleton
    det.predict({...features...}) # → dict with is_anomaly, confidence, anomaly_type
"""

import os
import numpy as np
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, classification_report

from anomaly_detection.data_prep import (
    load_csv, fit_scaler, features_from_dict, INT_TO_TYPE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "anomaly")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LR_PATH     = os.path.join(MODEL_DIR, "logistic_regression.joblib")
DT_PATH     = os.path.join(MODEL_DIR, "decision_tree.joblib")
RF_PATH     = os.path.join(MODEL_DIR, "random_forest.joblib")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    train_csv: str,
    test_csv:  str,
    model_dir: str = MODEL_DIR,
) -> dict:
    """
    Train Logistic Regression, Decision Tree, and Random Forest classifiers.
    Saves scaler + all three models as joblib files.

    Args:
        train_csv: Path to anomaly_train.csv
        test_csv:  Path to anomaly_test.csv
        model_dir: Directory where artefacts are saved.

    Returns:
        Dict mapping model name → {accuracy, precision, recall, f1}.
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("  Loading data ...")
    X_train, y_train, _ = load_csv(train_csv)
    X_test,  y_test,  _ = load_csv(test_csv)
    print(f"  Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")

    # ── Scaler ─────────────────────────────────────────────────────────────
    print("  Fitting StandardScaler ...")
    scaler     = fit_scaler(X_train, save_path=os.path.join(model_dir, "scaler.joblib"))
    X_train_s  = scaler.transform(X_train)
    X_test_s   = scaler.transform(X_test)

    # ── Model specs ────────────────────────────────────────────────────────
    model_specs = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000, class_weight="balanced",
                C=1.0, solver="lbfgs", random_state=42,
            ),
            os.path.join(model_dir, "logistic_regression.joblib"),
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=8, class_weight="balanced",
                min_samples_leaf=10, random_state=42,
            ),
            os.path.join(model_dir, "decision_tree.joblib"),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                min_samples_leaf=5, random_state=42, n_jobs=-1,
            ),
            os.path.join(model_dir, "random_forest.joblib"),
        ),
    ]

    results = {}

    for name, clf, path in model_specs:
        print(f"\n  Training {name} ...")
        clf.fit(X_train_s, y_train)
        joblib.dump(clf, path)

        y_pred  = clf.predict(X_test_s)
        acc     = accuracy_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy":  round(acc,                              4),
            "precision": round(report["1"]["precision"],         4),
            "recall":    round(report["1"]["recall"],            4),
            "f1":        round(report["1"]["f1-score"],          4),
        }

        print(f"  Saved -> {path}")
        print(f"  Acc={acc:.4f}  "
              f"P={results[name]['precision']:.4f}  "
              f"R={results[name]['recall']:.4f}  "
              f"F1={results[name]['f1']:.4f}")

    return results


# ── Inference ─────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Loads the trained models and exposes a predict() method for single-row
    inference.  Uses Random Forest as the primary classifier.

    Usage::

        det = AnomalyDetector()
        det.load()
        result = det.predict({
            "activities_per_day": 12,
            "daily_xp_total":     720.0,
            "streak_gap_days":    0,
            "intensity_switch_rate": 1.0,
            "avg_session_duration":  20.0,
            "max_session_duration":  30,
            "sessions_at_cap_ratio": 1.0,
            "xp_per_minute":         2.4,
        })
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.scaler: StandardScaler | None            = None
        self.rf: RandomForestClassifier | None        = None
        self.lr: LogisticRegression | None            = None
        self.dt: DecisionTreeClassifier | None        = None

    def load(self) -> None:
        """Load scaler and all available classifier artefacts from model_dir."""
        scaler_p = os.path.join(self.model_dir, "scaler.joblib")
        if not os.path.exists(scaler_p):
            raise FileNotFoundError(
                f"Scaler not found at {scaler_p}. "
                "Run `python -m anomaly_detection.train` first."
            )

        self.scaler = joblib.load(scaler_p)
        self.rf     = joblib.load(os.path.join(self.model_dir, "random_forest.joblib")) \
                      if os.path.exists(os.path.join(self.model_dir, "random_forest.joblib")) else None
        self.lr     = joblib.load(os.path.join(self.model_dir, "logistic_regression.joblib")) \
                      if os.path.exists(os.path.join(self.model_dir, "logistic_regression.joblib")) else None
        self.dt     = joblib.load(os.path.join(self.model_dir, "decision_tree.joblib")) \
                      if os.path.exists(os.path.join(self.model_dir, "decision_tree.joblib")) else None

        if self._primary is None:
            raise RuntimeError("No classifier found in model_dir. Run training first.")

    @property
    def _primary(self):
        """Return the best available classifier (RF > LR > DT)."""
        return self.rf or self.lr or self.dt

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        Predict whether a single day's activity log is anomalous.

        Args:
            features: Dict with exactly the 8 feature keys.

        Returns:
            {
              "is_anomaly":   bool,
              "confidence":   float,   # primary model's anomaly probability
              "anomaly_type": str | None,
              "scores": {
                  "logistic_regression": float | None,
                  "decision_tree":       float | None,
                  "random_forest":       float | None,
              }
            }
        """
        X   = features_from_dict(features)
        X_s = self.scaler.transform(X)

        scores: dict[str, float] = {}
        for key, clf in [
            ("logistic_regression", self.lr),
            ("decision_tree",       self.dt),
            ("random_forest",       self.rf),
        ]:
            if clf is not None:
                scores[key] = round(float(clf.predict_proba(X_s)[0][1]), 4)

        prob       = float(self._primary.predict_proba(X_s)[0][1])
        is_anomaly = prob >= 0.5

        return {
            "is_anomaly":   is_anomaly,
            "confidence":   round(prob, 4),
            "anomaly_type": _infer_type(features) if is_anomaly else None,
            "scores":       scores,
        }

    def predict_all_models(self, features: dict) -> dict:
        """
        Return is_anomaly + confidence from every loaded model independently.
        Useful for demo / comparison views.
        """
        X   = features_from_dict(features)
        X_s = self.scaler.transform(X)
        out = {}
        for name, clf in [
            ("logistic_regression", self.lr),
            ("decision_tree",       self.dt),
            ("random_forest",       self.rf),
        ]:
            if clf is None:
                continue
            prob = float(clf.predict_proba(X_s)[0][1])
            out[name] = {"is_anomaly": prob >= 0.5, "confidence": round(prob, 4)}
        return out


# ── Anomaly type heuristic ────────────────────────────────────────────────────

def _infer_type(features: dict) -> str:
    """
    Rule-based labelling of the detected anomaly.
    Applied only when the ML model already predicts is_anomaly=True.

    Rules match the generation logic in generate_anomaly_data.py:
      impossible_streak  — streak_gap_days >= 3
      intensity_spoofing — intensity_switch_rate >= 0.85
      xp_grinding        — activities_per_day >= 8 OR max_session >= 300 OR cap_ratio == 1
    """
    gap  = float(features.get("streak_gap_days",       0))
    isr  = float(features.get("intensity_switch_rate", 0))
    apd  = float(features.get("activities_per_day",    0))
    msd  = float(features.get("max_session_duration",  0))
    sacr = float(features.get("sessions_at_cap_ratio", 0))

    if gap >= 3:
        return "impossible_streak"
    if isr >= 0.85:
        return "intensity_spoofing"
    if apd >= 8 or msd >= 300 or sacr >= 1.0:
        return "xp_grinding"
    # softer fallbacks
    if isr > 0.5:
        return "intensity_spoofing"
    return "xp_grinding"


# ── Module-level singleton ────────────────────────────────────────────────────

_detector: AnomalyDetector | None = None


def get_detector() -> AnomalyDetector:
    """
    Return the module-level singleton AnomalyDetector (lazy-loaded).
    Thread-safe for single-worker FastAPI / uvicorn usage.
    """
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
        _detector.load()
    return _detector
