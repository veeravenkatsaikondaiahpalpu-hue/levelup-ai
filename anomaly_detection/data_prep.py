"""
anomaly_detection/data_prep.py
================================
Data loading and preprocessing for the anomaly detection pipeline.

Loads the anomaly CSV, extracts the 8 numeric features, applies
StandardScaler, and handles serialisation so the same scaler can
be reloaded at inference time.
"""

import os
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ── Feature and label definitions ────────────────────────────────────────────

FEATURES = [
    "activities_per_day",
    "daily_xp_total",
    "streak_gap_days",
    "intensity_switch_rate",
    "avg_session_duration",
    "max_session_duration",
    "sessions_at_cap_ratio",
    "xp_per_minute",
]

ANOMALY_TYPES = ["none", "xp_grinding", "impossible_streak", "intensity_spoofing"]
TYPE_TO_INT   = {t: i for i, t in enumerate(ANOMALY_TYPES)}
INT_TO_TYPE   = {i: t for t, i in TYPE_TO_INT.items()}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an anomaly CSV and return (X, y_binary, y_type_int).

    Args:
        path: Path to the CSV file (train or test split).

    Returns:
        X          — shape (n, 8), float32 feature matrix
        y_binary   — shape (n,),   int32  0=normal 1=anomaly
        y_type_int — shape (n,),   int32  0..3 → ANOMALY_TYPES index
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    X      = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    y_bin  = np.array([int(r["is_anomaly"])              for r in rows], dtype=np.int32)
    y_type = np.array([TYPE_TO_INT[r["anomaly_type"]]    for r in rows], dtype=np.int32)

    return X, y_bin, y_type


# ── Scaler helpers ────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray, save_path: str | None = None) -> StandardScaler:
    """
    Fit a StandardScaler on training features and optionally save it.

    Args:
        X_train:   Training feature matrix (n, 8).
        save_path: If given, persist scaler with joblib.

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path: str) -> StandardScaler:
    """Load a previously saved StandardScaler from disk."""
    return joblib.load(path)


# ── Inference helper ──────────────────────────────────────────────────────────

def features_from_dict(d: dict) -> np.ndarray:
    """
    Convert a feature dict (e.g. from an API request) to a (1, 8) numpy array,
    ordering columns to match FEATURES.

    Args:
        d: Dict with keys matching FEATURES.

    Returns:
        Shape (1, 8) float32 array ready for scaler.transform().
    """
    return np.array([[float(d[f]) for f in FEATURES]], dtype=np.float32)
