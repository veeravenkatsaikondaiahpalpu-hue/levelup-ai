"""
anomaly_detection/evaluate.py
==============================
Standalone evaluation script -runs all three trained classifiers
against the held-out test split and prints a comparison table.

Usage:
    python -m anomaly_detection.evaluate
    python -m anomaly_detection.evaluate --test data/raw/anomaly_test.csv
"""

import os
import sys
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from anomaly_detection.data_prep import load_csv

# -- Defaults ------------------------------------------------------------------

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "anomaly")
TEST_CSV  = os.path.join(ROOT, "data", "raw", "anomaly_test.csv")


# -- Evaluation ----------------------------------------------------------------

def evaluate_all(test_csv: str = TEST_CSV, model_dir: str = MODEL_DIR) -> list[dict]:
    """
    Evaluate all saved classifiers against test_csv.

    Args:
        test_csv:  Path to the test CSV split.
        model_dir: Directory containing joblib artefacts.

    Returns:
        List of dicts {name, accuracy, precision, recall, f1}.
    """
    print(f"\n{'='*62}")
    print(f"  LevelUp Anomaly Detector -Evaluation")
    print(f"{'='*62}")
    print(f"  Test set : {test_csv}")
    print(f"  Models   : {model_dir}\n")

    # -- Load test data ----------------------------------------------------
    scaler_p = os.path.join(model_dir, "scaler.joblib")
    if not os.path.exists(scaler_p):
        print("ERROR: scaler.joblib not found. Run training first.")
        return []

    X_test, y_test, _ = load_csv(test_csv)
    scaler             = joblib.load(scaler_p)
    X_test_s           = scaler.transform(X_test)

    n_pos = int(y_test.sum())
    n_neg = len(y_test) - n_pos
    print(f"  Samples  : {len(y_test):,}  (normal={n_neg:,}, anomaly={n_pos:,})\n")

    # -- Models to evaluate ------------------------------------------------
    model_map = [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Decision Tree",       "decision_tree.joblib"),
        ("Random Forest",       "random_forest.joblib"),
    ]

    summary = []

    for display_name, fname in model_map:
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP {display_name} -{fname} not found")
            continue

        clf    = joblib.load(path)
        y_pred = clf.predict(X_test_s)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        cm    = confusion_matrix(y_test, y_pred)

        print(f"{'-'*62}")
        print(f"  {display_name}")
        print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Normal   Anomaly")
        print(f"  Actual Normal   {cm[0][0]:5d}    {cm[0][1]:5d}   (FP={cm[0][1]})")
        print(f"  Actual Anomaly  {cm[1][0]:5d}    {cm[1][1]:5d}   (FN={cm[1][0]})")
        print(f"\n  Detailed report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["normal", "anomaly"],
            digits=4,
        ))

        summary.append({
            "name":      display_name,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        })

    # -- Summary table -----------------------------------------------------
    if summary:
        print(f"{'='*62}")
        print(f"  COMPARISON TABLE")
        print(f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"  {'-'*57}")
        for row in summary:
            print(f"  {row['name']:<25} "
                  f"{row['accuracy']:>7.4f} "
                  f"{row['precision']:>7.4f} "
                  f"{row['recall']:>7.4f} "
                  f"{row['f1']:>7.4f}")
        best = max(summary, key=lambda r: r["f1"])
        print(f"\n  Best F1 -> {best['name']} ({best['f1']:.4f})")
        print(f"{'='*62}\n")

    return summary


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    test_csv  = TEST_CSV
    model_dir = MODEL_DIR

    args = sys.argv[1:]
    if "--test" in args:
        idx = args.index("--test")
        test_csv = args[idx + 1]
    if "--models" in args:
        idx = args.index("--models")
        model_dir = args[idx + 1]

    evaluate_all(test_csv=test_csv, model_dir=model_dir)
