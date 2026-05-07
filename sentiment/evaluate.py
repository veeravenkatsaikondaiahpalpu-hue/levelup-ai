"""
sentiment/evaluate.py
======================
Standalone evaluation of trained sentiment classifiers.
Prints per-class metrics and a cross-model comparison table.

Usage:
    python -m sentiment.evaluate
"""

import os
import sys
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from sentiment.data_prep import load_jsonl, split, load_vectorizer, LABELS

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "sentiment")
DATA_PATH = os.path.join(ROOT, "data", "raw", "sentiment_dataset.jsonl")


def evaluate_all(data_path: str = DATA_PATH, model_dir: str = MODEL_DIR) -> list[dict]:
    """
    Evaluate all saved classifiers on the held-out test split.

    Returns list of dicts: {name, accuracy, macro_f1, per_class}.
    """
    print(f"\n{'='*64}")
    print(f"  LevelUp Sentiment Classifier - Evaluation")
    print(f"{'='*64}")
    print(f"  Data   : {data_path}")
    print(f"  Models : {model_dir}\n")

    dual_p   = os.path.join(model_dir, "dual_vectorizer.joblib")
    legacy_p = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    if not os.path.exists(dual_p) and not os.path.exists(legacy_p):
        print("ERROR: vectoriser not found. Run training first.")
        return []

    texts, labels = load_jsonl(data_path)
    _, test_texts, _, y_test = split(texts, labels)
    vec    = joblib.load(dual_p if os.path.exists(dual_p) else legacy_p)
    X_test = vec.transform(test_texts)

    n_per_class = {LABELS[i]: int((y_test == i).sum()) for i in range(3)}
    print(f"  Test samples: {len(y_test):,}  |  {n_per_class}\n")

    model_map = [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Linear SVC",          "linear_svc.joblib"),
        ("XGBoost",             "xgboost.joblib"),
        ("LightGBM",            "lightgbm.joblib"),
        ("Voting Ensemble",     "voting_ensemble.joblib"),
        ("Stacking Ensemble",   "stacking_ensemble.joblib"),
    ]

    summary = []

    for display_name, fname in model_map:
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP {display_name} - {fname} not found")
            continue

        clf    = joblib.load(path)
        y_pred = clf.predict(X_test)

        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        wtd_f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report   = classification_report(
            y_test, y_pred, target_names=LABELS, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"{'='*64}")
        print(f"  {display_name}")
        print(f"  Accuracy={acc:.4f}  MacroF1={macro_f1:.4f}  WeightedF1={wtd_f1:.4f}")

        print(f"\n  Per-class F1:")
        for lbl in LABELS:
            f1  = report[lbl]["f1-score"]
            prec = report[lbl]["precision"]
            rec  = report[lbl]["recall"]
            print(f"    {lbl:<12}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        header = "".join(f"{l:>12}" for l in LABELS)
        print(f"  {'':>12}{header}")
        for i, row in enumerate(cm):
            vals = "".join(f"{v:>12}" for v in row)
            print(f"  {LABELS[i]:>12}{vals}")

        print(f"\n  Full report:")
        print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

        summary.append({
            "name":      display_name,
            "accuracy":  round(acc, 4),
            "macro_f1":  round(macro_f1, 4),
            "per_class": {lbl: round(report[lbl]["f1-score"], 4) for lbl in LABELS},
        })

    if summary:
        print(f"{'='*64}")
        print("  COMPARISON TABLE")
        print(f"  {'Model':<22} {'Acc':>7} {'MacroF1':>9} {'neutral':>9} {'motivated':>11} {'struggling':>11}")
        print(f"  {'-'*71}")
        for row in summary:
            pc = row["per_class"]
            print(f"  {row['name']:<22} "
                  f"{row['accuracy']:>7.4f} "
                  f"{row['macro_f1']:>9.4f} "
                  f"{pc['neutral']:>9.4f} "
                  f"{pc['motivated']:>11.4f} "
                  f"{pc['struggling']:>11.4f}")
        best = max(summary, key=lambda r: r["macro_f1"])
        print(f"\n  Best MacroF1 -> {best['name']} ({best['macro_f1']:.4f})")
        print(f"{'='*64}\n")

    return summary


if __name__ == "__main__":
    data_path = DATA_PATH
    model_dir = MODEL_DIR
    args = sys.argv[1:]
    if "--data"   in args: data_path = args[args.index("--data")   + 1]
    if "--models" in args: model_dir = args[args.index("--models") + 1]
    evaluate_all(data_path=data_path, model_dir=model_dir)
