"""
sentiment/train.py
===================
Entry point: train all three sentiment classifiers, then evaluate.

Usage:
    python -m sentiment.train
    python -m sentiment.train --data data/raw/sentiment_dataset.jsonl
"""

import os
import sys

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "sentiment_dataset.jsonl")
MODEL_DIR = os.path.join(ROOT, "models", "sentiment")

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data"  in args: DATA_PATH = args[args.index("--data")  + 1]
    if "--out"   in args: MODEL_DIR = args[args.index("--out")   + 1]

    print("=" * 64)
    print("  LevelUp Sentiment Analysis - Training Pipeline (6 models)")
    print("=" * 64)
    print(f"  Data : {DATA_PATH}")
    print(f"  Out  : {MODEL_DIR}\n")

    from sentiment.models import train
    results = train(DATA_PATH, MODEL_DIR)

    print("\n" + "=" * 64)
    print("  Training complete - artefacts saved:")
    print(f"    {MODEL_DIR}/dual_vectorizer.joblib")
    print(f"    {MODEL_DIR}/logistic_regression.joblib")
    print(f"    {MODEL_DIR}/linear_svc.joblib")
    print(f"    {MODEL_DIR}/xgboost.joblib")
    print(f"    {MODEL_DIR}/lightgbm.joblib")
    print(f"    {MODEL_DIR}/voting_ensemble.joblib")
    print(f"    {MODEL_DIR}/stacking_ensemble.joblib")
    print("\n  Summary:")
    print(f"  {'Model':<25} {'Acc':>7} {'MacroF1':>9} {'struggling_F1':>14}")
    print(f"  {'-'*57}")
    for model, m in results.items():
        print(f"  {model:<25} "
              f"{m['accuracy']:>7.4f} "
              f"{m['macro_f1']:>9.4f} "
              f"{m['per_class']['struggling']:>14.4f}")
    print()

    from sentiment.evaluate import evaluate_all
    evaluate_all(data_path=DATA_PATH, model_dir=MODEL_DIR)
