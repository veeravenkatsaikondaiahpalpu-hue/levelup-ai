"""
anomaly_detection/train.py
===========================
Entry point: train all three anomaly detection classifiers, then
immediately evaluate them on the test split.

Usage:
    python -m anomaly_detection.train
"""

import os
import sys

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "data", "raw", "anomaly_train.csv")
TEST_CSV  = os.path.join(ROOT, "data", "raw", "anomaly_test.csv")
MODEL_DIR = os.path.join(ROOT, "models", "anomaly")


if __name__ == "__main__":
    # Allow overriding paths via CLI
    args = sys.argv[1:]
    if "--train" in args:
        TRAIN_CSV = args[args.index("--train") + 1]
    if "--test" in args:
        TEST_CSV = args[args.index("--test") + 1]
    if "--out" in args:
        MODEL_DIR = args[args.index("--out") + 1]

    print("=" * 62)
    print("  LevelUp Anomaly Detection -Training Pipeline")
    print("=" * 62)
    print(f"  Train : {TRAIN_CSV}")
    print(f"  Test  : {TEST_CSV}")
    print(f"  Out   : {MODEL_DIR}\n")

    # -- Train -------------------------------------------------------------
    from anomaly_detection.models import train
    results = train(TRAIN_CSV, TEST_CSV, MODEL_DIR)

    # -- Summary -----------------------------------------------------------
    print("\n" + "=" * 62)
    print("  Training complete -artefacts saved:")
    print(f"    {MODEL_DIR}/scaler.joblib")
    print(f"    {MODEL_DIR}/logistic_regression.joblib")
    print(f"    {MODEL_DIR}/decision_tree.joblib")
    print(f"    {MODEL_DIR}/random_forest.joblib")
    print("\n  Training metrics on test split:")
    print(f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*55}")
    for model, m in results.items():
        print(f"  {model:<25} "
              f"{m['accuracy']:>7.4f} "
              f"{m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} "
              f"{m['f1']:>7.4f}")
    print()

    # -- Detailed evaluation -----------------------------------------------
    from anomaly_detection.evaluate import evaluate_all
    evaluate_all(test_csv=TEST_CSV, model_dir=MODEL_DIR)
