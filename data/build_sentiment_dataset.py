"""
data/build_sentiment_dataset.py
================================
Merge all collected sentiment/emotion/mental-health datasets into a
single unified 3-class JSONL file for training the LevelUp sentiment classifier.

Target schema:
    {"text": "...", "label": "motivated" | "neutral" | "struggling"}

Sources merged:
  1. data/raw/sentiment_dataset.jsonl       — original 2,508 motivational quotes
  2. data/raw/sentiment_extra/emotion_dair.jsonl      — 20,000 rows (sadness/joy/love/anger/fear/surprise)
  3. data/raw/sentiment_extra/go_emotions.jsonl       — 54,263 rows (28 fine-grained labels)
  4. data/raw/sentiment_extra/dreaddit_stress.jsonl   — 3,553 rows (binary stress from Reddit)
  5. data/raw/sentiment_extra/mentalchat16k.jsonl     — 16,057 user inputs from counseling sessions

Label mapping strategy
-----------------------
dair-ai/emotion:
    sadness  -> struggling     (explicitly negative state)
    fear     -> struggling     (anxiety, dread)
    anger    -> struggling     (frustration, often co-occurs with struggle)
    joy      -> motivated      (positive energy)
    love     -> neutral        (affection, not performance-related)
    surprise -> neutral        (ambiguous valence)

go_emotions (28 labels):
    Struggling cluster:
        sadness, grief, fear, nervousness, disappointment, remorse,
        embarrassment, disgust, annoyance, disapproval
    Motivated cluster:
        admiration, amusement, excitement, gratitude, joy, love,
        optimism, pride, relief
    Neutral:
        neutral, curiosity, realization, surprise, confusion,
        caring, approval, desire

dreaddit:
    label=1 (stressed)  -> struggling
    label=0 (not stressed) -> neutral

mentalchat16k:
    All user inputs describe mental health struggles -> struggling
    (These are people seeking counseling — very high-quality struggling examples)

Usage:
    python -m data.build_sentiment_dataset
"""

import json
import os
import random
from collections import Counter

random.seed(42)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "raw")
EXTRA   = os.path.join(ROOT, "data", "raw", "sentiment_extra")

# ── Label maps ────────────────────────────────────────────────────────────────

DAIR_MAP = {
    "sadness":  "struggling",
    "fear":     "struggling",
    "anger":    "struggling",
    "joy":      "motivated",
    "love":     "neutral",
    "surprise": "neutral",
}

GO_STRUGGLING = {
    "sadness", "grief", "fear", "nervousness", "disappointment",
    "remorse", "embarrassment", "disgust", "annoyance", "disapproval",
}
GO_MOTIVATED = {
    "admiration", "amusement", "excitement", "gratitude", "joy",
    "love", "optimism", "pride", "relief",
}


# ── Source loaders ────────────────────────────────────────────────────────────

def load_original(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"text": obj["text"], "label": obj["label"], "source": "original"})
    return rows


def load_dair(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = DAIR_MAP.get(obj["emotion_label"], "neutral")
            rows.append({"text": obj["text"], "label": label, "source": "dair_emotion"})
    return rows


def load_go_emotions(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj    = json.loads(line)
            labels = set(obj.get("labels", []))

            # Determine dominant label by priority
            if labels & GO_STRUGGLING:
                label = "struggling"
            elif labels & GO_MOTIVATED:
                label = "motivated"
            else:
                label = "neutral"

            text = obj["text"].strip()
            if len(text) >= 10:
                rows.append({"text": text, "label": label, "source": "go_emotions"})
    return rows


def load_dreaddit(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj   = json.loads(line)
            label = "struggling" if obj["is_stressed"] == 1 else "neutral"
            text  = obj["text"].strip()
            # Dreaddit texts can be very long — cap at 500 chars
            if len(text) >= 20:
                rows.append({"text": text[:500], "label": label, "source": "dreaddit"})
    return rows


def load_mentalchat(path: str) -> list[dict]:
    """All mentalchat user inputs are mental health struggles."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            text = obj["text"].strip()
            if len(text) >= 20:
                rows.append({"text": text[:400], "label": "struggling", "source": "mentalchat"})
    return rows


# ── Balancer ──────────────────────────────────────────────────────────────────

def balance(rows: list[dict], target_per_class: int = 8000) -> list[dict]:
    """
    Downsample over-represented classes to target_per_class.
    Keeps all samples if a class has fewer than target_per_class.
    """
    by_label: dict[str, list] = {"motivated": [], "neutral": [], "struggling": []}
    for r in rows:
        if r["label"] in by_label:
            by_label[r["label"]].append(r)

    balanced = []
    for label, samples in by_label.items():
        if len(samples) > target_per_class:
            samples = random.sample(samples, target_per_class)
        balanced.extend(samples)

    random.shuffle(balanced)
    return balanced


# ── Main ──────────────────────────────────────────────────────────────────────

def build(
    target_per_class: int = 8000,
    output_path: str | None = None,
) -> str:
    if output_path is None:
        output_path = os.path.join(OUT_DIR, "sentiment_dataset_v2.jsonl")

    all_rows: list[dict] = []

    # 1. Original quotes dataset
    orig_path = os.path.join(OUT_DIR, "sentiment_dataset.jsonl")
    if os.path.exists(orig_path):
        rows = load_original(orig_path)
        all_rows.extend(rows)
        print(f"  Original quotes     : {len(rows):>6,} rows")

    # 2. dair-ai/emotion
    dair_path = os.path.join(EXTRA, "emotion_dair.jsonl")
    if os.path.exists(dair_path):
        rows = load_dair(dair_path)
        all_rows.extend(rows)
        print(f"  dair-ai/emotion     : {len(rows):>6,} rows")

    # 3. go_emotions
    go_path = os.path.join(EXTRA, "go_emotions.jsonl")
    if os.path.exists(go_path):
        rows = load_go_emotions(go_path)
        all_rows.extend(rows)
        print(f"  go_emotions         : {len(rows):>6,} rows")

    # 4. dreaddit
    dread_path = os.path.join(EXTRA, "dreaddit_stress.jsonl")
    if os.path.exists(dread_path):
        rows = load_dreaddit(dread_path)
        all_rows.extend(rows)
        print(f"  dreaddit (stress)   : {len(rows):>6,} rows")

    # 5. MentalChat16K
    mc_path = os.path.join(EXTRA, "mentalchat16k.jsonl")
    if os.path.exists(mc_path):
        rows = load_mentalchat(mc_path)
        all_rows.extend(rows)
        print(f"  MentalChat16K       : {len(rows):>6,} rows")

    # Stats before balancing
    counts = Counter(r["label"] for r in all_rows)
    print(f"\n  Before balancing: {dict(counts)}")
    print(f"  Total: {len(all_rows):,}")

    # Balance
    balanced = balance(all_rows, target_per_class=target_per_class)
    counts_b = Counter(r["label"] for r in balanced)
    print(f"\n  After balancing ({target_per_class}/class max): {dict(counts_b)}")
    print(f"  Total: {len(balanced):,}")

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

    print(f"\n  Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("  LevelUp Sentiment Dataset Builder")
    print("=" * 60)
    path = build(target_per_class=8000)
    print(f"\nDone. Use this file for training:")
    print(f"  python -m sentiment.train --data {path}")
