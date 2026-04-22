"""
dataset_prep.py
===============
Load and format the LevelUp JSONL training data into a HuggingFace Dataset
ready for SFTTrainer.

Each JSONL sample has:  {"system": "...", "user": "...", "assistant": "..."}

We convert each to the LLaMA 3 instruct format via format_training_sample().
The resulting Dataset has a single "text" column.

Usage:
    from chatbot.fine_tuning.dataset_prep import load_levelup_dataset
    train_ds, val_ds = load_levelup_dataset()
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

from datasets import Dataset

# Allow running as a standalone script from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chatbot.prompt_template import format_training_sample, detect_build, FINETUNE_SYSTEM_PROMPTS

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "finetune_train_v2.jsonl"
VAL_PATH   = ROOT / "data" / "raw" / "finetune_val_v2.jsonl"


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, build_filter: Optional[str] = None) -> list[dict]:
    """Load a JSONL file, optionally filtering to a single build."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if build_filter:
                build = detect_build(s.get("system", ""))
                if build != build_filter:
                    continue
            samples.append(s)
    return samples


def _to_hf_dataset(samples: list[dict], shuffle: bool = True, seed: int = 42) -> Dataset:
    """Convert list of {system, user, assistant} dicts to a HuggingFace Dataset."""
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    texts = []
    for s in samples:
        system    = s.get("system", "").strip()
        user      = s.get("user", "").strip()
        assistant = s.get("assistant", "").strip()
        if not system or not user or not assistant:
            continue
        # Normalise system prompts that use old/emoji variants to canonical ones
        build = detect_build(system)
        if build and build in FINETUNE_SYSTEM_PROMPTS:
            system = FINETUNE_SYSTEM_PROMPTS[build]
        texts.append(format_training_sample(system, user, assistant))

    return Dataset.from_dict({"text": texts})


# ── Public API ─────────────────────────────────────────────────────────────────

def load_levelup_dataset(
    build_filter: Optional[str] = None,
    train_path: Path = TRAIN_PATH,
    val_path:   Path = VAL_PATH,
    max_train_samples: Optional[int] = None,
    max_val_samples:   Optional[int] = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load the LevelUp training and validation datasets.

    Args:
        build_filter       : if set (e.g. "TITAN"), only load that build's samples
        train_path / val_path : override default file paths
        max_train_samples  : cap training set size (useful for quick tests)
        max_val_samples    : cap validation set size
        seed               : random seed for shuffling

    Returns:
        (train_dataset, val_dataset) as HuggingFace Dataset objects
    """
    print(f"Loading dataset ...")
    print(f"  Train: {train_path}")
    print(f"  Val  : {val_path}")
    if build_filter:
        print(f"  Filter: {build_filter} only")

    train_samples = _load_jsonl(train_path, build_filter)
    val_samples   = _load_jsonl(val_path,   build_filter)

    if max_train_samples:
        train_samples = train_samples[:max_train_samples]
    if max_val_samples:
        val_samples = val_samples[:max_val_samples]

    train_ds = _to_hf_dataset(train_samples, shuffle=True,  seed=seed)
    val_ds   = _to_hf_dataset(val_samples,   shuffle=False, seed=seed)

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val   samples: {len(val_ds):,}")
    return train_ds, val_ds


def get_build_counts(path: Path = TRAIN_PATH) -> dict[str, int]:
    """Return sample counts per build for a JSONL file."""
    from collections import Counter
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
                build = detect_build(s.get("system", "")) or "UNKNOWN"
                counts[build] += 1
            except:
                pass
    return dict(counts)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("LevelUp Dataset Prep -- Quick Test")
    print("=" * 55)

    # Print per-build counts
    print("\nTrain set build distribution:")
    counts = get_build_counts(TRAIN_PATH)
    total = sum(counts.values())
    for build, n in sorted(counts.items()):
        bar = "#" * (n // 500)
        print(f"  {build:<10} {n:>7,}  {bar}")
    print(f"  {'TOTAL':<10} {total:>7,}")

    # Load a small slice and verify format
    print("\nLoading first 5 ORACLE samples ...")
    train_ds, val_ds = load_levelup_dataset(
        build_filter="ORACLE",
        max_train_samples=5,
        max_val_samples=2,
    )
    print(f"\nSample text (first 400 chars):\n{train_ds[0]['text'][:400]}")
    print(f"\n... [{len(train_ds[0]['text'])} total chars]")
