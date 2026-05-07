"""
sentiment/data_prep.py
=======================
Load, clean, and split the LevelUp sentiment dataset.

The dataset is a JSONL file with one record per line:
    {"text": "...", "label": "motivated" | "neutral" | "struggling"}

Label map:
    0 = neutral    (majority class — ~80 %)
    1 = motivated  (~16 %)
    2 = struggling (~4 %)

The heavy class imbalance is handled downstream via class_weight="balanced"
and an oversampled training split for the rare "struggling" class.
"""

import json
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ── Label definitions ────────────────────────────────────────────────────────

LABELS      = ["neutral", "motivated", "struggling"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
INT_TO_LABEL = {i: l for l, i in LABEL_TO_INT.items()}


# ── Text cleaning ─────────────────────────────────────────────────────────────

# Smart quotes, em dashes, ellipsis, etc. → plain ASCII
_CLEAN_MAP = str.maketrans({
    "‘": "'", "’": "'",   # left/right single quote
    "“": '"', "”": '"',   # left/right double quote
    "—": "-", "–": "-",   # em dash, en dash
    "…": "...",                # ellipsis
    "é": "e", "à": "a",  # accented chars
})


def clean_text(text: str) -> str:
    """
    Normalise a raw quote string to clean ASCII for TF-IDF:
      - translate smart punctuation
      - collapse whitespace
      - strip leading/trailing spaces
    """
    text = text.translate(_CLEAN_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> tuple[list[str], np.ndarray]:
    """
    Load sentiment JSONL file.

    Returns:
        texts  — list of raw strings
        labels — int32 array (0=neutral, 1=motivated, 2=struggling)
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(clean_text(obj["text"]))
            labels.append(LABEL_TO_INT[obj["label"]])

    return texts, np.array(labels, dtype=np.int32)


# ── Train / test split ────────────────────────────────────────────────────────

def split(
    texts: list[str],
    labels: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Stratified 80/20 train-test split."""
    return train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


# ── TF-IDF vectoriser ─────────────────────────────────────────────────────────

def fit_vectorizer(
    texts: list[str],
    save_path: str | None = None,
    max_features: int = 8000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectoriser on training texts and optionally save it.

    Args:
        texts:        Training text strings.
        save_path:    If given, persist with joblib.
        max_features: Vocabulary size cap.
        ngram_range:  Unigrams + bigrams by default.
        sublinear_tf: Apply log(1+tf) scaling (better for short text).

    Returns:
        Fitted TfidfVectorizer.
    """
    vec = TfidfVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        sublinear_tf  = sublinear_tf,
        strip_accents = "unicode",
        analyzer      = "word",
        token_pattern = r"\b[a-zA-Z']{2,}\b",  # alphabetic tokens only
        min_df        = 2,                       # drop hapax legomena
    )
    vec.fit(texts)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vec, save_path)
    return vec


def load_vectorizer(path: str) -> TfidfVectorizer:
    return joblib.load(path)


# ── Dual TF-IDF (word + character n-grams) ───────────────────────────────────

class DualVectorizer:
    """
    Combines a word-level and a character-level TF-IDF vectorizer.

    Word TF-IDF captures semantic tokens (unigrams + bigrams).
    Char TF-IDF captures morphological patterns and handles typos /
    informal spellings common in user messages (e.g. "soooo stressed").

    The two sparse matrices are hstacked: shape (n, word_features + char_features).

    Usage::

        dv = DualVectorizer()
        X_train = dv.fit_transform(train_texts)
        X_test  = dv.transform(test_texts)
        joblib.dump(dv, "dual_vec.joblib")
    """

    def __init__(
        self,
        word_max_features: int = 12_000,
        char_max_features: int = 20_000,
    ):
        self.word_vec = TfidfVectorizer(
            max_features  = word_max_features,
            ngram_range   = (1, 3),        # unigrams, bigrams, trigrams
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"\b[a-zA-Z']{2,}\b",
            min_df        = 2,
        )
        self.char_vec = TfidfVectorizer(
            max_features  = char_max_features,
            ngram_range   = (2, 5),        # character 2-to-5-grams
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "char_wb",     # pads word boundaries with spaces
            min_df        = 3,
        )

    def fit_transform(self, texts: list[str]):
        from scipy.sparse import hstack
        X_word = self.word_vec.fit_transform(texts)
        X_char = self.char_vec.fit_transform(texts)
        return hstack([X_word, X_char], format="csr")

    def transform(self, texts: list[str]):
        from scipy.sparse import hstack
        X_word = self.word_vec.transform(texts)
        X_char = self.char_vec.transform(texts)
        return hstack([X_word, X_char], format="csr")

    @property
    def n_features(self) -> int:
        return (len(self.word_vec.vocabulary_) +
                len(self.char_vec.vocabulary_))


def fit_dual_vectorizer(
    texts: list[str],
    save_path: str | None = None,
    word_max_features: int = 12_000,
    char_max_features: int = 20_000,
) -> DualVectorizer:
    """
    Fit a DualVectorizer on training texts and optionally save it.
    """
    dv = DualVectorizer(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
    )
    dv.fit_transform  # just initialise — actual fitting in fit_transform below
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # We fit-transform then drop the result (caller will call transform)
    # but we need the vectorisers fitted, so:
    from scipy.sparse import hstack
    dv.word_vec.fit(texts)
    dv.char_vec.fit(texts)

    if save_path:
        joblib.dump(dv, save_path)
    return dv


# ── Augmentation for rare "struggling" class ──────────────────────────────────

def oversample_minority(
    texts: list[str],
    labels: np.ndarray,
    target_label: int = 2,   # "struggling"
    target_ratio: float = 0.12,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """
    Duplicate minority-class samples until they reach target_ratio of total.
    Simple but effective for small text datasets where SMOTE can't be used.
    """
    rng   = np.random.default_rng(random_state)
    total = len(labels)
    idx_minority = np.where(labels == target_label)[0]
    n_minority   = len(idx_minority)
    target_count = int(total * target_ratio)

    if n_minority >= target_count:
        return texts, labels   # already sufficient

    n_needed  = target_count - n_minority
    chosen    = rng.choice(idx_minority, size=n_needed, replace=True)
    extra_t   = [texts[i] for i in chosen]
    extra_l   = labels[chosen]

    aug_texts  = texts + extra_t
    aug_labels = np.concatenate([labels, extra_l])

    # Shuffle so duplicates aren't all at the end
    perm = rng.permutation(len(aug_labels))
    return [aug_texts[i] for i in perm], aug_labels[perm]
