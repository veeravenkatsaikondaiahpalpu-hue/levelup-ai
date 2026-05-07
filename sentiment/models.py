"""
sentiment/models.py
====================
Train and serve six sentiment classifiers for LevelUp AI.

Classifiers (all use dual TF-IDF: word + char n-grams):
  1. Logistic Regression   — interpretable, strong sparse-text baseline
  2. LinearSVC             — max-margin, calibrated for probabilities
  3. XGBoost               — proper gradient boosting (faster, better than sklearn GB)
  4. LightGBM              — Microsoft boosting, often best on sparse features
  5. Voting Ensemble       — soft vote: LR + SVC + LGB
  6. Stacking Ensemble     — LR + SVC + LGB base -> LR meta-learner

Feature engineering:
  - Word TF-IDF: 12K features, (1,3)-grams
  - Char TF-IDF: 20K features, (2,5)-grams, char_wb analyser
  - Combined: 32K sparse features (hstacked)

Labels:
    0 = neutral      (matter-of-fact, no strong emotion)
    1 = motivated    (energised, positive, driven)
    2 = struggling   (low, negative, overwhelmed)

Training:
    python -m sentiment.train

Inference:
    from sentiment.models import get_analyzer
    analyzer = get_analyzer()
    result = analyzer.predict("I feel like I can't keep going today")
    # -> {"label": "struggling", "confidence": 0.91, ...}
"""

import os
import warnings
import numpy as np
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import LinearSVC
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.ensemble      import VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics       import accuracy_score, classification_report, f1_score
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier

from sentiment.data_prep import (
    load_jsonl, split, fit_dual_vectorizer,
    oversample_minority, clean_text, INT_TO_LABEL, LABELS,
)

# ── Keyword lexicon (domain-specific boost for conversational text) ────────────

import re as _re

# Phrase patterns (matched as substrings on lowercased text)
_STRUGGLING_PHRASES = [
    "falling behind", "give up", "giving up", "want to quit", "want to stop",
    "can't keep going", "can't do this", "nothing works", "doesn't work",
    "no motivation", "zero motivation", "lost motivation", "burned out",
    "burnt out", "bad day", "rough day", "terrible day", "hard day",
    "hate myself", "hate this", "want to stop", "ready to quit",
]

# Word patterns (matched as whole words)
_STRUGGLING_WORDS = [
    "exhausted", "exhausting", "hopeless", "overwhelmed", "depressed",
    "burnout", "stuck", "failing", "failure", "worthless", "pointless",
    "useless", "unmotivated", "drained", "struggling", "broken", "slipping",
    "relapsed", "missed", "skipped", "failed", "anxious", "anxiety",
    "stressed", "stress", "crying", "behind",
]

_MOTIVATED_PHRASES = [
    "crushed it", "killed it", "nailed it", "beast mode", "on fire",
    "let's go", "lets go", "no excuses", "personal best", "personal record",
    "new record", "best ever", "leveled up", "levelled up", "all in",
    "showing up", "30-day streak", "best day", "great day",
]

_MOTIVATED_WORDS = [
    "unstoppable", "hyped", "pumped", "incredible", "accomplished",
    "milestone", "proud", "gains", "winning", "motivated", "energized",
    "energised", "excited", "smashed", "destroyed", "awesome", "fantastic",
    "streak", "pr",  # keep "pr" as word-boundary match
]


def _count_signals(text: str, phrases: list[str], words: list[str]) -> int:
    """Count how many phrase/word patterns match in text."""
    lower = text.lower()
    count = 0
    for ph in phrases:
        if ph in lower:
            count += 1
    for w in words:
        if _re.search(r"\b" + _re.escape(w) + r"\b", lower):
            count += 1
    return count


def _detect_keyword_class(text: str) -> tuple[str | None, int]:
    """
    Return (dominant_class, hit_count) if keyword signals fire,
    else (None, 0).
    """
    s_hits = _count_signals(text, _STRUGGLING_PHRASES, _STRUGGLING_WORDS)
    m_hits = _count_signals(text, _MOTIVATED_PHRASES,  _MOTIVATED_WORDS)

    if s_hits == 0 and m_hits == 0:
        return None, 0
    if s_hits > m_hits:
        return "struggling", s_hits
    if m_hits > s_hits:
        return "motivated", m_hits
    return None, 0  # tie → let ML decide

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "sentiment")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_jsonl: str,
    model_dir:  str = MODEL_DIR,
) -> dict:
    """
    Full training pipeline:
      load data -> dual TF-IDF -> 6 classifiers -> save artefacts.

    Models trained:
      logistic_regression, linear_svc, xgboost, lightgbm,
      voting_ensemble, stacking_ensemble

    Args:
        data_jsonl: Path to sentiment JSONL file (v2 recommended).
        model_dir:  Directory for saved artefacts.

    Returns:
        Dict: model_name -> {accuracy, macro_f1, per_class_f1}
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("  Loading data ...")
    texts, labels = load_jsonl(data_jsonl)
    dist = {lbl: int((labels == i).sum()) for i, lbl in enumerate(LABELS)}
    print(f"  Loaded {len(texts):,} samples  |  {dist}")

    # ── Split ─────────────────────────────────────────────────────────────
    train_texts, test_texts, y_train, y_test = split(texts, labels)
    print(f"  Train: {len(train_texts):,}   Test: {len(test_texts):,}")

    # ── Dual TF-IDF vectoriser (word + char n-grams) ──────────────────────
    print("  Fitting Dual TF-IDF vectoriser (word 1-3gram + char 2-5gram) ...")
    vec_path = os.path.join(model_dir, "dual_vectorizer.joblib")
    dv       = fit_dual_vectorizer(train_texts, save_path=vec_path)
    X_train  = dv.transform(train_texts)
    X_test   = dv.transform(test_texts)
    print(f"  Feature dimensions: {X_train.shape[1]:,}  "
          f"(word={len(dv.word_vec.vocabulary_):,} + char={len(dv.char_vec.vocabulary_):,})")

    # ── Also keep legacy single vectoriser for backwards compatibility ─────
    from sentiment.data_prep import fit_vectorizer
    legacy_vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    lv = fit_vectorizer(train_texts, save_path=legacy_vec_path)

    # ── Build base classifiers ────────────────────────────────────────────
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.5,
        solver="lbfgs", random_state=42,
    )
    svc_base = LinearSVC(
        class_weight="balanced", max_iter=2000, C=0.8, random_state=42,
    )
    svc = CalibratedClassifierCV(svc_base, cv=3, method="sigmoid")

    xgb = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 6,
        learning_rate      = 0.1,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        use_label_encoder  = False,
        eval_metric        = "mlogloss",
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = 0,
    )

    lgb = LGBMClassifier(
        n_estimators    = 400,
        num_leaves      = 63,
        learning_rate   = 0.08,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
        verbose         = -1,
    )

    # Voting: soft-vote across LR + SVC + LGB
    voting = VotingClassifier(
        estimators = [("lr", lr), ("svc", svc), ("lgb", lgb)],
        voting     = "soft",
        n_jobs     = -1,
    )

    # Stacking: LR + SVC + LGB base -> LR meta-learner (3-fold CV OOF)
    stacking = StackingClassifier(
        estimators = [
            ("lr",  LogisticRegression(class_weight="balanced", max_iter=1000, C=1.5, solver="lbfgs", random_state=42)),
            ("svc", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, C=0.8, random_state=42), cv=3, method="sigmoid")),
            ("lgb", LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
        ],
        final_estimator = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42),
        cv              = 3,
        n_jobs          = -1,
        passthrough     = False,
    )

    model_specs = [
        ("logistic_regression", lr,      "logistic_regression.joblib"),
        ("linear_svc",          svc,     "linear_svc.joblib"),
        ("xgboost",             xgb,     "xgboost.joblib"),
        ("lightgbm",            lgb,     "lightgbm.joblib"),
        ("voting_ensemble",     voting,  "voting_ensemble.joblib"),
        ("stacking_ensemble",   stacking,"stacking_ensemble.joblib"),
    ]

    results = {}

    for name, clf, fname in model_specs:
        print(f"\n  Training {name} ...")
        clf.fit(X_train, y_train)
        path = os.path.join(model_dir, fname)
        joblib.dump(clf, path)

        y_pred   = clf.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report   = classification_report(
            y_test, y_pred, target_names=LABELS,
            output_dict=True, zero_division=0,
        )

        results[name] = {
            "accuracy":  round(acc, 4),
            "macro_f1":  round(macro_f1, 4),
            "per_class": {lbl: round(report[lbl]["f1-score"], 4) for lbl in LABELS},
        }

        print(f"  Saved -> {path}")
        print(f"  Acc={acc:.4f}  MacroF1={macro_f1:.4f}  "
              f"struggling_F1={results[name]['per_class']['struggling']:.4f}")

    return results


# ── Inference ─────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Wraps the trained TF-IDF + classifier pipeline for single-text inference.

    The primary classifier is Logistic Regression (best calibrated probabilities).
    LinearSVC and Gradient Boosting scores are returned for comparison.

    Usage::

        analyzer = SentimentAnalyzer()
        analyzer.load()
        result = analyzer.predict("I've been feeling really unmotivated lately")
        # {
        #   "label":      "struggling",
        #   "confidence": 0.84,
        #   "scores":     {"neutral": 0.08, "motivated": 0.08, "struggling": 0.84},
        #   "flag":       "support_mode",
        #   "all_models": {...}
        # }
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir  = model_dir
        self.vec        = None   # DualVectorizer (preferred) or TfidfVectorizer (fallback)
        self.lr         = None
        self.svc        = None
        self.xgb        = None
        self.lgb        = None
        self.voting     = None
        self.stacking   = None

    def load(self) -> None:
        """Load dual vectoriser + all available classifier artefacts."""
        # Prefer new dual vectorizer; fall back to legacy single vectorizer
        dual_p   = os.path.join(self.model_dir, "dual_vectorizer.joblib")
        legacy_p = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")

        if os.path.exists(dual_p):
            self.vec = joblib.load(dual_p)
        elif os.path.exists(legacy_p):
            self.vec = joblib.load(legacy_p)
        else:
            raise FileNotFoundError(
                f"No vectoriser found in {self.model_dir}. "
                "Run `python -m sentiment.train` first."
            )

        self.lr       = _load_opt(os.path.join(self.model_dir, "logistic_regression.joblib"))
        self.svc      = _load_opt(os.path.join(self.model_dir, "linear_svc.joblib"))
        self.xgb      = _load_opt(os.path.join(self.model_dir, "xgboost.joblib"))
        self.lgb      = _load_opt(os.path.join(self.model_dir, "lightgbm.joblib"))
        self.voting   = _load_opt(os.path.join(self.model_dir, "voting_ensemble.joblib"))
        self.stacking = _load_opt(os.path.join(self.model_dir, "stacking_ensemble.joblib"))

        if self._primary is None:
            raise RuntimeError("No classifier found. Run training first.")

    @property
    def _primary(self):
        # Priority: stacking > voting > lgb > svc > lr
        return self.stacking or self.voting or self.lgb or self.svc or self.lr

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Analyse the sentiment of a single text string.

        Args:
            text: Raw user message (any length).

        Returns::

            {
              "label":      "neutral" | "motivated" | "struggling",
              "confidence": float,    # primary model's max class probability
              "scores": {
                  "neutral":    float,
                  "motivated":  float,
                  "struggling": float,
              },
              "flag":       None | "support_mode" | "hype_mode",
              "all_models": {
                  "logistic_regression": {"label": ..., "confidence": ...},
                  "linear_svc":          {"label": ..., "confidence": ...},
                  "gradient_boosting":   {"label": ..., "confidence": ...},
              }
            }
        """
        text_clean = clean_text(text)
        X = self.vec.transform([text_clean])   # works for both DualVectorizer and TfidfVectorizer

        # ML probabilities (suppress LightGBM feature-name warning on sparse input)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            raw_proba = self._primary.predict_proba(X)[0]
        ml_scores = {LABELS[i]: float(raw_proba[i]) for i in range(3)}

        # Keyword detection (stronger signal for conversational text)
        kw_class, kw_hits = _detect_keyword_class(text)

        if kw_class is not None and kw_hits >= 2:
            # Strong keyword signal: override ML entirely
            label = kw_class
            # Build scores: keyword class dominates, scaled by hit strength
            base_conf = min(0.85, 0.55 + kw_hits * 0.08)
            remainder = (1.0 - base_conf) / 2
            scores = {lbl: round(remainder, 4) for lbl in LABELS}
            scores[kw_class] = round(base_conf, 4)
            conf = base_conf

        elif kw_class is not None and kw_hits == 1:
            # Weak keyword signal: blend ML 50/50 with keyword boost
            blended = dict(ml_scores)
            blended[kw_class] = min(0.90, blended[kw_class] + 0.35)
            total   = sum(blended.values()) or 1.0
            scores  = {lbl: round(v / total, 4) for lbl, v in blended.items()}
            label   = max(scores, key=scores.__getitem__)
            conf    = scores[label]

        else:
            # No keyword signal: use ML probabilities directly
            scores = {lbl: round(v, 4) for lbl, v in ml_scores.items()}
            label  = max(scores, key=scores.__getitem__)
            conf   = scores[label]

        # Flag for chatbot tone switching
        flag = None
        if label == "struggling" and conf >= 0.45:
            flag = "support_mode"
        elif label == "motivated" and conf >= 0.50:
            flag = "hype_mode"

        # All models
        all_models: dict = {}
        for name, clf in [
            ("logistic_regression", self.lr),
            ("linear_svc",          self.svc),
            ("xgboost",             self.xgb),
            ("lightgbm",            self.lgb),
            ("voting_ensemble",     self.voting),
            ("stacking_ensemble",   self.stacking),
        ]:
            if clf is None:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                p = clf.predict_proba(X)[0]
            all_models[name] = {
                "label":      INT_TO_LABEL[int(p.argmax())],
                "confidence": round(float(p.max()), 4),
            }

        return {
            "label":      label,
            "confidence": conf,
            "scores":     scores,
            "flag":       flag,
            "all_models": all_models,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() on a list of texts. Returns one dict per input."""
        return [self.predict(t) for t in texts]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_opt(path: str):
    """Load a joblib file if it exists, else return None."""
    return joblib.load(path) if os.path.exists(path) else None


# ── Module-level singleton ────────────────────────────────────────────────────

_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    """
    Return the module-level singleton SentimentAnalyzer (lazy-loaded).
    Thread-safe for single-worker FastAPI / uvicorn usage.
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
        _analyzer.load()
    return _analyzer
