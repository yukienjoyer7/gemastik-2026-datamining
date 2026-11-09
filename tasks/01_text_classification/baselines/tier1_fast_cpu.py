#!/usr/bin/env python3
"""
Task 01 — Text Classification: Tier 1 Fast CPU Baseline
=======================================================
Standalone Out-Of-Fold (OOF) pipeline combining:
  - FeatureUnion of:
    1. Word (1, 3) TF-IDF (`sublinear_tf=True`, `strip_accents='unicode'`)
    2. Char-wb (3, 5) TF-IDF (`sublinear_tf=True`)
    3. Lexical & surface statistics (char length, word count, uppercase ratio,
       digit ratio, punctuation count, average word length + StandardScaler)
  - Out-Of-Fold ensemble blending:
    1. `LogisticRegression` (L2 regularized cross-entropy)
    2. `CalibratedClassifierCV(LinearSVC)` (Platt-scaled max-margin SVM)
    3. `RidgeClassifier` (with temperature-scaled softmax conversion)
  - Evaluates OOF Macro-F1, LogLoss, and Macro ROC-AUC.
  - Exports standardized artifacts:
    `oof_tier1_tfidf.npy`, `test_tier1_tfidf.npy`, and `sub_tier1_tfidf.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import string
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class CFG:
    """Configuration for Tier 1 TF-IDF + Lexical CPU Baseline."""

    exp_name: str = "tier1_tfidf"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"

    # TF-IDF hyperparameters
    word_ngram_range: Tuple[int, int] = (1, 3)
    char_ngram_range: Tuple[int, int] = (3, 5)
    max_word_features: int = 80000
    max_char_features: int = 100000
    min_df: int = 2

    # Classifier hyperparameters
    logreg_c: float = 2.0
    svc_c: float = 0.5
    ridge_alpha: float = 1.5
    blend_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)

    seed: int = 42
    smoke_test: bool = False


class LexicalStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 6 dense surface/lexical features per document:
      0. log1p(char_length)
      1. log1p(word_count)
      2. uppercase_char_ratio
      3. digit_char_ratio
      4. punctuation_char_ratio
      5. average_word_length
    Returns a CSR sparse matrix for seamless compatibility with FeatureUnion.
    """

    def fit(self, X: Sequence[str], y: Optional[np.ndarray] = None) -> "LexicalStatsExtractor":
        return self

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        punct_set = set(string.punctuation)
        feats = np.zeros((len(X), 6), dtype=np.float32)

        for i, raw in enumerate(X):
            text = str(raw) if raw is not None else ""
            c_len = len(text)
            words = text.split()
            w_cnt = len(words)
            denom = max(c_len, 1)

            upper_cnt = sum(1 for ch in text if ch.isupper())
            digit_cnt = sum(1 for ch in text if ch.isdigit())
            punct_cnt = sum(1 for ch in text if ch in punct_set)
            avg_w_len = (sum(len(w) for w in words) / max(w_cnt, 1)) if w_cnt > 0 else 0.0

            feats[i, 0] = np.log1p(c_len)
            feats[i, 1] = np.log1p(w_cnt)
            feats[i, 2] = upper_cnt / denom
            feats[i, 3] = digit_cnt / denom
            feats[i, 4] = punct_cnt / denom
            feats[i, 5] = avg_w_len

        return sparse.csr_matrix(feats)


def build_feature_union(cfg: CFG) -> FeatureUnion:
    """Constructs the Word TF-IDF + Char-wb TF-IDF + Scaled Lexical FeatureUnion."""
    word_tfidf = TfidfVectorizer(
        ngram_range=cfg.word_ngram_range,
        max_features=cfg.max_word_features,
        min_df=1 if cfg.smoke_test else cfg.min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=cfg.char_ngram_range,
        max_features=cfg.max_char_features,
        min_df=1 if cfg.smoke_test else cfg.min_df,
        sublinear_tf=True,
        analyzer="char_wb",
    )
    lexical_pipe = Pipeline(
        [
            ("lex", LexicalStatsExtractor()),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
            ("lexical_stats", lexical_pipe),
        ]
    )


def ridge_decision_to_probs(decision: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert RidgeClassifier decision function outputs into calibrated probabilities."""
    if decision.ndim == 1 or num_classes == 2:
        scores = np.vstack([-decision, decision]).T if decision.ndim == 1 else decision
        return softmax(scores, axis=1)
    return softmax(decision, axis=1)


def compute_metrics(y_true: np.ndarray, oof_probs: np.ndarray) -> dict:
    preds = np.argmax(oof_probs, axis=1)
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    ll = float(log_loss(y_true, oof_probs))
    num_classes = oof_probs.shape[1]
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(y_true, oof_probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, oof_probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {"macro_f1": macro_f1, "log_loss": ll, "roc_auc": auc}


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fit FeatureUnion and OOF Linear/SVM ensemble across pre-assigned folds."""
    classes = np.sort(train_df[cfg.target_col].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = train_df[cfg.target_col].map(label_to_idx).values
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_classes = len(classes)

    train_texts = train_df[cfg.text_col].fillna("").astype(str).tolist()
    test_texts = test_df[cfg.text_col].fillna("").astype(str).tolist()

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
    w_lr, w_svc, w_ridge = cfg.blend_weights

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        featurizer = build_feature_union(cfg)
        X_tr = featurizer.fit_transform([train_texts[i] for i in tr_idx])
        X_va = featurizer.transform([train_texts[i] for i in va_idx])
        X_te = featurizer.transform(test_texts)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. Logistic Regression
        clf_lr = LogisticRegression(
            C=cfg.logreg_c, max_iter=1000, random_state=cfg.seed + int(f)
        )
        clf_lr.fit(X_tr, y_tr)
        p_va_lr = clf_lr.predict_proba(X_va)
        p_te_lr = clf_lr.predict_proba(X_te)

        # 2. Calibrated LinearSVC
        base_svc = LinearSVC(C=cfg.svc_c, max_iter=3000, random_state=cfg.seed + int(f))
        clf_svc = CalibratedClassifierCV(estimator=base_svc, cv=3, method="sigmoid")
        clf_svc.fit(X_tr, y_tr)
        p_va_svc = clf_svc.predict_proba(X_va)
        p_te_svc = clf_svc.predict_proba(X_te)

        # 3. RidgeClassifier
        clf_ridge = RidgeClassifier(alpha=cfg.ridge_alpha, random_state=cfg.seed + int(f))
        clf_ridge.fit(X_tr, y_tr)
        p_va_ridge = ridge_decision_to_probs(clf_ridge.decision_function(X_va), num_classes)
        p_te_ridge = ridge_decision_to_probs(clf_ridge.decision_function(X_te), num_classes)

        fold_va = w_lr * p_va_lr + w_svc * p_va_svc + w_ridge * p_va_ridge
        fold_te = w_lr * p_te_lr + w_svc * p_te_svc + w_ridge * p_te_ridge

        oof_probs[va_idx] = fold_va
        test_probs += fold_te / len(unique_folds)

        fold_f1 = f1_score(y_va, np.argmax(fold_va, axis=1), average="macro")
        print(f"  [Fold {f}] val_size={len(va_idx)} | Macro-F1={fold_f1:.4f} | features={X_tr.shape[1]}")

    metrics = compute_metrics(y, oof_probs)
    print(
        f"[OOF SUMMARY] Macro-F1={metrics['macro_f1']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
    )

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / f"oof_{cfg.exp_name}.npy"
    test_path = out_dir / f"test_{cfg.exp_name}.npy"
    sub_path = out_dir / f"sub_{cfg.exp_name}.csv"

    np.save(oof_path, oof_probs.astype(np.float32))
    np.save(test_path, test_probs.astype(np.float32))

    pred_classes = [classes[i] for i in np.argmax(test_probs, axis=1)]
    sub_df = pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values, cfg.target_col: pred_classes})
    sub_df.to_csv(sub_path, index=False)
    print(f"[SAVED] {oof_path.name} {oof_probs.shape}, {test_path.name} {test_probs.shape}, {sub_path.name}")

    return oof_probs, test_probs, metrics


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def generate_synthetic_dataset(n_train: int = 150, n_test: int = 45, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vocab_by_class = {
        0: ["macroeconomics", "inflation", "central", "bank", "interest", "fiscal", "currency", "bond", "yield"],
        1: ["championship", "striker", "goalkeeper", "penalty", "stadium", "tournament", "coach", "victory"],
        2: ["transformer", "backpropagation", "gradient", "tensor", "attention", "tokenizer", "CUDA", "pretraining"],
    }
    noise_vocab = ["the", "and", "in", "of", "for", "with", "report", "analysis", "update", "2026"]

    train_rows, test_rows = [], []
    for i in range(n_train):
        cls = i % 3
        sig_words = rng.choice(vocab_by_class[cls], size=12, replace=True)
        bg_words = rng.choice(noise_vocab, size=8, replace=True)
        tokens = list(sig_words) + list(bg_words)
        rng.shuffle(tokens)
        train_rows.append({"id": f"tr_{i}", "text": " ".join(tokens) + "!", "label": cls, "fold": i % 5})

    for i in range(n_test):
        cls = i % 3
        sig_words = rng.choice(vocab_by_class[cls], size=10, replace=True)
        bg_words = rng.choice(noise_vocab, size=8, replace=True)
        tokens = list(sig_words) + list(bg_words)
        rng.shuffle(tokens)
        test_rows.append({"id": f"te_{i}", "text": " ".join(tokens) + "?"})

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Tier 1 Fast CPU TF-IDF Baseline")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Tier 1 TF-IDF + Lexical OOF Pipeline...")
            train_df, test_df = generate_synthetic_dataset(seed=cfg.seed)
            oof, test_preds, metrics = run_oof_pipeline(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            assert metrics["macro_f1"] > 0.70, f"Expected high F1 on synthetic data, got {metrics['macro_f1']}"
            print("[SUCCESS] Tier 1 smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_pipeline(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
