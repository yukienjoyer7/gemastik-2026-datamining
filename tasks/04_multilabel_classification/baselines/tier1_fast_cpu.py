#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Tier 1 Fast CPU 2-Stage OOF Classifier Chain
=================================================================================
Standalone 2-Stage Out-Of-Fold Classifier Chain exploiting label correlations without
data leakage:
  - **Feature Extraction**: Word `(1, 2)` TF-IDF + Char-wb `(3, 5)` TF-IDF + `TruncatedSVD`
    dense representation.
  - **Stage 1 (Independent OOF Base Models)**:
    Fits blended `LogisticRegression` + Platt-calibrated `RidgeClassifier` per label $l \in \{1 \dots L\}$
    inside CV folds to produce leak-free `oof_stage1 (N, L)` and `test_stage1 (M, L)`.
  - **Stage 2 (Leak-Free Correlation Stacking / Classifier Chain)**:
    Augments the SVD features with:
      1. `oof_stage1` (all $L$ Stage-1 out-of-fold probabilities)
      2. Empirical co-occurrence expectation features $P_{\text{stage1}} \cdot C_{\text{cond}}$
         (expected activation of label $l$ given correlated labels' Stage-1 probabilities)
      3. Row-wise Stage-1 entropy & expected label cardinality ($\sum_l P_{il}^{(1)}$)
    and trains Stage-2 regularized linear + HistGradientBoosting meta-classifiers per label.
  - Standardized output contract: saves `oof_tier1_multilabel.npy`, `test_tier1_multilabel.npy`,
    and `sub_tier1_multilabel.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion


@dataclass
class CFG:
    """Configuration for Tier 1 2-Stage OOF Multilabel Classifier Chain."""

    exp_name: str = "tier1_multilabel"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    label_prefix: str = "label_"
    label_cols: Optional[Tuple[str, ...]] = None
    fold_col: str = "fold"

    max_word_features: int = 50000
    max_char_features: int = 60000
    svd_components: int = 32
    logreg_c: float = 2.0
    ridge_alpha: float = 1.2
    stage2_blend: float = 0.65

    seed: int = 42
    smoke_test: bool = False


def resolve_label_columns(df: pd.DataFrame, cfg: CFG) -> List[str]:
    if cfg.label_cols is not None and len(cfg.label_cols) > 0:
        return list(cfg.label_cols)
    cols = [c for c in df.columns if c.startswith(cfg.label_prefix)]
    if not cols:
        excluded = {cfg.id_col, cfg.text_col, cfg.fold_col}
        cols = [c for c in df.columns if c not in excluded]
    return cols


def evaluate_multilabel(Y_true: np.ndarray, Y_prob: np.ndarray, thr: float = 0.5) -> dict:
    Y_pred = (Y_prob >= thr).astype(int)
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    aucs = []
    for j in range(Y_true.shape[1]):
        if len(np.unique(Y_true[:, j])) > 1:
            aucs.append(float(roc_auc_score(Y_true[:, j], Y_prob[:, j])))
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "macro_auc": macro_auc}


def build_stage2_meta_features(
    X_svd: np.ndarray, stage1_probs: np.ndarray, cond_matrix: np.ndarray
) -> np.ndarray:
    """
    Constructs Stage-2 feature matrix combining:
      1. `X_svd` (dense LSA semantic features)
      2. `stage1_probs` ($L$ out-of-fold label probabilities)
      3. `cooc_dot = stage1_probs @ cond_matrix` (co-occurrence expected activations)
      4. `row_sum` (expected label cardinality) and `row_max` (max label confidence)
    """
    cooc_dot = stage1_probs @ cond_matrix
    row_sum = stage1_probs.sum(axis=1, keepdims=True)
    row_max = stage1_probs.max(axis=1, keepdims=True)
    p_clip = np.clip(stage1_probs, 1e-6, 1.0 - 1e-6)
    binary_entropy = -(
        p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip)
    ).mean(axis=1, keepdims=True)
    return np.hstack([X_svd, stage1_probs, cooc_dot, row_sum, row_max, binary_entropy])


def run_two_stage_chain(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    label_cols = resolve_label_columns(train_df, cfg)
    Y = train_df[label_cols].values.astype(np.int64)
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    n_train, n_labels = Y.shape
    n_test = len(test_df)

    tr_texts = train_df[cfg.text_col].fillna("").astype(str).tolist()
    te_texts = test_df[cfg.text_col].fillna("").astype(str).tolist()

    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=cfg.max_word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    max_features=cfg.max_char_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    X_tr_sparse = union.fit_transform(tr_texts)
    X_te_sparse = union.transform(te_texts)

    n_svd = min(cfg.svd_components, X_tr_sparse.shape[0] - 1, X_tr_sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components=max(2, n_svd), random_state=cfg.seed)
    X_tr_svd = svd.fit_transform(X_tr_sparse)
    X_te_svd = svd.transform(X_te_sparse)

    # =========================================================================
    # Stage 1: Independent OOF Linear Models per Label
    # =========================================================================
    oof_stage1 = np.zeros((n_train, n_labels), dtype=np.float64)
    test_stage1 = np.zeros((n_test, n_labels), dtype=np.float64)

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        for j in range(n_labels):
            y_tr_j = Y[tr_idx, j]
            if len(np.unique(y_tr_j)) < 2:
                const_p = float(y_tr_j.mean())
                oof_stage1[va_idx, j] = const_p
                test_stage1[:, j] += const_p / len(unique_folds)
                continue

            clf_lr = LogisticRegression(
                C=cfg.logreg_c, max_iter=1000, random_state=cfg.seed + int(f)
            )
            clf_lr.fit(X_tr_sparse[tr_idx], y_tr_j)
            p_va_lr = clf_lr.predict_proba(X_tr_sparse[va_idx])[:, 1]
            p_te_lr = clf_lr.predict_proba(X_te_sparse)[:, 1]

            clf_ridge = RidgeClassifier(alpha=cfg.ridge_alpha, random_state=cfg.seed + int(f))
            clf_ridge.fit(X_tr_sparse[tr_idx], y_tr_j)
            p_va_rg = expit(clf_ridge.decision_function(X_tr_sparse[va_idx]))
            p_te_rg = expit(clf_ridge.decision_function(X_te_sparse))

            oof_stage1[va_idx, j] = 0.65 * p_va_lr + 0.35 * p_va_rg
            test_stage1[:, j] += (0.65 * p_te_lr + 0.35 * p_te_rg) / len(unique_folds)

    s1_metrics = evaluate_multilabel(Y, oof_stage1)
    print(
        f"[STAGE 1 OOF] Micro-F1={s1_metrics['micro_f1']:.4f} | "
        f"Macro-F1={s1_metrics['macro_f1']:.4f} | Macro-AUC={s1_metrics['macro_auc']:.4f}"
    )

    # =========================================================================
    # Stage 2: Leak-Free OOF Classifier Chain with Co-Occurrence Features
    # =========================================================================
    oof_stage2 = np.zeros((n_train, n_labels), dtype=np.float64)
    test_stage2 = np.zeros((n_test, n_labels), dtype=np.float64)

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        # Estimate conditional co-occurrence matrix strictly on fold training split
        Y_tr_f = Y[tr_idx].astype(np.float64)
        cooc_f = Y_tr_f.T @ Y_tr_f
        diag_f = np.maximum(np.diag(cooc_f), 1.0)
        cond_f = cooc_f / diag_f[:, None]
        np.fill_diagonal(cond_f, 0.0)

        X_meta_tr = build_stage2_meta_features(X_tr_svd[tr_idx], oof_stage1[tr_idx], cond_f)
        X_meta_va = build_stage2_meta_features(X_tr_svd[va_idx], oof_stage1[va_idx], cond_f)
        X_meta_te = build_stage2_meta_features(X_te_svd, test_stage1, cond_f)

        for j in range(n_labels):
            y_tr_j = Y[tr_idx, j]
            if len(np.unique(y_tr_j)) < 2:
                oof_stage2[va_idx, j] = oof_stage1[va_idx, j]
                test_stage2[:, j] += test_stage1[:, j] / len(unique_folds)
                continue

            meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed + int(f))
            meta_lr.fit(X_meta_tr, y_tr_j)
            p_va_meta_lr = meta_lr.predict_proba(X_meta_va)[:, 1]
            p_te_meta_lr = meta_lr.predict_proba(X_meta_te)[:, 1]

            meta_gbdt = HistGradientBoostingClassifier(
                max_iter=50 if cfg.smoke_test else 120,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=5 if cfg.smoke_test else 15,
                random_state=cfg.seed + int(f),
            )
            meta_gbdt.fit(X_meta_tr, y_tr_j)
            p_va_gbdt = meta_gbdt.predict_proba(X_meta_va)[:, 1]
            p_te_gbdt = meta_gbdt.predict_proba(X_meta_te)[:, 1]

            p_va_s2 = 0.5 * p_va_meta_lr + 0.5 * p_va_gbdt
            p_te_s2 = 0.5 * p_te_meta_lr + 0.5 * p_te_gbdt

            oof_stage2[va_idx, j] = (
                (1.0 - cfg.stage2_blend) * oof_stage1[va_idx, j] + cfg.stage2_blend * p_va_s2
            )
            test_stage2[:, j] += (
                (1.0 - cfg.stage2_blend) * test_stage1[:, j] + cfg.stage2_blend * p_te_s2
            ) / len(unique_folds)

    s2_metrics = evaluate_multilabel(Y, oof_stage2)
    print(
        f"[STAGE 2 OOF] Micro-F1={s2_metrics['micro_f1']:.4f} | "
        f"Macro-F1={s2_metrics['macro_f1']:.4f} | Macro-AUC={s2_metrics['macro_auc']:.4f}"
    )

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_stage2.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_stage2.astype(np.float32))

    sub_df = pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values})
    for idx, col in enumerate(label_cols):
        sub_df[col] = (test_stage2[:, idx] >= 0.5).astype(int)
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    return oof_stage2, test_stage2


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Tier 1 2-Stage OOF Classifier Chain")
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
        rng = np.random.default_rng(cfg.seed)
        vocab = {
            0: ["inflation", "bank", "fiscal", "bond", "yield"],
            1: ["interest", "monetary", "central", "rate"],
            2: ["transformer", "neural", "tensor", "gpu"],
            3: ["attention", "backpropagation", "gradient"],
        }
        tr_rows, te_rows = [], []
        for i in range(120):
            y0 = int(i % 2 == 0)
            y1 = int(y0 == 1 and i % 4 == 0)  # correlated child label of y0
            y2 = int(i % 3 == 0)
            y3 = int(y2 == 1 and i % 6 == 0)  # correlated child label of y2
            words = ["document", "report"]
            for lid, active in enumerate([y0, y1, y2, y3]):
                if active:
                    words.extend(rng.choice(vocab[lid], size=6, replace=True))
            tr_rows.append(
                {
                    "id": f"tr_{i}",
                    "text": " ".join(words),
                    "label_0": y0,
                    "label_1": y1,
                    "label_2": y2,
                    "label_3": y3,
                    "fold": i % 5,
                }
            )
        for i in range(30):
            words = list(rng.choice(vocab[i % 4], size=8, replace=True))
            te_rows.append({"id": f"te_{i}", "text": " ".join(words)})

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 04 Tier 1 2-Stage OOF Classifier Chain...")
            oof, test_p = run_two_stage_chain(pd.DataFrame(tr_rows), pd.DataFrame(te_rows), cfg)
            assert oof.shape == (120, 4)
            assert test_p.shape == (30, 4)
            print("[SUCCESS] Task 04 Tier 1 smoke test passed.")
    else:
        run_two_stage_chain(load_table(cfg.train_path), load_table(cfg.test_path), cfg)


if __name__ == "__main__":
    main()
