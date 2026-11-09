#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Support-Regularized Threshold Optimizer & Fallback
=======================================================================================
Standalone post-processing pipeline addressing the two biggest traps in multilabel F1:
  1. **Empirical-Bayes Support-Regularized Per-Class Threshold Optimization**:
     Optimizing independent thresholds $\tau_c^*$ on rare tail classes ($n_c < 20$ positives)
     severely overfits OOF noise. We first find the global threshold $\tau_{\text{global}}$
     maximizing Micro-F1, then find each label's raw optimal threshold $\tau_c^*$, and
     shrink rare-class thresholds toward $\tau_{\text{global}}$ proportional to positive
     support $n_c$:
       $\hat{\tau}_c = \frac{n_c}{n_c + \lambda} \tau_c^* + \frac{\lambda}{n_c + \lambda} \tau_{\text{global}}$
  2. **Minimum-1-Label Argmax Fallback**:
     In competitions where every sample has $\ge 1$ ground-truth label, any row where all
     predicted probabilities fall below $\hat{\tau}_c$ has its highest-margin label
     $\arg\max_c (P_{ic} - \hat{\tau}_c)$ activated.
  3. Standardized output contract: saves `oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     and `sub_<exp_name>.csv`.

Usage:
    python postprocess.py --oof-path oof_tier2_multilabel_dl.npy --test-probs-path test_tier2_multilabel_dl.npy
    python postprocess.py --smoke-test
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
from sklearn.metrics import f1_score


@dataclass
class CFG:
    """Configuration for Task 04 Regularized Per-Class Threshold Post-Processing."""

    exp_name: str = "tier2_multilabel_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_multilabel.npy"
    test_probs_path: str = "test_tier1_multilabel.npy"
    output_dir: str = "."
    id_col: str = "id"
    label_prefix: str = "label_"
    label_cols: Optional[Tuple[str, ...]] = None
    fold_col: str = "fold"

    shrinkage_lambda: float = 25.0  # prior support strength for rare-label shrinkage
    enforce_min_one_label: bool = True
    seed: int = 42
    smoke_test: bool = False


def resolve_label_columns(df: pd.DataFrame, cfg: CFG) -> List[str]:
    if cfg.label_cols is not None and len(cfg.label_cols) > 0:
        return list(cfg.label_cols)
    cols = [c for c in df.columns if c.startswith(cfg.label_prefix)]
    if not cols:
        excluded = {cfg.id_col, "text", cfg.fold_col}
        cols = [c for c in df.columns if c not in excluded]
    return cols


def optimize_regularized_thresholds(
    oof_probs: np.ndarray,
    Y_true: np.ndarray,
    shrinkage_lambda: float = 25.0,
    grid_size: int = 161,
) -> Tuple[np.ndarray, float, np.ndarray]:
    r"""
    Finds:
      1. $\tau_{\text{global}}$ maximizing global Micro-F1 across all $(N \times L)$ cells.
      2. Raw per-label threshold $\tau_c^*$ maximizing F1 for label $c$.
      3. Support-shrunk threshold:
         $\hat{\tau}_c = \frac{n_c}{n_c + \lambda} \tau_c^* + \frac{\lambda}{n_c + \lambda} \tau_{\text{global}}$
    Returns `(shrunk_thresholds, global_threshold, raw_thresholds)`.
    """
    grid = np.linspace(0.10, 0.90, grid_size)
    Y_bin = (Y_true > 0.5).astype(int)

    # 1. Global threshold optimizing Micro-F1
    best_global_thr = 0.50
    best_micro = -1.0
    for thr in grid:
        score = float(f1_score(Y_bin, (oof_probs >= thr).astype(int), average="micro", zero_division=0))
        if score > best_micro:
            best_micro = score
            best_global_thr = float(thr)

    # 2. Per-class threshold with Empirical-Bayes shrinkage toward best_global_thr
    n_labels = oof_probs.shape[1]
    raw_thrs = np.full(n_labels, best_global_thr, dtype=np.float64)
    shrunk_thrs = np.full(n_labels, best_global_thr, dtype=np.float64)

    for c in range(n_labels):
        y_c = Y_bin[:, c]
        p_c = oof_probs[:, c]
        n_pos = float(y_c.sum())

        best_t_c = best_global_thr
        best_f1_c = -1.0
        if n_pos > 0:
            for thr in grid:
                f1_c = float(f1_score(y_c, (p_c >= thr).astype(int), zero_division=0))
                if f1_c > best_f1_c:
                    best_f1_c = f1_c
                    best_t_c = float(thr)

        raw_thrs[c] = best_t_c
        weight_c = n_pos / max(n_pos + shrinkage_lambda, 1e-9)
        shrunk_thrs[c] = weight_c * best_t_c + (1.0 - weight_c) * best_global_thr

    return shrunk_thrs, best_global_thr, raw_thrs


def apply_thresholds_with_min_one_fallback(
    probs: np.ndarray,
    thresholds: np.ndarray,
    enforce_min_one: bool = True,
) -> np.ndarray:
    """
    Applies per-class decision thresholds `thresholds (L,)` and optionally activates
    `argmax_c (probs[i, c] - thresholds[c])` for any row where all labels are 0.
    """
    thrs = thresholds.reshape(1, -1)
    preds = (probs >= thrs).astype(np.int64)
    if enforce_min_one:
        zero_rows = np.where(preds.sum(axis=1) == 0)[0]
        if len(zero_rows) > 0:
            best_cols = np.argmax(probs[zero_rows] - thrs, axis=1)
            preds[zero_rows, best_cols] = 1
    return preds


def calibrate_margin_shifted_probs(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Shifts probabilities in log-odds space so that the decision boundary `p_c == thresholds[c]`
    maps to `0.50` in the saved `.npy` probability artifact.
    """
    p = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    t = np.clip(thresholds.reshape(1, -1).astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    thr_logits = np.log(t / (1.0 - t))
    shifted = 1.0 / (1.0 + np.exp(-(logits - thr_logits)))
    return shifted.astype(np.float32)


def run_postprocess(
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    Y_true: np.ndarray,
    test_ids: np.ndarray,
    label_cols: Sequence[str],
    cfg: CFG,
    folds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    default_preds = (oof_probs >= 0.5).astype(int)
    base_macro = float(f1_score(Y_true, default_preds, average="macro", zero_division=0))
    base_micro = float(f1_score(Y_true, default_preds, average="micro", zero_division=0))

    if folds is not None:
        unique_folds = np.sort(np.unique(folds))
        oof_preds = np.zeros_like(Y_true, dtype=np.int64)
        fold_thrs = []
        for f in unique_folds:
            tr_m = folds != f
            va_m = folds == f
            thrs_f, _, _ = optimize_regularized_thresholds(
                oof_probs[tr_m], Y_true[tr_m], shrinkage_lambda=cfg.shrinkage_lambda
            )
            oof_preds[va_m] = apply_thresholds_with_min_one_fallback(
                oof_probs[va_m], thrs_f, enforce_min_one=cfg.enforce_min_one_label
            )
            fold_thrs.append(thrs_f)
        shrunk_thrs = np.mean(fold_thrs, axis=0)
        _, global_thr, raw_thrs = optimize_regularized_thresholds(
            oof_probs, Y_true, shrinkage_lambda=cfg.shrinkage_lambda
        )
    else:
        shrunk_thrs, global_thr, raw_thrs = optimize_regularized_thresholds(
            oof_probs, Y_true, shrinkage_lambda=cfg.shrinkage_lambda
        )
        oof_preds = apply_thresholds_with_min_one_fallback(
            oof_probs, shrunk_thrs, enforce_min_one=cfg.enforce_min_one_label
        )

    opt_macro = float(f1_score(Y_true, oof_preds, average="macro", zero_division=0))
    opt_micro = float(f1_score(Y_true, oof_preds, average="micro", zero_division=0))
    test_preds = apply_thresholds_with_min_one_fallback(
        test_probs, shrunk_thrs, enforce_min_one=cfg.enforce_min_one_label
    )

    cal_oof = calibrate_margin_shifted_probs(oof_probs, shrunk_thrs)
    cal_test = calibrate_margin_shifted_probs(test_probs, shrunk_thrs)

    print("=" * 72)
    print("TASK 04 SUPPORT-REGULARIZED MULTILABEL THRESHOLD OPTIMIZATION")
    print("=" * 72)
    print(f"  Default (0.50) OOF  : Macro-F1={base_macro:.4f} | Micro-F1={base_micro:.4f}")
    print(f"  Global Optimal Thr  : tau_global={global_thr:.3f}")
    print(f"  Raw Per-Class Thrs  : {np.round(raw_thrs, 3)}")
    print(f"  Shrunk Class Thrs   : {np.round(shrunk_thrs, 3)} (lambda={cfg.shrinkage_lambda})")
    print(f"  Post-Processed OOF  : Macro-F1={opt_macro:.4f} | Micro-F1={opt_micro:.4f}")
    print(f"  Zero-Label Test Rows: {int((test_preds.sum(axis=1) == 0).sum())}/{len(test_preds)}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", cal_oof)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", cal_test)

    sub_df = pd.DataFrame({cfg.id_col: test_ids})
    for idx, col in enumerate(label_cols):
        sub_df[col] = test_preds[:, idx]
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    return cal_oof, cal_test, shrunk_thrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Multilabel Threshold Post-Processing")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--oof-path", type=str, default=CFG.oof_path)
    parser.add_argument("--test-probs-path", type=str, default=CFG.test_probs_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        oof_path=args.oof_path,
        test_probs_path=args.test_probs_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(cfg.seed)
        n_oof, n_test, n_labels = 250, 60, 5
        label_cols = [f"label_{i}" for i in range(n_labels)]
        # Imbalanced supports including a rare tail class (label_4)
        probs_prior = [0.45, 0.30, 0.20, 0.12, 0.03]
        Y_true = np.column_stack([rng.binomial(1, p, size=n_oof) for p in probs_prior])
        # Ensure every row has at least 1 label
        zero_m = Y_true.sum(axis=1) == 0
        Y_true[zero_m, 0] = 1
        folds = np.arange(n_oof) % 5

        logits = rng.normal(-1.4, 0.8, size=(n_oof, n_labels)) + Y_true * 2.1
        oof_probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        te_logits = rng.normal(-1.5, 0.9, size=(n_test, n_labels))
        test_probs = (1.0 / (1.0 + np.exp(-te_logits))).astype(np.float32)
        test_ids = np.array([f"te_{i}" for i in range(n_test)])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            cal_oof, cal_test, thrs = run_postprocess(
                oof_probs, test_probs, Y_true, test_ids, label_cols, cfg, folds=folds
            )
            assert cal_oof.shape == oof_probs.shape
            assert cal_test.shape == test_probs.shape
            assert len(thrs) == n_labels
            print("[SUCCESS] Task 04 post-processing smoke test passed.")
    else:
        p_tr = pathlib.Path(cfg.train_path)
        train_df = pd.read_parquet(p_tr) if p_tr.suffix.lower() == ".parquet" else pd.read_csv(p_tr)
        p_te = pathlib.Path(cfg.test_path)
        test_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
        label_cols = resolve_label_columns(train_df, cfg)
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        folds = train_df[cfg.fold_col].values if cfg.fold_col in train_df.columns else None
        run_postprocess(
            oof_probs,
            test_probs,
            train_df[label_cols].values,
            test_df[cfg.id_col].values,
            label_cols,
            cfg,
            folds=folds,
        )


if __name__ == "__main__":
    main()
