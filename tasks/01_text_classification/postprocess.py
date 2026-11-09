#!/usr/bin/env python3
"""
Task 01 — Text Classification: Post-Processing & Pseudo-Labeling
================================================================
Standalone post-processing pipeline providing:
  1. `TemperatureScaler`: Optimizes a scalar temperature `T > 0` on OOF logits/probabilities
     to minimize Negative Log-Likelihood (LogLoss) via L-BFGS-B.
  2. `optimize_class_multipliers`: Nelder-Mead / Powell simplex optimization of per-class
     probability multipliers `w_c` (`argmax_c (w_c * P(y=c | x))`) to directly maximize
     Out-Of-Fold Macro-F1 on imbalanced multiclass distributions.
  3. `select_pseudo_labels`: High-confidence pseudo-label selector with optional class-balance
     cap and entropy filtering (`select_pseudo_labels(test_probs, high_conf_thr=0.95)`).
  4. Standardized OOF & prediction contract (`oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     `sub_<exp_name>.csv`).

Usage:
    python postprocess.py --oof-path oof_tier2_deberta.npy --test-probs-path test_tier2_deberta.npy
    python postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import f1_score, log_loss


@dataclass
class CFG:
    """Configuration for Task 01 Post-Processing and Pseudo-Labeling."""

    exp_name: str = "tier1_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_tfidf.npy"
    test_probs_path: str = "test_tier1_tfidf.npy"
    output_dir: str = "."
    id_col: str = "id"
    target_col: str = "label"
    fold_col: str = "fold"

    high_conf_thr: float = 0.95
    max_nelder_mead_iter: int = 300
    seed: int = 42
    smoke_test: bool = False


def probs_to_log_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return np.log(clipped)


class TemperatureScaler:
    """Single-parameter Temperature Scaling (`softmax(log(p) / T)`) for LogLoss calibration."""

    def __init__(self, init_temp: float = 1.0) -> None:
        self.temperature: float = init_temp

    def fit(self, probs_or_logits: np.ndarray, y_true: np.ndarray, is_probs: bool = True) -> "TemperatureScaler":
        logits = probs_to_log_probs(probs_or_logits) if is_probs else np.asarray(probs_or_logits, dtype=np.float64)

        def nll_obj(t_arr: np.ndarray) -> float:
            temp = max(float(t_arr[0]), 1e-3)
            scaled = softmax(logits / temp, axis=1)
            return float(log_loss(y_true, scaled))

        res = minimize(
            nll_obj,
            x0=np.array([self.temperature], dtype=np.float64),
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )
        self.temperature = float(res.x[0])
        return self

    def transform(self, probs_or_logits: np.ndarray, is_probs: bool = True) -> np.ndarray:
        logits = probs_to_log_probs(probs_or_logits) if is_probs else np.asarray(probs_or_logits, dtype=np.float64)
        return softmax(logits / max(self.temperature, 1e-3), axis=1)


def apply_class_multipliers(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    scaled = probs * multipliers.reshape(1, -1)
    denom = np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)
    return scaled / denom


def optimize_class_multipliers(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    folds: Optional[np.ndarray] = None,
    max_iter: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimizes class probability multipliers `w in R^C` using coordinate-wise + Nelder-Mead
    search to maximize Macro-F1. If `folds` is provided, fits multipliers out-of-fold
    to prevent threshold overfitting.
    """
    num_classes = oof_probs.shape[1]

    def _fit_single(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        best_w = np.ones(num_classes, dtype=np.float64)
        best_score = float(f1_score(targets, np.argmax(probs, axis=1), average="macro"))

        # Coordinate grid warm-start
        grid = np.linspace(0.4, 2.2, 19)
        for c in range(num_classes):
            for val in grid:
                cand_w = best_w.copy()
                cand_w[c] = val
                preds = np.argmax(probs * cand_w.reshape(1, -1), axis=1)
                score = float(f1_score(targets, preds, average="macro"))
                if score > best_score:
                    best_score = score
                    best_w = cand_w

        # Nelder-Mead simplex refinement
        def neg_macro_f1(w: np.ndarray) -> float:
            if np.any(w <= 0.05) or np.any(w > 5.0):
                return 1.0
            preds = np.argmax(probs * w.reshape(1, -1), axis=1)
            return -float(f1_score(targets, preds, average="macro"))

        res = minimize(
            neg_macro_f1,
            x0=best_w,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-4},
        )
        if -float(res.fun) >= best_score:
            best_w = np.asarray(res.x, dtype=np.float64)
        return best_w / np.mean(best_w)

    if folds is None:
        w = _fit_single(oof_probs, y_true)
        return w, apply_class_multipliers(oof_probs, w)

    unique_folds = np.sort(np.unique(folds))
    oof_adjusted = np.zeros_like(oof_probs)
    fold_weights = []
    for f in unique_folds:
        tr_mask = folds != f
        va_mask = folds == f
        w_f = _fit_single(oof_probs[tr_mask], y_true[tr_mask])
        oof_adjusted[va_mask] = apply_class_multipliers(oof_probs[va_mask], w_f)
        fold_weights.append(w_f)

    mean_weights = np.mean(fold_weights, axis=0)
    return mean_weights, oof_adjusted


def select_pseudo_labels(
    test_probs: np.ndarray,
    high_conf_thr: float = 0.95,
    max_per_class: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Selects high-confidence test samples (`max_c P(y=c|x) >= high_conf_thr`) for
    Stage-2 semi-supervised retraining.
    Returns a dictionary containing `indices`, `pseudo_labels`, and `confidences`.
    """
    confidences = np.max(test_probs, axis=1)
    pred_labels = np.argmax(test_probs, axis=1)
    selected_mask = confidences >= high_conf_thr
    selected_indices = np.where(selected_mask)[0]

    if max_per_class is not None and len(selected_indices) > 0:
        kept_indices = []
        for c in np.unique(pred_labels[selected_indices]):
            cls_idx = selected_indices[pred_labels[selected_indices] == c]
            order = np.argsort(confidences[cls_idx])[::-1]
            kept_indices.append(cls_idx[order[:max_per_class]])
        selected_indices = np.sort(np.concatenate(kept_indices))

    return {
        "indices": selected_indices,
        "pseudo_labels": pred_labels[selected_indices],
        "confidences": confidences[selected_indices],
    }


def run_postprocess(
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: np.ndarray,
    test_ids: np.ndarray,
    folds: Optional[np.ndarray],
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    raw_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro"))
    raw_ll = float(log_loss(y_true, oof_probs))

    scaler = TemperatureScaler().fit(oof_probs, y_true, is_probs=True)
    cal_oof = scaler.transform(oof_probs, is_probs=True)
    cal_test = scaler.transform(test_probs, is_probs=True)
    cal_ll = float(log_loss(y_true, cal_oof))

    multipliers, adj_oof = optimize_class_multipliers(
        cal_oof, y_true, folds=folds, max_iter=cfg.max_nelder_mead_iter, seed=cfg.seed
    )
    adj_test = apply_class_multipliers(cal_test, multipliers)
    adj_f1 = float(f1_score(y_true, np.argmax(adj_oof, axis=1), average="macro"))

    pseudo = select_pseudo_labels(cal_test, high_conf_thr=cfg.high_conf_thr)

    print("=" * 72)
    print("TASK 01 POST-PROCESSING & PSEUDO-LABELING SUMMARY")
    print("=" * 72)
    print(f"  Raw OOF        : Macro-F1={raw_f1:.4f} | LogLoss={raw_ll:.4f}")
    print(f"  Temp Scaled    : T={scaler.temperature:.4f}   | LogLoss={cal_ll:.4f}")
    print(f"  Multiplier Opt : Macro-F1={adj_f1:.4f} | Multipliers={np.round(multipliers, 3)}")
    print(
        f"  Pseudo-Labels  : {len(pseudo['indices'])}/{len(test_probs)} test samples "
        f"(>= {cfg.high_conf_thr:.2f} confidence)"
    )

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", adj_oof.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", adj_test.astype(np.float32))
    sub_df = pd.DataFrame({cfg.id_col: test_ids, cfg.target_col: np.argmax(adj_test, axis=1)})
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    return adj_oof, adj_test, pseudo


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Post-Processing & Pseudo-Labeling")
    parser.add_argument("--oof-path", type=str, default=CFG.oof_path)
    parser.add_argument("--test-probs-path", type=str, default=CFG.test_probs_path)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--high-conf-thr", type=float, default=CFG.high_conf_thr)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        oof_path=args.oof_path,
        test_probs_path=args.test_probs_path,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        high_conf_thr=args.high_conf_thr,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(cfg.seed)
        n_oof, n_test, n_classes = 300, 100, 3
        y_true = rng.choice([0, 1, 2], size=n_oof, p=[0.60, 0.25, 0.15])
        folds = np.arange(n_oof) % 5

        # Simulate miscalibrated and class-biased logits
        logits = rng.normal(0, 1.2, size=(n_oof, n_classes))
        for i, y_i in enumerate(y_true):
            logits[i, y_i] += 2.2
        logits[:, 0] += 0.8  # artificial majority bias
        oof_probs = softmax(logits * 2.5, axis=1)

        test_logits = rng.normal(0, 1.2, size=(n_test, n_classes))
        for i in range(n_test):
            test_logits[i, i % n_classes] += 3.5
        test_probs = softmax(test_logits, axis=1)
        test_ids = np.array([f"te_{i}" for i in range(n_test)])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            adj_oof, adj_test, pseudo = run_postprocess(
                oof_probs, test_probs, y_true, test_ids, folds, cfg
            )
            assert adj_oof.shape == oof_probs.shape
            assert adj_test.shape == test_probs.shape
            assert len(pseudo["indices"]) > 0
            print("[SUCCESS] Post-processing smoke test passed.")
    else:
        p_tr = pathlib.Path(cfg.train_path)
        train_df = pd.read_parquet(p_tr) if p_tr.suffix.lower() == ".parquet" else pd.read_csv(p_tr)
        p_te = pathlib.Path(cfg.test_path)
        test_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        folds = train_df[cfg.fold_col].values if cfg.fold_col in train_df.columns else None
        run_postprocess(
            oof_probs,
            test_probs,
            train_df[cfg.target_col].values,
            test_df[cfg.id_col].values,
            folds,
            cfg,
        )


if __name__ == "__main__":
    main()
