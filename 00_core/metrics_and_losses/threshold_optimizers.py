#!/usr/bin/env python3
"""
threshold_optimizers.py
=======================
Out-Of-Fold (OOF) Post-Processing Threshold, Multiplier, and Ordinal Cutpoint Optimizers.
Includes:
  1. `optimize_binary_threshold(oof_probs, y_true, metric="f1"|"mcc"|"fbeta"|"jaccard")`
  2. `optimize_multiclass_multipliers(oof_probs, y_true, method="Powell"|"Nelder-Mead")` for Macro-F1
  3. `optimize_multilabel_thresholds(oof_probs, y_true, min_pos_shrinkage=20)` with Empirical Bayes
     shrinkage toward the global threshold for rare labels to prevent OOF threshold overfitting
  4. `OptimizedRounder` for Quadratic Weighted Kappa (QWK) continuous-to-ordinal cutpoint tuning via Nelder-Mead

Usage:
  python3 00_core/metrics_and_losses/threshold_optimizers.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

from scipy.optimize import minimize
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    jaccard_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    grid_size: int = 199
    min_thresh: float = 0.01
    max_thresh: float = 0.99
    min_pos_shrinkage: int = 20
    multiclass_method: str = "Powell"
    fbeta: float = 2.0
    seed: int = 42


def _eval_binary_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    beta: float = 2.0,
) -> float:
    m = metric.lower()
    if m == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if m == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if m == "fbeta":
        return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
    if m == "jaccard":
        return float(jaccard_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported binary metric: {metric}")


def optimize_binary_threshold(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
    beta: float = 2.0,
    grid_size: int = 199,
) -> Dict[str, float]:
    """
    Finds the probability threshold t* in [0.01, 0.99] maximizing `metric` on OOF predictions.
    """
    probs = np.asarray(oof_probs, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    default_score = _eval_binary_metric(y, (probs >= 0.5).astype(int), metric=metric, beta=beta)

    thresholds = np.linspace(0.01, 0.99, grid_size)
    best_t = 0.5
    best_score = default_score
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = _eval_binary_metric(y, preds, metric=metric, beta=beta)
        if score > best_score:
            best_score = score
            best_t = float(t)

    return {
        "best_threshold": round(best_t, 5),
        "default_05_score": round(default_score, 6),
        "optimized_score": round(best_score, 6),
        "gain": round(best_score - default_score, 6),
    }


def optimize_multiclass_multipliers(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    method: str = "Powell",
    maxiter: int = 500,
) -> Tuple[np.ndarray, float, float]:
    """
    Optimizes class probability multipliers w in R^C such that argmax_c (w_c * p_{i,c})
    maximizes Macro-F1 on OOF predictions using coordinate/simplex optimization (Powell or Nelder-Mead).
    """
    P = np.asarray(oof_probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=int).ravel()
    n_classes = P.shape[1]

    raw_preds = np.argmax(P, axis=1)
    baseline_f1 = float(f1_score(y, raw_preds, average="macro", zero_division=0))

    def loss_func(weights: np.ndarray) -> float:
        scaled = P * weights[None, :]
        preds = np.argmax(scaled, axis=1)
        return -float(f1_score(y, preds, average="macro", zero_division=0))

    init_w = np.ones(n_classes, dtype=np.float64)
    res = minimize(
        loss_func,
        init_w,
        method=method,
        options={"maxiter": maxiter, "disp": False},
    )
    best_w = np.asarray(res.x, dtype=np.float64)
    best_w = best_w / max(np.mean(np.abs(best_w)), 1e-8)
    opt_f1 = -float(loss_func(best_w))
    if opt_f1 < baseline_f1:
        return init_w, baseline_f1, baseline_f1
    return best_w, baseline_f1, opt_f1


def optimize_multilabel_thresholds(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    min_pos_shrinkage: int = 20,
    grid_size: int = 99,
) -> Dict[str, object]:
    """
    Per-class threshold optimization for Macro-F1 in multilabel classification with
    Empirical Bayes shrinkage toward the global optimum threshold for rare classes:
      t_shrunk[c] = alpha_c * t_class[c] + (1 - alpha_c) * t_global
      where alpha_c = n_pos[c] / (n_pos[c] + min_pos_shrinkage)
    """
    P = np.asarray(oof_probs, dtype=np.float64)
    Y = np.asarray(y_true, dtype=int)
    n_classes = P.shape[1]
    thresholds_grid = np.linspace(0.05, 0.95, grid_size)

    baseline_macro_f1 = float(f1_score(Y, (P >= 0.5).astype(int), average="macro", zero_division=0))

    # 1. Global threshold optimizing Macro-F1 across all classes simultaneously
    best_global_t = 0.5
    best_global_f1 = baseline_macro_f1
    for t in thresholds_grid:
        score = float(f1_score(Y, (P >= t).astype(int), average="macro", zero_division=0))
        if score > best_global_f1:
            best_global_f1 = score
            best_global_t = float(t)

    # 2. Per-class threshold + Empirical Bayes shrinkage for rare positive classes
    shrunk_thresholds = np.full(n_classes, best_global_t, dtype=np.float64)
    raw_thresholds = np.full(n_classes, best_global_t, dtype=np.float64)
    for c in range(n_classes):
        y_c = Y[:, c]
        p_c = P[:, c]
        n_pos = int(y_c.sum())
        if n_pos == 0:
            continue
        best_tc = best_global_t
        best_fc = float(f1_score(y_c, (p_c >= best_global_t).astype(int), zero_division=0))
        for t in thresholds_grid:
            fc = float(f1_score(y_c, (p_c >= t).astype(int), zero_division=0))
            if fc > best_fc:
                best_fc = fc
                best_tc = float(t)
        raw_thresholds[c] = best_tc
        alpha = n_pos / float(n_pos + max(min_pos_shrinkage, 1))
        shrunk_thresholds[c] = alpha * best_tc + (1.0 - alpha) * best_global_t

    preds_shrunk = (P >= shrunk_thresholds[None, :]).astype(int)
    opt_macro_f1 = float(f1_score(Y, preds_shrunk, average="macro", zero_division=0))

    return {
        "global_threshold": round(best_global_t, 4),
        "raw_thresholds": np.round(raw_thresholds, 4),
        "shrunk_thresholds": np.round(shrunk_thresholds, 4),
        "baseline_05_macro_f1": round(baseline_macro_f1, 6),
        "global_opt_macro_f1": round(best_global_f1, 6),
        "shrunk_per_class_macro_f1": round(opt_macro_f1, 6),
    }


class OptimizedRounder:
    """
    Nelder-Mead cutpoint optimizer converting continuous regression predictions into
    ordinal integer classes `0 .. K-1` to maximize Quadratic Weighted Kappa (QWK).
    """

    def __init__(self, n_classes: int, init_coef: Optional[Sequence[float]] = None) -> None:
        self.n_classes = n_classes
        if init_coef is not None:
            self.coef_ = np.asarray(init_coef, dtype=np.float64)
        else:
            self.coef_ = np.arange(0.5, n_classes - 0.5, 1.0, dtype=np.float64)

    @staticmethod
    def _discretize(preds: np.ndarray, coef: np.ndarray) -> np.ndarray:
        sorted_coef = np.sort(coef)
        return np.searchsorted(sorted_coef, preds).astype(int)

    def _kappa_loss(self, coef: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> float:
        discrete = self._discretize(preds, coef)
        return -float(cohen_kappa_score(y_true, discrete, weights="quadratic"))

    def fit(self, preds: np.ndarray, y_true: np.ndarray, use_quantiles_init: bool = True) -> "OptimizedRounder":
        p = np.asarray(preds, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=int).ravel()
        if use_quantiles_init:
            # Initialize cutpoints using empirical ground-truth class cumulative proportions
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            cum_props = np.cumsum(counts)[:-1] / max(len(y), 1)
            cum_props = np.clip(cum_props, 1e-3, 1.0 - 1e-3)
            self.coef_ = np.quantile(p, cum_props)

        res = minimize(
            self._kappa_loss,
            self.coef_,
            args=(p, y),
            method="Nelder-Mead",
            options={"maxiter": 600, "disp": False},
        )
        self.coef_ = np.sort(np.asarray(res.x, dtype=np.float64))
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        return self._discretize(np.asarray(preds, dtype=np.float64).ravel(), self.coef_)

    def score(self, preds: np.ndarray, y_true: np.ndarray) -> float:
        y_hat = self.predict(preds)
        return float(cohen_kappa_score(np.asarray(y_true, dtype=int).ravel(), y_hat, weights="quadratic"))


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n = 600

    # 1. Binary threshold test (imbalanced positive class)
    y_bin = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    oof_bin = np.clip(y_bin * 0.35 + rng.uniform(0.0, 0.55, size=n), 0.01, 0.99)
    bin_res = optimize_binary_threshold(oof_bin, y_bin, metric="f1")
    print(f"[Binary F1 Optimizer] {bin_res}")
    assert bin_res["optimized_score"] >= bin_res["default_05_score"]

    # 2. Multiclass Macro-F1 multiplier optimization
    y_mc = rng.choice([0, 1, 2], size=n, p=[0.65, 0.25, 0.10])
    logits_mc = rng.normal(0.0, 1.0, size=(n, 3))
    for i in range(n):
        logits_mc[i, y_mc[i]] += 1.2
    logits_mc[:, 2] -= 0.7  # Under-confident minority class
    exp_l = np.exp(logits_mc - logits_mc.max(axis=1, keepdims=True))
    oof_mc = exp_l / exp_l.sum(axis=1, keepdims=True)
    w_mc, base_f1, opt_f1 = optimize_multiclass_multipliers(oof_mc, y_mc)
    print(f"[Multiclass Multipliers] base_macro_f1={base_f1:.5f} -> opt_macro_f1={opt_f1:.5f} | weights={np.round(w_mc, 3)}")
    assert opt_f1 >= base_f1

    # 3. Multilabel thresholds with Empirical Bayes shrinkage
    Y_ml = np.column_stack([
        rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        rng.choice([0, 1], size=n, p=[0.97, 0.03]),  # rare class
    ])
    P_ml = np.clip(Y_ml * 0.4 + rng.uniform(0.0, 0.55, size=(n, 3)), 0.01, 0.99)
    ml_res = optimize_multilabel_thresholds(P_ml, Y_ml, min_pos_shrinkage=20)
    print(f"[Multilabel Optimizer] base={ml_res['baseline_05_macro_f1']:.5f} -> shrunk={ml_res['shrunk_per_class_macro_f1']:.5f} | thresholds={ml_res['shrunk_thresholds']}")
    assert ml_res["shrunk_per_class_macro_f1"] >= ml_res["baseline_05_macro_f1"]

    # 4. QWK OptimizedRounder
    y_ord = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    pred_cont = y_ord.astype(float) * 0.85 + 0.4 + rng.normal(0.0, 0.45, size=n)
    rounder = OptimizedRounder(n_classes=5)
    base_qwk = rounder.score(pred_cont, y_ord)
    rounder.fit(pred_cont, y_ord)
    opt_qwk = rounder.score(pred_cont, y_ord)
    print(f"[OptimizedRounder QWK] base_qwk={base_qwk:.5f} -> opt_qwk={opt_qwk:.5f} | cutpoints={np.round(rounder.coef_, 4)}")
    assert opt_qwk >= base_qwk
    print("[SMOKE TEST PASSED] threshold_optimizers.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF threshold, multiplier, and QWK cutpoint optimizers.")
    parser.add_argument("--oof-path", type=str, default=None, help="Path to oof_*.npy")
    parser.add_argument("--y-path", type=str, default=None, help="Path to ground truth .npy")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass", "multilabel", "qwk"])
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test or not args.oof_path:
        run_smoke_test()
    else:
        oof = np.load(args.oof_path)
        y = np.load(args.y_path)
        if args.task == "binary":
            print(optimize_binary_threshold(oof, y, metric=args.metric))
        elif args.task == "multiclass":
            w, b, o = optimize_multiclass_multipliers(oof, y)
            print({"weights": w.tolist(), "baseline_f1": b, "optimized_f1": o})
        elif args.task == "multilabel":
            print(optimize_multilabel_thresholds(oof, y))
        elif args.task == "qwk":
            n_cls = int(np.max(y)) + 1
            r = OptimizedRounder(n_classes=n_cls).fit(oof, y)
            print({"cutpoints": r.coef_.tolist(), "qwk": r.score(oof, y)})
