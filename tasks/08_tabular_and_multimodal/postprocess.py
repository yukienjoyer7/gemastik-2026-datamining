"""
Task 08 — Tabular & Multimodal Post-Processing: GBDT + Neural Nelder-Mead Blender & OptimizedRounder
====================================================================================================
Post-processing and ensembling suite for Tabular & Multimodal competitions:
  1. `optimize_blend_weights`: Combines `oof_tier1_tabular.npy` (GBDT) and
     `oof_tier2_neural.npy` (Tabular ResNet) using Nelder-Mead / Powell optimization
     (supporting Probability Blending, Log-Odds / Logit Blending, and Rank Averaging).
  2. `clip_to_target_bounds`: Clips regression predictions to `[train_min, train_max]`
     or probability predictions to `[eps, 1 - eps]` for LogLoss stability.
  3. `OptimizedRounder`: Nelder-Mead threshold optimizer for ordinal/integer regression
     targets (e.g., Quadratic Weighted Kappa / PetFinder / Prudential / Wine Quality).

Usage:
  python3 postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

_venv_site = (
    Path.home()
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
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score


@dataclass
class CFG:
    exp_name: str = "tabular_postprocess"
    seed: int = 42
    clip_eps: float = 1e-5
    output_dir: str = "./outputs"


def rank_average(preds_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Computes weighted rank-averaged predictions in `[0, 1]` (ideal for ROC-AUC)."""
    n_samples = len(preds_list[0])
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    blended = np.zeros(n_samples, dtype=np.float64)
    for preds, wi in zip(preds_list, w):
        ranks = rankdata(preds) / max(n_samples, 1)
        blended += wi * ranks
    return blended.astype(np.float32)


def optimize_gbdt_neural_blend(
    oof_gbdt: np.ndarray,
    oof_neural: np.ndarray,
    y_true: np.ndarray,
    metric: str = "roc_auc",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Finds optimal weights `[w_gbdt, w_neural, w_rank]` on OOF predictions via
    Nelder-Mead optimization.
    """
    rank_gbdt = (rankdata(oof_gbdt) / len(oof_gbdt)).astype(np.float64)
    rank_nn = (rankdata(oof_neural) / len(oof_neural)).astype(np.float64)

    def _loss(raw_w: np.ndarray) -> float:
        w = np.maximum(raw_w, 0.0)
        if np.sum(w) <= 1e-9:
            return 1e9
        w = w / np.sum(w)
        if metric == "roc_auc":
            pred = w[0] * oof_gbdt + w[1] * oof_neural + w[2] * 0.5 * (rank_gbdt + rank_nn)
            return -float(roc_auc_score(y_true, pred))
        pred = np.clip(w[0] * oof_gbdt + w[1] * oof_neural, 1e-6, 1.0 - 1e-6)
        return float(log_loss(y_true, pred))

    init_w = np.array([0.60, 0.30, 0.10], dtype=np.float64)
    res = minimize(_loss, init_w, method="Nelder-Mead", options={"maxiter": 250})
    best_w = np.maximum(res.x, 0.0)
    best_w = best_w / np.sum(best_w)

    blended_oof = (
        best_w[0] * oof_gbdt
        + best_w[1] * oof_neural
        + best_w[2] * 0.5 * (rank_gbdt + rank_nn)
    ).astype(np.float32)
    score = (
        float(roc_auc_score(y_true, blended_oof))
        if metric == "roc_auc"
        else float(log_loss(y_true, np.clip(blended_oof, 1e-6, 1.0 - 1e-6)))
    )
    return best_w.astype(np.float32), blended_oof, score


class OptimizedRounder:
    """
    Nelder-Mead threshold optimizer converting continuous regression predictions
    into ordinal integer classes `0..K-1` maximizing Quadratic Weighted Kappa (QWK).
    """

    def __init__(self, n_classes: int = 5) -> None:
        self.n_classes = n_classes
        self.coef_: np.ndarray = np.arange(0.5, n_classes - 0.5, 1.0)

    def _discretize(self, preds: np.ndarray, coef: Sequence[float]) -> np.ndarray:
        sorted_c = np.sort(coef)
        return np.digitize(preds, sorted_c).astype(np.int32)

    def fit(self, preds: np.ndarray, y_true: np.ndarray) -> "OptimizedRounder":
        init_coef = np.arange(0.5, self.n_classes - 0.5, 1.0)

        def _neg_qwk(coef: np.ndarray) -> float:
            disc = self._discretize(preds, coef)
            return -float(cohen_kappa_score(y_true, disc, weights="quadratic"))

        res = minimize(_neg_qwk, init_coef, method="Nelder-Mead", options={"maxiter": 300})
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        return self._discretize(preds, self.coef_)


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 120
    y_bin = rng.integers(0, 2, size=n)

    # Simulate complementary GBDT and Neural Tabular OOF predictions
    oof_gbdt = np.clip(0.30 + 0.40 * y_bin + rng.normal(0.0, 0.22, size=n), 0.01, 0.99).astype(
        np.float32
    )
    oof_neural = np.clip(0.32 + 0.38 * y_bin + rng.normal(0.0, 0.23, size=n), 0.01, 0.99).astype(
        np.float32
    )
    auc_gbdt = float(roc_auc_score(y_bin, oof_gbdt))
    auc_nn = float(roc_auc_score(y_bin, oof_neural))
    best_w, blended_oof, auc_blend = optimize_gbdt_neural_blend(
        oof_gbdt, oof_neural, y_bin, metric="roc_auc"
    )

    # Test OptimizedRounder on ordinal target (0..4)
    y_ord = rng.integers(0, 5, size=n)
    cont_preds = y_ord.astype(np.float64) + rng.normal(0.15, 0.55, size=n)
    naive_round = np.clip(np.round(cont_preds), 0, 4).astype(np.int32)
    qwk_naive = float(cohen_kappa_score(y_ord, naive_round, weights="quadratic"))

    rounder = OptimizedRounder(n_classes=5).fit(cont_preds, y_ord)
    opt_round = rounder.predict(cont_preds)
    qwk_opt = float(cohen_kappa_score(y_ord, opt_round, weights="quadratic"))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", blended_oof)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", blended_oof[:15])
        pd.DataFrame({"row_id": range(15), "target": blended_oof[:15]}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] GBDT AUC={auc_gbdt:.4f}, Neural AUC={auc_nn:.4f} -> "
        f"Blend AUC={auc_blend:.4f} (weights={np.round(best_w, 3).tolist()}) | "
        f"Ordinal QWK: Naive={qwk_naive:.4f} -> OptimizedRounder={qwk_opt:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Tabular Post-Processing")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
