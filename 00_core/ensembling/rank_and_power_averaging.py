#!/usr/bin/env python3
"""
rank_and_power_averaging.py
===========================
Rank Averaging (`scipy.stats.rankdata` normalized to `(0, 1]`) and Power Averaging
(`sum_m w_m * p_m^gamma`) with Out-Of-Fold (OOF) `gamma` optimization for scale-free
metrics (ROC-AUC, PR-AUC, Spearman/Kendall rank correlation).
When models come from heterogeneous families (e.g. DeBERTa + LightGBM + CatBoost) with
mismatched probability calibration, Rank Averaging and Power Averaging prevent uncalibrated
models from dominating arithmetic means.

Usage:
  python3 00_core/ensembling/rank_and_power_averaging.py --oof-paths oof_1.npy oof_2.npy --test-paths test_1.npy test_2.npy --train-path train_folds.parquet
  python3 00_core/ensembling/rank_and_power_averaging.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    exp_name: str = "rank_power_blend"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = None
    target_col: str = "target"
    id_col: str = "id"
    oof_paths: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    output_dir: str = "."
    metric: str = "roc_auc"  # roc_auc | pr_auc
    gamma_min: float = 0.25
    gamma_max: float = 4.0
    gamma_steps: int = 61


def eval_rank_metric(y_true: np.ndarray, preds: np.ndarray, metric: str = "roc_auc") -> float:
    if metric.lower() == "pr_auc":
        return float(average_precision_score(y_true, preds))
    return float(roc_auc_score(y_true, preds))


def to_percentile_ranks(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64)
    if flat.ndim == 1:
        return rankdata(flat, method="average") / float(len(flat))
    return np.column_stack([
        rankdata(flat[:, c], method="average") / float(len(flat))
        for c in range(flat.shape[1])
    ])


def rank_average(arrays: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    n = len(arrays)
    w = np.full(n, 1.0 / n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, wi in zip(arrays, w):
        out += wi * to_percentile_ranks(arr)
    return out


def power_average(
    arrays: Sequence[np.ndarray],
    gamma: float = 2.0,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    n = len(arrays)
    w = np.full(n, 1.0 / n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, wi in zip(arrays, w):
        clipped = np.clip(arr.astype(np.float64), 1e-8, 1.0)
        out += wi * np.power(clipped, gamma)
    return out


def optimize_rank_and_power_blend(
    oof_list: Sequence[np.ndarray],
    y_true: np.ndarray,
    cfg: Optional[CFG] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = CFG()
    y = np.asarray(y_true).ravel()
    arith_oof = np.mean([a.astype(np.float64) for a in oof_list], axis=0)
    arith_score = eval_rank_metric(y, arith_oof, cfg.metric)

    rank_oof = rank_average(oof_list)
    rank_score = eval_rank_metric(y, rank_oof, cfg.metric)

    gammas = np.linspace(cfg.gamma_min, cfg.gamma_max, cfg.gamma_steps)
    best_gamma = 1.0
    best_power_score = arith_score
    best_power_oof = arith_oof
    for g in gammas:
        cand_oof = power_average(oof_list, gamma=float(g))
        sc = eval_rank_metric(y, cand_oof, cfg.metric)
        if sc > best_power_score:
            best_power_score = sc
            best_gamma = float(g)
            best_power_oof = cand_oof

    if rank_score >= best_power_score:
        best_mode = "rank_average"
        best_oof = rank_oof
        best_score = rank_score
    else:
        best_mode = "power_average"
        best_oof = best_power_oof
        best_score = best_power_score

    print("=" * 72)
    print(f"[RANK & POWER AVERAGING] metric={cfg.metric}")
    print(f"  Arithmetic Mean OOF {cfg.metric}: {arith_score:.6f}")
    print(f"  Rank Average OOF {cfg.metric}:    {rank_score:.6f}")
    print(f"  Power Average OOF {cfg.metric}:   {best_power_score:.6f} (best gamma={best_gamma:.3f})")
    print(f"  Selected Mode: {best_mode} -> {best_score:.6f}")
    print("=" * 72)

    return {
        "best_mode": best_mode,
        "best_gamma": best_gamma,
        "arith_score": arith_score,
        "rank_score": rank_score,
        "power_score": best_power_score,
        "best_score": best_score,
        "oof_blend": best_oof,
    }


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 450, 100
    y = rng.choice([0, 1], size=n_tr, p=[0.8, 0.2])
    s = y.astype(float)

    # Model 1 has well-scaled probabilities; Model 2 is compressed into [0.49, 0.51]
    # Arithmetic mean would almost completely ignore Model 2, whereas Rank Averaging rescues it!
    z1 = s * 1.3 + rng.normal(0, 1.0, size=n_tr)
    z2 = s * 1.5 + rng.normal(0, 1.0, size=n_tr)
    oof_1 = 1.0 / (1.0 + np.exp(-z1))
    oof_2 = 0.50 + 0.005 * np.tanh(z2)

    tests = [rng.uniform(0.1, 0.9, size=n_te), rng.uniform(0.495, 0.505, size=n_te)]

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = CFG(exp_name="rank_power_smoke", output_dir=tmpdir, metric="roc_auc")
        res = optimize_rank_and_power_blend([oof_1, oof_2], y, cfg)
        assert res["rank_score"] > res["arith_score"], "Rank average should beat arithmetic mean on miscalibrated scales"
        assert res["best_score"] >= res["arith_score"]

        if res["best_mode"] == "rank_average":
            te_blend = rank_average(tests)
        else:
            te_blend = power_average(tests, gamma=res["best_gamma"])

        oof_p = os.path.join(tmpdir, f"oof_{cfg.exp_name}.npy")
        te_p = os.path.join(tmpdir, f"test_{cfg.exp_name}.npy")
        sub_p = os.path.join(tmpdir, f"sub_{cfg.exp_name}.csv")
        np.save(oof_p, res["oof_blend"])
        np.save(te_p, te_blend)
        pd.DataFrame({"id": np.arange(n_te), "target": te_blend}).to_csv(sub_p, index=False)
        assert os.path.exists(oof_p) and os.path.exists(te_p) and os.path.exists(sub_p)
        print("[SMOKE TEST PASSED] rank_and_power_averaging.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Percentile Rank Averaging and OOF Power Averaging.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--oof-paths", type=str, nargs="*", default=[])
    parser.add_argument("--test-paths", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--metric", type=str, default=CFG.metric, choices=["roc_auc", "pr_auc"])
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test or not args.oof_paths:
        run_smoke_test()
    else:
        cfg = CFG(
            exp_name=args.exp_name,
            train_path=args.train_path,
            test_path=args.test_path,
            target_col=args.target_col,
            id_col=args.id_col,
            oof_paths=args.oof_paths,
            test_paths=args.test_paths,
            output_dir=args.output_dir,
            metric=args.metric,
        )
        p = Path(cfg.train_path)
        tr_df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        y = tr_df[cfg.target_col].to_numpy()
        oofs = [np.load(fp) for fp in cfg.oof_paths]
        res = optimize_rank_and_power_blend(oofs, y, cfg)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", res["oof_blend"])
        if cfg.test_paths:
            tests = [np.load(fp) for fp in cfg.test_paths]
            te_blend = (
                rank_average(tests)
                if res["best_mode"] == "rank_average"
                else power_average(tests, gamma=res["best_gamma"])
            )
            np.save(out_dir / f"test_{cfg.exp_name}.npy", te_blend)
            pd.DataFrame({cfg.id_col: np.arange(len(te_blend)), cfg.target_col: te_blend}).to_csv(
                out_dir / f"sub_{cfg.exp_name}.csv", index=False
            )
