#!/usr/bin/env python3
"""
hill_climbing_ensemble.py
=========================
Greedy Forward Stepwise Out-Of-Fold (OOF) Hill Climbing Ensembler (Caruana et al. + Kaggle negative-weight extension).
Features:
  - Starts from the single strongest OOF model on the chosen competition metric.
  - Iteratively searches over all candidate models `m` and a fine weight grid `w in np.linspace(-0.25, 0.50, 151)`
    (when `allow_negative_weights=True`):
      `current_blend = (1 - w) * current_blend + w * oof_m`
  - Stops early when marginal OOF improvement `< tol` or `max_iters` is reached.
  - Accumulates exact net model weights `W` (`sum(W) = 1.0`) and applies them to `test_*.npy`,
    saving standardized `oof_<exp_name>.npy`, `test_<exp_name>.npy`, and `sub_<exp_name>.csv`.

Usage:
  python3 00_core/ensembling/hill_climbing_ensemble.py --oof-paths oof_m1.npy oof_m2.npy --test-paths test_m1.npy test_m2.npy --train-path train_folds.parquet --metric roc_auc
  python3 00_core/ensembling/hill_climbing_ensemble.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    exp_name: str = "hillclimb_blend"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = None
    target_col: str = "target"
    id_col: str = "id"
    oof_paths: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    output_dir: str = "."
    metric: str = "roc_auc"  # roc_auc | pr_auc | neg_logloss | neg_rmse | macro_f1
    allow_negative_weights: bool = True
    min_weight: float = -0.25
    max_weight: float = 0.50
    grid_size: int = 151
    max_iters: int = 50
    tol: float = 1e-6


def get_metric_fn(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns a metric function where HIGHER is always BETTER."""
    m = metric.lower()
    if m == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    if m == "pr_auc":
        return lambda y, p: float(average_precision_score(y, p))
    if m in {"logloss", "neg_logloss"}:
        return lambda y, p: -float(log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7)))
    if m in {"rmse", "neg_rmse"}:
        return lambda y, p: -float(np.sqrt(mean_squared_error(y, p)))
    if m == "macro_f1":
        def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
            preds = np.argmax(p, axis=1) if p.ndim == 2 else (p >= 0.5).astype(int)
            return float(f1_score(y, preds, average="macro", zero_division=0))
        return _macro_f1
    raise ValueError(f"Unsupported metric: {metric}")


def fit_hill_climbing(
    oof_list: Sequence[np.ndarray],
    y_true: np.ndarray,
    model_names: Optional[Sequence[str]] = None,
    cfg: Optional[CFG] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = CFG()
    n_models = len(oof_list)
    if n_models == 0:
        raise ValueError("`oof_list` must contain at least 1 model OOF array.")
    names = list(model_names) if model_names is not None else [f"model_{i}" for i in range(n_models)]
    metric_fn = get_metric_fn(cfg.metric)

    individual_scores = [metric_fn(y_true, oof) for oof in oof_list]
    best_init_idx = int(np.argmax(individual_scores))
    best_score = float(individual_scores[best_init_idx])
    current_blend = oof_list[best_init_idx].astype(np.float64).copy()

    weights = np.zeros(n_models, dtype=np.float64)
    weights[best_init_idx] = 1.0

    low_w = cfg.min_weight if cfg.allow_negative_weights else 0.01
    weight_grid = np.linspace(low_w, cfg.max_weight, cfg.grid_size)
    weight_grid = weight_grid[np.abs(weight_grid) > 1e-9]

    history: List[Dict[str, Any]] = [{
        "step": 0,
        "added_model": names[best_init_idx],
        "step_weight": 1.0,
        "oof_score": round(best_score, 7),
        "improvement": 0.0,
    }]

    print("=" * 72)
    print(f"[HILL CLIMBING] metric={cfg.metric} | allow_negative={cfg.allow_negative_weights} | grid={len(weight_grid)}")
    for idx, (nm, sc) in enumerate(zip(names, individual_scores)):
        marker = " <-- INIT BEST" if idx == best_init_idx else ""
        print(f"  Candidate [{idx}] {nm:24s}: {sc:.6f}{marker}")
    print("-" * 72)

    for step in range(1, cfg.max_iters + 1):
        step_best_score = best_score
        step_best_idx = -1
        step_best_w = 0.0
        step_best_blend = current_blend

        for m_idx in range(n_models):
            cand_oof = oof_list[m_idx]
            for w in weight_grid:
                trial_blend = (1.0 - w) * current_blend + w * cand_oof
                sc = metric_fn(y_true, trial_blend)
                if sc > step_best_score:
                    step_best_score = sc
                    step_best_idx = m_idx
                    step_best_w = float(w)
                    step_best_blend = trial_blend

        improvement = step_best_score - best_score
        if step_best_idx == -1 or improvement < cfg.tol:
            print(f"  Step {step:2d}: Converged (max improvement {max(improvement, 0.0):.8f} < tol {cfg.tol}).")
            break

        current_blend = step_best_blend
        best_score = step_best_score
        weights = (1.0 - step_best_w) * weights
        weights[step_best_idx] += step_best_w

        history.append({
            "step": step,
            "added_model": names[step_best_idx],
            "step_weight": round(step_best_w, 4),
            "oof_score": round(best_score, 7),
            "improvement": round(improvement, 7),
        })
        print(f"  Step {step:2d}: + {names[step_best_idx]:20s} (w={step_best_w:+.4f}) -> OOF {cfg.metric}={best_score:.6f} (+{improvement:.6f})")

    print("-" * 72)
    print("Final Portfolio Weights:")
    for nm, w in zip(names, weights):
        if abs(w) > 1e-6:
            print(f"  {nm:24s}: {w:+.5f}")
    print("=" * 72)

    return {
        "weights": weights,
        "model_names": names,
        "oof_blend": current_blend,
        "best_init_score": individual_scores[best_init_idx],
        "final_oof_score": best_score,
        "history": history,
    }


def apply_weights(arrays: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    blend = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, w in zip(arrays, weights):
        if abs(w) > 1e-9:
            blend += w * arr.astype(np.float64)
    return blend


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 500, 120
    y = rng.choice([0, 1], size=n_tr, p=[0.7, 0.3])
    signal = y.astype(float)

    # 3 diverse base models + 1 noisy/redundant model where negative weight helps
    oof_1 = np.clip(0.55 * signal + rng.normal(0.2, 0.30, size=n_tr), 0.01, 0.99)
    oof_2 = np.clip(0.50 * signal + rng.normal(0.25, 0.28, size=n_tr), 0.01, 0.99)
    oof_3 = np.clip(0.48 * signal + rng.normal(0.22, 0.29, size=n_tr), 0.01, 0.99)
    bias_noise = rng.normal(0.0, 0.25, size=n_tr)
    oof_1_biased = np.clip(oof_1 + bias_noise, 0.01, 0.99)
    oof_4_corr_noise = np.clip(0.10 * signal + bias_noise + rng.normal(0.3, 0.1, size=n_tr), 0.01, 0.99)

    oofs = [oof_1_biased, oof_2, oof_3, oof_4_corr_noise]
    tests = [rng.uniform(0.1, 0.9, size=n_te) for _ in range(4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = CFG(
            exp_name="hc_smoke",
            output_dir=tmpdir,
            metric="roc_auc",
            allow_negative_weights=True,
        )
        res = fit_hill_climbing(oofs, y, model_names=["lgbm", "catboost", "xgb", "correlated_bias"], cfg=cfg)
        assert res["final_oof_score"] >= res["best_init_score"]
        assert np.isclose(np.sum(res["weights"]), 1.0, atol=1e-5)

        test_blend = apply_weights(tests, res["weights"])
        oof_path = os.path.join(tmpdir, f"oof_{cfg.exp_name}.npy")
        test_path = os.path.join(tmpdir, f"test_{cfg.exp_name}.npy")
        sub_path = os.path.join(tmpdir, f"sub_{cfg.exp_name}.csv")
        np.save(oof_path, res["oof_blend"])
        np.save(test_path, test_blend)
        pd.DataFrame({"id": np.arange(n_te), "target": test_blend}).to_csv(sub_path, index=False)

        assert os.path.exists(oof_path) and os.path.exists(test_path) and os.path.exists(sub_path)
        print("[SMOKE TEST PASSED] hill_climbing_ensemble.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy forward stepwise OOF Hill Climbing Ensembler.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--oof-paths", type=str, nargs="*", default=[])
    parser.add_argument("--test-paths", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--metric", type=str, default=CFG.metric)
    parser.add_argument("--no-negative-weights", action="store_true")
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
            allow_negative_weights=not args.no_negative_weights,
        )
        p = Path(cfg.train_path)
        tr_df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        y = tr_df[cfg.target_col].to_numpy()
        oofs = [np.load(fp) for fp in cfg.oof_paths]
        names = [Path(fp).stem for fp in cfg.oof_paths]
        res = fit_hill_climbing(oofs, y, model_names=names, cfg=cfg)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", res["oof_blend"])
        if cfg.test_paths:
            tests = [np.load(fp) for fp in cfg.test_paths]
            test_blend = apply_weights(tests, res["weights"])
            np.save(out_dir / f"test_{cfg.exp_name}.npy", test_blend)
            sub_df = pd.DataFrame({cfg.id_col: np.arange(len(test_blend)), cfg.target_col: test_blend})
            if cfg.test_path and os.path.exists(cfg.test_path):
                tp = Path(cfg.test_path)
                te_df = pd.read_parquet(tp) if tp.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(tp)
                if cfg.id_col in te_df.columns:
                    sub_df[cfg.id_col] = te_df[cfg.id_col].values
            sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
