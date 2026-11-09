#!/usr/bin/env python3
"""
nelder_mead_blender.py
======================
Continuous Simplex Weight Optimization + Temperature Calibration (`scipy.optimize.minimize`
using SLSQP or Nelder-Mead) on Out-Of-Fold (OOF) predictions.
Supports:
  - Simplex-constrained blending (`sum(w) = 1`, `w_m in [min_w, 1.0]`)
  - Optional post-blend temperature scaling `T > 0` on logit probabilities (`sigmoid(logit(p) / T)`)
    for LogLoss / Brier calibration
  - Standardized output artifacts: `oof_<exp_name>.npy`, `test_<exp_name>.npy`, `sub_<exp_name>.csv`.

Usage:
  python3 00_core/ensembling/nelder_mead_blender.py --oof-paths oof_1.npy oof_2.npy --test-paths test_1.npy test_2.npy --train-path train_folds.parquet
  python3 00_core/ensembling/nelder_mead_blender.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    exp_name: str = "simplex_blend"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = None
    target_col: str = "target"
    id_col: str = "id"
    oof_paths: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    output_dir: str = "."
    objective: str = "logloss"  # logloss | rmse | roc_auc
    optimizer: str = "SLSQP"    # SLSQP | Nelder-Mead
    calibrate_temperature: bool = True
    min_weight: float = -0.15


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return expit(logit(p) / max(temperature, 1e-3))


def loss_function(
    y_true: np.ndarray,
    pred: np.ndarray,
    objective: str = "logloss",
) -> float:
    obj = objective.lower()
    if obj == "logloss":
        return float(log_loss(y_true, np.clip(pred, 1e-7, 1.0 - 1e-7)))
    if obj == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, pred)))
    if obj == "roc_auc":
        return -float(roc_auc_score(y_true, pred))
    raise ValueError(f"Unsupported objective: {objective}")


def fit_simplex_blender(
    oof_list: Sequence[np.ndarray],
    y_true: np.ndarray,
    cfg: Optional[CFG] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = CFG()
    n_models = len(oof_list)
    M = np.column_stack([arr.ravel().astype(np.float64) for arr in oof_list])
    y = np.asarray(y_true).ravel()

    init_w = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    equal_blend = M @ init_w
    equal_loss = loss_function(y, equal_blend, cfg.objective)

    if cfg.optimizer.upper() == "SLSQP":
        bounds = [(cfg.min_weight, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        res = minimize(
            lambda w: loss_function(y, M @ w, cfg.objective),
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        best_w = np.asarray(res.x, dtype=np.float64)
        best_w = best_w / np.sum(best_w)
    else:
        def _nm_obj(raw_w: np.ndarray) -> float:
            s = np.sum(raw_w)
            if abs(s) < 1e-8:
                return 1e9
            w = raw_w / s
            if np.any(w < cfg.min_weight):
                return 1e9
            return loss_function(y, M @ w, cfg.objective)

        res = minimize(
            _nm_obj,
            init_w,
            method="Nelder-Mead",
            options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-7},
        )
        best_w = np.asarray(res.x, dtype=np.float64)
        best_w = best_w / np.sum(best_w)

    oof_blend = M @ best_w
    opt_loss = loss_function(y, oof_blend, cfg.objective)

    best_temp = 1.0
    if cfg.calibrate_temperature and cfg.objective.lower() == "logloss":
        res_t = minimize(
            lambda t: loss_function(y, apply_temperature(oof_blend, float(t[0])), "logloss"),
            x0=np.array([1.0]),
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        cand_t = float(np.clip(res_t.x[0], 0.1, 10.0))
        cand_blend = apply_temperature(oof_blend, cand_t)
        cand_loss = loss_function(y, cand_blend, "logloss")
        if cand_loss < opt_loss:
            best_temp = cand_t
            oof_blend = cand_blend
            opt_loss = cand_loss

    print("=" * 72)
    print(f"[SIMPLEX BLENDER] optimizer={cfg.optimizer} | objective={cfg.objective}")
    print(f"Equal-weight loss: {equal_loss:.6f} -> Optimized loss: {opt_loss:.6f} | temp={best_temp:.4f}")
    print(f"Weights: {np.round(best_w, 5)}")
    print("=" * 72)

    return {
        "weights": best_w,
        "temperature": best_temp,
        "oof_blend": oof_blend,
        "equal_weight_loss": equal_loss,
        "optimized_loss": opt_loss,
    }


def transform_simplex_blender(
    test_list: Sequence[np.ndarray],
    weights: np.ndarray,
    temperature: float = 1.0,
    apply_temp: bool = True,
) -> np.ndarray:
    M_te = np.column_stack([arr.ravel().astype(np.float64) for arr in test_list])
    pred = M_te @ weights
    if apply_temp and abs(temperature - 1.0) > 1e-5:
        pred = apply_temperature(pred, temperature)
    return pred


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 400, 100
    y = rng.choice([0, 1], size=n_tr, p=[0.65, 0.35])
    true_logit = (y * 2.4 - 1.2)
    oof_1 = expit(true_logit + rng.normal(0, 0.8, size=n_tr))
    oof_2 = expit(true_logit * 1.4 + rng.normal(0, 1.0, size=n_tr))  # overconfident
    oof_3 = expit(rng.normal(0, 1.5, size=n_tr))                      # weak noisy model

    tests = [rng.uniform(0.05, 0.95, size=n_te) for _ in range(3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        for opt in ["SLSQP", "Nelder-Mead"]:
            cfg = CFG(
                exp_name=f"blend_{opt.lower()}",
                output_dir=tmpdir,
                objective="logloss",
                optimizer=opt,
                calibrate_temperature=True,
            )
            res = fit_simplex_blender([oof_1, oof_2, oof_3], y, cfg)
            assert res["optimized_loss"] <= res["equal_weight_loss"] + 1e-6
            assert np.isclose(np.sum(res["weights"]), 1.0, atol=1e-4)

            te_blend = transform_simplex_blender(tests, res["weights"], res["temperature"], apply_temp=True)
            oof_p = os.path.join(tmpdir, f"oof_{cfg.exp_name}.npy")
            te_p = os.path.join(tmpdir, f"test_{cfg.exp_name}.npy")
            sub_p = os.path.join(tmpdir, f"sub_{cfg.exp_name}.csv")
            np.save(oof_p, res["oof_blend"])
            np.save(te_p, te_blend)
            pd.DataFrame({"id": np.arange(n_te), "target": te_blend}).to_csv(sub_p, index=False)
            assert os.path.exists(oof_p) and os.path.exists(te_p) and os.path.exists(sub_p)
        print("[SMOKE TEST PASSED] nelder_mead_blender.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nelder-Mead / SLSQP Simplex Blender + Temperature Scaling.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--oof-paths", type=str, nargs="*", default=[])
    parser.add_argument("--test-paths", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--objective", type=str, default=CFG.objective, choices=["logloss", "rmse", "roc_auc"])
    parser.add_argument("--optimizer", type=str, default=CFG.optimizer, choices=["SLSQP", "Nelder-Mead"])
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
            objective=args.objective,
            optimizer=args.optimizer,
        )
        p = Path(cfg.train_path)
        tr_df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        y = tr_df[cfg.target_col].to_numpy()
        oofs = [np.load(fp) for fp in cfg.oof_paths]
        res = fit_simplex_blender(oofs, y, cfg)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", res["oof_blend"])
        if cfg.test_paths:
            tests = [np.load(fp) for fp in cfg.test_paths]
            te_blend = transform_simplex_blender(
                tests, res["weights"], res["temperature"], apply_temp=(cfg.objective == "logloss")
            )
            np.save(out_dir / f"test_{cfg.exp_name}.npy", te_blend)
            sub_df = pd.DataFrame({cfg.id_col: np.arange(len(te_blend)), cfg.target_col: te_blend})
            sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
