#!/usr/bin/env python3
"""
stacking_meta_learner.py
========================
Leak-Free 2nd-Stage Out-Of-Fold (OOF) Stacking Meta-Learner.
Features:
  - Uses the EXACT `fold` indices from `train_folds.parquet` to train the 2nd-stage meta-learner
    strictly out-of-fold (preventing target leakage and meta-overfitting).
  - Supports multiple meta-learner families (`--meta-model`):
      * `logreg`  — L2-regularized `LogisticRegression`
      * `ridge`   — `Ridge` / `RidgeClassifier`
      * `mlp`     — Shallow 2-layer `MLPClassifier` / `MLPRegressor` with weight decay & early stopping
  - Constructs rich row-wise meta-features across base models:
      * Raw predictions + clipped logit transforms `log(p / (1 - p))`
      * Row-wise summary statistics (`mean`, `std`, `min`, `max`, `max - min` spread)
      * Pairwise differences between top base models
  - Outputs standardized `oof_<exp_name>.npy`, `test_<exp_name>.npy`, and `sub_<exp_name>.csv`.

Usage:
  python3 00_core/ensembling/stacking_meta_learner.py --oof-paths oof_1.npy oof_2.npy --test-paths test_1.npy test_2.npy --train-path train_folds.parquet --meta-model logreg
  python3 00_core/ensembling/stacking_meta_learner.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    exp_name: str = "stacking_meta"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = None
    target_col: str = "target"
    id_col: str = "id"
    fold_col: str = "fold"
    oof_paths: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    output_dir: str = "."
    task: str = "binary"          # binary | regression
    meta_model: str = "logreg"    # logreg | ridge | mlp
    use_nonlinear_meta_features: bool = True
    reg_c: float = 1.0
    seed: int = 42


def build_meta_features(
    pred_list: Sequence[np.ndarray],
    task: str = "binary",
    use_nonlinear: bool = True,
) -> np.ndarray:
    """
    Constructs 2nd-stage meta-feature matrix from a list of 1D (or 2D flattened) base model predictions.
    """
    M = np.column_stack([np.asarray(a, dtype=np.float64).reshape(len(a), -1) for a in pred_list])
    if not use_nonlinear or M.shape[1] < 2:
        return M

    feats = [M]
    if task.lower() == "binary":
        clipped = np.clip(M, 1e-5, 1.0 - 1e-5)
        feats.append(logit(clipped))

    row_mean = np.mean(M, axis=1, keepdims=True)
    row_std = np.std(M, axis=1, keepdims=True)
    row_min = np.min(M, axis=1, keepdims=True)
    row_max = np.max(M, axis=1, keepdims=True)
    row_spread = row_max - row_min
    feats.extend([row_mean, row_std, row_min, row_max, row_spread])

    # Pairwise differences up to first 5 columns
    n_cols = min(M.shape[1], 5)
    diffs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            diffs.append((M[:, i] - M[:, j])[:, None])
    if diffs:
        feats.extend(diffs)

    return np.hstack(feats)


def fit_stacking_meta_learner(
    oof_list: Sequence[np.ndarray],
    y_true: np.ndarray,
    folds: np.ndarray,
    test_list: Optional[Sequence[np.ndarray]] = None,
    cfg: Optional[CFG] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = CFG()
    X_meta = build_meta_features(oof_list, task=cfg.task, use_nonlinear=cfg.use_nonlinear_meta_features)
    X_test_meta = (
        build_meta_features(test_list, task=cfg.task, use_nonlinear=cfg.use_nonlinear_meta_features)
        if test_list is not None and len(test_list) > 0
        else None
    )

    y = np.asarray(y_true)
    unique_folds = sorted([int(f) for f in np.unique(folds) if f >= 0])
    n_splits = len(unique_folds)

    oof_meta = np.zeros(len(y), dtype=np.float64)
    test_meta = np.zeros(len(X_test_meta), dtype=np.float64) if X_test_meta is not None else None

    task = cfg.task.lower()
    fam = cfg.meta_model.lower()

    for fold in unique_folds:
        tr_m = folds != fold
        va_m = folds == fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_meta[tr_m])
        X_va = scaler.transform(X_meta[va_m])
        X_te = scaler.transform(X_test_meta) if X_test_meta is not None else None
        y_tr = y[tr_m]

        if task == "binary":
            if fam == "logreg":
                model = LogisticRegression(C=cfg.reg_c, max_iter=500, random_state=cfg.seed + fold)
                model.fit(X_tr, y_tr)
                oof_meta[va_m] = model.predict_proba(X_va)[:, 1]
                if X_te is not None and test_meta is not None:
                    test_meta += model.predict_proba(X_te)[:, 1] / n_splits
            elif fam == "ridge":
                model = RidgeClassifier(alpha=1.0 / max(cfg.reg_c, 1e-4), random_state=cfg.seed + fold)
                model.fit(X_tr, y_tr)
                oof_meta[va_m] = expit(model.decision_function(X_va))
                if X_te is not None and test_meta is not None:
                    test_meta += expit(model.decision_function(X_te)) / n_splits
            elif fam == "mlp":
                model = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    alpha=0.1,
                    max_iter=300,
                    early_stopping=True,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_tr, y_tr)
                oof_meta[va_m] = model.predict_proba(X_va)[:, 1]
                if X_te is not None and test_meta is not None:
                    test_meta += model.predict_proba(X_te)[:, 1] / n_splits
            else:
                raise ValueError(f"Unknown meta_model: {cfg.meta_model}")
        else:
            if fam in {"ridge", "logreg"}:
                model = Ridge(alpha=1.0 / max(cfg.reg_c, 1e-4), random_state=cfg.seed + fold)
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=0.1,
                    max_iter=300,
                    early_stopping=True,
                    random_state=cfg.seed + fold,
                )
            model.fit(X_tr, y_tr)
            oof_meta[va_m] = model.predict(X_va)
            if X_te is not None and test_meta is not None:
                test_meta += model.predict(X_te) / n_splits

    if task == "binary":
        metrics = {
            "roc_auc": float(roc_auc_score(y, oof_meta)),
            "logloss": float(log_loss(y, np.clip(oof_meta, 1e-7, 1.0 - 1e-7))),
        }
    else:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, oof_meta))),
        }

    print("=" * 72)
    print(f"[2ND-STAGE STACKING] meta_model={cfg.meta_model} | task={task} | meta_features={X_meta.shape[1]}")
    print(f"OOF Metrics: {metrics}")
    print("=" * 72)

    return {
        "oof_preds": oof_meta,
        "test_preds": test_meta,
        "metrics": metrics,
        "n_meta_features": X_meta.shape[1],
    }


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 400, 100
    folds = np.arange(n_tr) % 5
    y = rng.choice([0, 1], size=n_tr, p=[0.65, 0.35])
    sig = y.astype(float)

    oofs = [
        expit((sig - 0.5) * 2.2 + rng.normal(0, 0.8, size=n_tr)),
        expit((sig - 0.5) * 2.0 + rng.normal(0, 0.9, size=n_tr)),
        expit((sig - 0.5) * 1.8 + rng.normal(0, 0.85, size=n_tr)),
    ]
    tests = [rng.uniform(0.1, 0.9, size=n_te) for _ in range(3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        for meta in ["logreg", "ridge", "mlp"]:
            cfg = CFG(
                exp_name=f"stack_{meta}",
                output_dir=tmpdir,
                task="binary",
                meta_model=meta,
                use_nonlinear_meta_features=True,
            )
            res = fit_stacking_meta_learner(oofs, y, folds, tests, cfg)
            assert res["oof_preds"].shape == (n_tr,)
            assert res["test_preds"].shape == (n_te,)
            assert res["metrics"]["roc_auc"] > 0.75

            oof_p = os.path.join(tmpdir, f"oof_{cfg.exp_name}.npy")
            te_p = os.path.join(tmpdir, f"test_{cfg.exp_name}.npy")
            sub_p = os.path.join(tmpdir, f"sub_{cfg.exp_name}.csv")
            np.save(oof_p, res["oof_preds"])
            np.save(te_p, res["test_preds"])
            pd.DataFrame({"id": np.arange(n_te), "target": res["test_preds"]}).to_csv(sub_p, index=False)
            assert os.path.exists(oof_p) and os.path.exists(te_p) and os.path.exists(sub_p)
        print("[SMOKE TEST PASSED] stacking_meta_learner.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2nd-stage leak-free OOF Stacking Meta-Learner.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--fold-col", type=str, default=CFG.fold_col)
    parser.add_argument("--oof-paths", type=str, nargs="*", default=[])
    parser.add_argument("--test-paths", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--task", type=str, default=CFG.task, choices=["binary", "regression"])
    parser.add_argument("--meta-model", type=str, default=CFG.meta_model, choices=["logreg", "ridge", "mlp"])
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
            fold_col=args.fold_col,
            oof_paths=args.oof_paths,
            test_paths=args.test_paths,
            output_dir=args.output_dir,
            task=args.task,
            meta_model=args.meta_model,
        )
        p = Path(cfg.train_path)
        tr_df = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        y = tr_df[cfg.target_col].to_numpy()
        folds = tr_df[cfg.fold_col].to_numpy(dtype=int)
        oofs = [np.load(fp) for fp in cfg.oof_paths]
        tests = [np.load(fp) for fp in cfg.test_paths] if cfg.test_paths else None
        res = fit_stacking_meta_learner(oofs, y, folds, tests, cfg)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", res["oof_preds"])
        if res["test_preds"] is not None:
            np.save(out_dir / f"test_{cfg.exp_name}.npy", res["test_preds"])
            pd.DataFrame({cfg.id_col: np.arange(len(res["test_preds"])), cfg.target_col: res["test_preds"]}).to_csv(
                out_dir / f"sub_{cfg.exp_name}.csv", index=False
            )
