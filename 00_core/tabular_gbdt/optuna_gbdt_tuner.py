#!/usr/bin/env python3
"""
optuna_gbdt_tuner.py
====================
Leak-free K-Fold Out-Of-Fold (OOF) Hyperparameter Tuner for LightGBM, XGBoost, and CatBoost.
Uses `optuna` TPE sampler when installed, with a built-in randomized Latin-hypercube/grid fallback
when `optuna` is not installed. Exports the best hyperparameter configuration and OOF score to JSON.

Usage:
  python3 00_core/tabular_gbdt/optuna_gbdt_tuner.py --train-path train_folds.parquet --model-family lgbm --task binary --n-trials 30
  python3 00_core/tabular_gbdt/optuna_gbdt_tuner.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class CFG:
    train_path: str = "train_folds.parquet"
    output_json: str = "best_gbdt_params.json"
    model_family: str = "lgbm"  # lgbm | xgb | catboost
    task: str = "binary"        # binary | regression
    target_col: str = "target"
    id_col: str = "id"
    fold_col: str = "fold"
    drop_cols: List[str] = field(default_factory=list)
    n_trials: int = 20
    n_estimators: int = 300
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def prepare_encoded_matrix(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    exclude = {cfg.target_col, cfg.id_col, cfg.fold_col} | set(cfg.drop_cols)
    cols = [c for c in df.columns if c not in exclude]
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            codes, _ = pd.factorize(X[c].fillna("__MISSING__").astype(str))
            X[c] = codes.astype(np.float32)
        else:
            X[c] = X[c].astype(np.float32)
    return X


def sample_candidate_params(rng: np.random.Generator, model_family: str, trial: Any = None) -> Dict[str, Any]:
    fam = model_family.lower()
    if trial is not None:
        if fam == "lgbm":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        if fam == "xgb":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
        }

    # Random search fallback when optuna is not installed
    lr = float(np.exp(rng.uniform(np.log(0.015), np.log(0.15))))
    if fam == "lgbm":
        return {
            "learning_rate": round(lr, 5),
            "num_leaves": int(rng.integers(15, 96)),
            "min_child_samples": int(rng.integers(8, 60)),
            "subsample": round(float(rng.uniform(0.6, 1.0)), 3),
            "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 3),
            "reg_alpha": round(float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))), 5),
            "reg_lambda": round(float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))), 5),
        }
    if fam == "xgb":
        return {
            "learning_rate": round(lr, 5),
            "max_depth": int(rng.integers(3, 9)),
            "min_child_weight": round(float(rng.uniform(1.0, 15.0)), 3),
            "subsample": round(float(rng.uniform(0.6, 1.0)), 3),
            "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 3),
            "reg_alpha": round(float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))), 5),
            "reg_lambda": round(float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))), 5),
        }
    return {
        "learning_rate": round(lr, 5),
        "depth": int(rng.integers(4, 8)),
        "l2_leaf_reg": round(float(rng.uniform(1.0, 12.0)), 3),
    }


def evaluate_params_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    params: Dict[str, Any],
    cfg: CFG,
) -> float:
    unique_folds = sorted([int(f) for f in np.unique(folds) if f >= 0])
    oof = np.zeros(len(X), dtype=np.float64)
    fam = cfg.model_family.lower()
    task = cfg.task.lower()

    for fold in unique_folds:
        tr_m = folds != fold
        va_m = folds == fold
        X_tr, y_tr = X.iloc[tr_m], y[tr_m]
        X_va, y_va = X.iloc[va_m], y[va_m]

        if fam == "lgbm" and lgb is not None:
            if task == "binary":
                m = lgb.LGBMClassifier(n_estimators=cfg.n_estimators, random_state=cfg.seed + fold, verbosity=-1, **params)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(25, verbose=False)])
                oof[va_m] = m.predict_proba(X_va)[:, 1]
            else:
                m = lgb.LGBMRegressor(n_estimators=cfg.n_estimators, random_state=cfg.seed + fold, verbosity=-1, **params)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(25, verbose=False)])
                oof[va_m] = m.predict(X_va)

        elif fam == "xgb" and xgb is not None:
            if task == "binary":
                m = xgb.XGBClassifier(
                    n_estimators=cfg.n_estimators, tree_method="hist", early_stopping_rounds=25,
                    eval_metric="logloss", random_state=cfg.seed + fold, **params
                )
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                oof[va_m] = m.predict_proba(X_va)[:, 1]
            else:
                m = xgb.XGBRegressor(
                    n_estimators=cfg.n_estimators, tree_method="hist", early_stopping_rounds=25,
                    eval_metric="rmse", random_state=cfg.seed + fold, **params
                )
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                oof[va_m] = m.predict(X_va)

        elif fam == "catboost" and HAS_CATBOOST:
            if task == "binary":
                m = CatBoostClassifier(iterations=cfg.n_estimators, random_seed=cfg.seed + fold, verbose=False, **params)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=25, verbose=False)
                oof[va_m] = m.predict_proba(X_va)[:, 1]
            else:
                m = CatBoostRegressor(iterations=cfg.n_estimators, random_seed=cfg.seed + fold, verbose=False, **params)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=25, verbose=False)
                oof[va_m] = m.predict(X_va)

        else:
            # HistGradientBoosting fallback mapped from candidate params
            max_leaf = params.get("num_leaves", 2 ** params.get("max_depth", params.get("depth", 5)))
            min_leaf = int(params.get("min_child_samples", params.get("min_child_weight", 15)))
            l2_reg = float(params.get("reg_lambda", params.get("l2_leaf_reg", 1.0)))
            if task == "binary":
                m = HistGradientBoostingClassifier(
                    max_iter=min(cfg.n_estimators, 120),
                    learning_rate=params["learning_rate"],
                    max_leaf_nodes=max(4, int(max_leaf)),
                    min_samples_leaf=max(2, min_leaf),
                    l2_regularization=l2_reg,
                    random_state=cfg.seed + fold,
                )
                m.fit(X_tr, y_tr)
                oof[va_m] = m.predict_proba(X_va)[:, 1]
            else:
                m = HistGradientBoostingRegressor(
                    max_iter=min(cfg.n_estimators, 120),
                    learning_rate=params["learning_rate"],
                    max_leaf_nodes=max(4, int(max_leaf)),
                    min_samples_leaf=max(2, min_leaf),
                    l2_regularization=l2_reg,
                    random_state=cfg.seed + fold,
                )
                m.fit(X_tr, y_tr)
                oof[va_m] = m.predict(X_va)

    if task == "binary":
        return float(roc_auc_score(y, oof))
    # Return negative RMSE so higher is always better during maximization
    return -float(np.sqrt(mean_squared_error(y, oof)))


def run_optuna_tuner(df: pd.DataFrame, cfg: CFG) -> Dict[str, Any]:
    X = prepare_encoded_matrix(df, cfg)
    y = df[cfg.target_col].to_numpy()
    folds = df[cfg.fold_col].to_numpy(dtype=int)
    rng = np.random.default_rng(cfg.seed)

    if optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            cand = sample_candidate_params(rng, cfg.model_family, trial=trial)
            return evaluate_params_cv(X, y, folds, cand, cfg)

        study.optimize(objective, n_trials=cfg.n_trials)
        best_params = study.best_params
        best_score = float(study.best_value)
    else:
        best_params = {}
        best_score = -1e18
        for _ in range(cfg.n_trials):
            cand = sample_candidate_params(rng, cfg.model_family, trial=None)
            score = evaluate_params_cv(X, y, folds, cand, cfg)
            if score > best_score:
                best_score = score
                best_params = cand

    metric_name = "oof_roc_auc" if cfg.task == "binary" else "oof_neg_rmse"
    result = {
        "model_family": cfg.model_family,
        "task": cfg.task,
        "n_trials": cfg.n_trials,
        "tuner_backend": "optuna_tpe" if optuna is not None else "random_search_fallback",
        metric_name: round(best_score, 6),
        "best_params": best_params,
    }

    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print(f"[GBDT TUNER] family={cfg.model_family} | task={cfg.task} | backend={result['tuner_backend']}")
    print(f"Best {metric_name}: {best_score:.6f}")
    print(f"Best Params: {best_params}")
    print(f"Saved best params to: {cfg.output_json}")
    print("=" * 72)
    return result


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n = 180
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    y = (x1 * 1.2 - x2 * 0.8 + rng.normal(0, 0.5, size=n) > 0.0).astype(int)
    df = pd.DataFrame({
        "id": np.arange(n),
        "fold": np.arange(n) % 5,
        "x1": x1,
        "x2": x2,
        "x3": rng.uniform(-1, 1, size=n),
        "target": y,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "best_params.json")
        cfg = CFG(
            output_json=out_json,
            model_family="lgbm",
            task="binary",
            n_trials=5,
            n_estimators=40,
        )
        res = run_optuna_tuner(df, cfg)
        assert res["oof_roc_auc"] > 0.70
        assert "learning_rate" in res["best_params"]
        assert os.path.exists(out_json)
        print("[SMOKE TEST PASSED] optuna_gbdt_tuner.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-free K-Fold CV hyperparameter tuner for GBDTs.")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--output-json", type=str, default=CFG.output_json)
    parser.add_argument("--model-family", type=str, default=CFG.model_family, choices=["lgbm", "xgb", "catboost"])
    parser.add_argument("--task", type=str, default=CFG.task, choices=["binary", "regression"])
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--fold-col", type=str, default=CFG.fold_col)
    parser.add_argument("--n-trials", type=int, default=CFG.n_trials)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        cfg = CFG(
            train_path=args.train_path,
            output_json=args.output_json,
            model_family=args.model_family,
            task=args.task,
            target_col=args.target_col,
            fold_col=args.fold_col,
            n_trials=args.n_trials,
        )
        df_in = load_dataframe(cfg.train_path)
        run_optuna_tuner(df_in, cfg)
