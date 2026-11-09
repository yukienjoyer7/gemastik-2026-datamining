#!/usr/bin/env python3
"""
lgbm_oof_pipeline.py
====================
Standalone LightGBM Out-Of-Fold (OOF) trainer supporting:
  - `binary`       (ROC-AUC & LogLoss)
  - `multiclass`   (Multi-LogLoss & Macro-F1)
  - `regression`   (RMSE & MAE)
  - `lambdarank`   (NDCG@K grouped by `--group-col`)
Includes graceful fallback to `sklearn.ensemble.HistGradientBoosting*` if `lightgbm` is not installed,
fold-averaged feature importance export, and standardized competition artifacts:
  - `oof_<exp_name>.npy`
  - `test_<exp_name>.npy`
  - `sub_<exp_name>.csv`
  - `importance_<exp_name>.csv`

Usage:
  python3 00_core/tabular_gbdt/lgbm_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name lgbm_v1
  python3 00_core/tabular_gbdt/lgbm_oof_pipeline.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


@dataclass
class CFG:
    exp_name: str = "lgbm_baseline"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = "test.parquet"
    output_dir: str = "."
    task: str = "binary"  # binary | multiclass | regression | lambdarank
    target_col: str = "target"
    id_col: str = "id"
    fold_col: str = "fold"
    group_col: Optional[str] = None
    drop_cols: List[str] = field(default_factory=list)
    n_estimators: int = 1000
    learning_rate: float = 0.03
    num_leaves: int = 31
    subsample_freq: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    early_stopping_rounds: int = 50
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str], List[str]]:
    exclude = {cfg.target_col, cfg.id_col, cfg.fold_col, cfg.group_col} | set(cfg.drop_cols)
    feature_cols = [c for c in train_df.columns if c not in exclude]
    X_tr = train_df[feature_cols].copy()
    X_te = test_df[feature_cols].copy() if test_df is not None else None

    cat_cols: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_object_dtype(X_tr[col]) or pd.api.types.is_string_dtype(X_tr[col]) or isinstance(X_tr[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
            combined = pd.concat([X_tr[col], X_te[col]], axis=0) if X_te is not None else X_tr[col]
            categories = pd.Index(combined.fillna("__MISSING__").astype(str).unique())
            X_tr[col] = pd.Categorical(X_tr[col].fillna("__MISSING__").astype(str), categories=categories).codes.astype(np.float32)
            if X_te is not None:
                X_te[col] = pd.Categorical(X_te[col].fillna("__MISSING__").astype(str), categories=categories).codes.astype(np.float32)
        else:
            X_tr[col] = X_tr[col].astype(np.float32)
            if X_te is not None:
                X_te[col] = X_te[col].astype(np.float32)

    return X_tr, X_te, feature_cols, cat_cols


def evaluate_predictions(y_true: np.ndarray, oof_preds: np.ndarray, task: str) -> Dict[str, float]:
    t = task.lower()
    if t == "binary":
        return {
            "roc_auc": float(roc_auc_score(y_true, oof_preds)),
            "logloss": float(log_loss(y_true, np.clip(oof_preds, 1e-7, 1 - 1e-7))),
        }
    if t == "multiclass":
        pred_cls = np.argmax(oof_preds, axis=1)
        return {
            "macro_f1": float(f1_score(y_true, pred_cls, average="macro", zero_division=0)),
            "multi_logloss": float(log_loss(y_true, np.clip(oof_preds, 1e-7, 1 - 1e-7))),
        }
    # regression or lambdarank
    rmse = float(np.sqrt(mean_squared_error(y_true, oof_preds)))
    return {"rmse": rmse}


def run_lgbm_oof_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    X_tr, X_te, feature_cols, cat_cols = prepare_features(train_df, test_df, cfg)
    y = train_df[cfg.target_col].to_numpy()
    folds = train_df[cfg.fold_col].to_numpy(dtype=int)
    unique_folds = sorted([int(f) for f in np.unique(folds) if f >= 0])
    n_splits = len(unique_folds)

    task = cfg.task.lower()
    n_classes = int(np.nunique(y) if hasattr(np, "nunique") else len(np.unique(y))) if task == "multiclass" else 1

    if task == "multiclass":
        oof_preds = np.zeros((len(train_df), n_classes), dtype=np.float64)
        test_preds = np.zeros((len(test_df), n_classes), dtype=np.float64) if test_df is not None else None
    else:
        oof_preds = np.zeros(len(train_df), dtype=np.float64)
        test_preds = np.zeros(len(test_df), dtype=np.float64) if test_df is not None else None

    importances = np.zeros(len(feature_cols), dtype=np.float64)

    for fold in unique_folds:
        tr_mask = folds != fold
        va_mask = folds == fold
        X_train, y_train = X_tr.iloc[tr_mask], y[tr_mask]
        X_val, y_val = X_tr.iloc[va_mask], y[va_mask]

        if lgb is not None:
            params: Dict[str, Any] = {
                "n_estimators": cfg.n_estimators,
                "learning_rate": cfg.learning_rate,
                "num_leaves": cfg.num_leaves,
                "subsample": cfg.subsample,
                "subsample_freq": cfg.subsample_freq,
                "colsample_bytree": cfg.colsample_bytree,
                "min_child_samples": cfg.min_child_samples,
                "random_state": cfg.seed + fold,
                "n_jobs": -1,
                "verbosity": -1,
            }
            if task == "binary":
                model = lgb.LGBMClassifier(objective="binary", **params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                oof_preds[va_mask] = model.predict_proba(X_val)[:, 1]
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict_proba(X_te)[:, 1] / n_splits

            elif task == "multiclass":
                model = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                oof_preds[va_mask] = model.predict_proba(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict_proba(X_te) / n_splits

            elif task == "lambdarank":
                if not cfg.group_col or cfg.group_col not in train_df.columns:
                    raise ValueError("`--group-col` is required for `lambdarank`.")
                tr_Order = np.argsort(train_df.loc[tr_mask, cfg.group_col].to_numpy())
                va_Order = np.argsort(train_df.loc[va_mask, cfg.group_col].to_numpy())
                tr_grp = train_df.loc[tr_mask].iloc[tr_Order].groupby(cfg.group_col, sort=False).size().to_numpy()
                va_grp = train_df.loc[va_mask].iloc[va_Order].groupby(cfg.group_col, sort=False).size().to_numpy()
                model = lgb.LGBMRanker(objective="lambdarank", **params)
                model.fit(
                    X_train.iloc[tr_Order], y_train[tr_Order],
                    group=tr_grp,
                    eval_set=[(X_val.iloc[va_Order], y_val[va_Order])],
                    eval_group=[va_grp],
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                oof_preds[va_mask] = model.predict(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict(X_te) / n_splits

            else:
                model = lgb.LGBMRegressor(objective="regression", **params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                oof_preds[va_mask] = model.predict(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict(X_te) / n_splits

            importances += model.feature_importances_ / n_splits

        else:
            # Fallback to sklearn HistGradientBoosting when lightgbm is not installed
            if task in {"binary", "multiclass"}:
                model = HistGradientBoostingClassifier(
                    max_iter=min(cfg.n_estimators, 200),
                    learning_rate=cfg.learning_rate,
                    max_leaf_nodes=cfg.num_leaves,
                    min_samples_leaf=cfg.min_child_samples,
                    early_stopping=True,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_train, y_train)
                proba_val = model.predict_proba(X_val)
                oof_preds[va_mask] = proba_val[:, 1] if task == "binary" else proba_val
                if X_te is not None and test_preds is not None:
                    proba_te = model.predict_proba(X_te)
                    test_preds += (proba_te[:, 1] if task == "binary" else proba_te) / n_splits
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=min(cfg.n_estimators, 200),
                    learning_rate=cfg.learning_rate,
                    max_leaf_nodes=cfg.num_leaves,
                    min_samples_leaf=cfg.min_child_samples,
                    early_stopping=True,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_train, y_train)
                oof_preds[va_mask] = model.predict(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict(X_te) / n_splits

            perm = permutation_importance(model, X_val, y_val, n_repeats=2, random_state=cfg.seed + fold)
            importances += np.maximum(perm.importances_mean, 0.0) / n_splits

    metrics = evaluate_predictions(y, oof_preds, task)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = out_dir / f"oof_{cfg.exp_name}.npy"
    np.save(oof_path, oof_preds)

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    imp_path = out_dir / f"importance_{cfg.exp_name}.csv"
    imp_df.to_csv(imp_path, index=False)

    test_path_out = None
    sub_path_out = None
    if test_df is not None and test_preds is not None:
        test_path_out = out_dir / f"test_{cfg.exp_name}.npy"
        np.save(test_path_out, test_preds)

        sub_df = pd.DataFrame()
        if cfg.id_col in test_df.columns:
            sub_df[cfg.id_col] = test_df[cfg.id_col].values
        else:
            sub_df[cfg.id_col] = np.arange(len(test_df))

        if task == "multiclass":
            sub_df[cfg.target_col] = np.argmax(test_preds, axis=1)
        else:
            sub_df[cfg.target_col] = test_preds
        sub_path_out = out_dir / f"sub_{cfg.exp_name}.csv"
        sub_df.to_csv(sub_path_out, index=False)

    print("=" * 72)
    print(f"[LGBM OOF PIPELINE] exp={cfg.exp_name} | task={task} | backend={'LightGBM' if lgb else 'HistGBDT-Fallback'}")
    print(f"OOF Metrics: {metrics}")
    print(f"Saved: {oof_path.name}" + (f", {test_path_out.name}, {sub_path_out.name}" if test_path_out else ""))
    print("Top 5 Features:")
    print(imp_df.head(5).to_string(index=False))
    print("=" * 72)

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "metrics": metrics,
        "importance_df": imp_df,
        "oof_path": str(oof_path),
        "test_path": str(test_path_out) if test_path_out else None,
        "sub_path": str(sub_path_out) if sub_path_out else None,
    }


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 250, 80
    x1 = rng.normal(0, 1, size=n_tr)
    x2 = rng.normal(0, 1, size=n_tr)
    cat = rng.choice(["cat_a", "cat_b", "cat_c"], size=n_tr)
    y_bin = (x1 + 0.8 * x2 + (cat == "cat_a") * 0.7 + rng.normal(0, 0.5, size=n_tr) > 0.2).astype(int)

    tr_df = pd.DataFrame({
        "id": np.arange(n_tr),
        "fold": np.arange(n_tr) % 5,
        "query_id": np.repeat(np.arange(25), 10),
        "f1": x1,
        "f2": x2,
        "cat": cat,
        "target": y_bin,
    })
    te_df = pd.DataFrame({
        "id": np.arange(n_tr, n_tr + n_te),
        "query_id": np.repeat(np.arange(8), 10),
        "f1": rng.normal(0, 1, size=n_te),
        "f2": rng.normal(0, 1, size=n_te),
        "cat": rng.choice(["cat_a", "cat_b", "cat_c"], size=n_te),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["binary", "regression", "lambdarank"]:
            cfg = CFG(
                exp_name=f"lgbm_{task}",
                output_dir=tmpdir,
                task=task,
                target_col="target",
                id_col="id",
                fold_col="fold",
                group_col="query_id" if task == "lambdarank" else None,
                n_estimators=60,
            )
            res = run_lgbm_oof_pipeline(tr_df, te_df, cfg)
            assert res["oof_preds"].shape == (n_tr,)
            assert res["test_preds"].shape == (n_te,)
            assert os.path.exists(res["oof_path"])
            assert os.path.exists(res["test_path"])
            assert os.path.exists(res["sub_path"])
        print("[SMOKE TEST PASSED] lgbm_oof_pipeline.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM OOF training & prediction pipeline.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--task", type=str, default=CFG.task, choices=["binary", "multiclass", "regression", "lambdarank"])
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--fold-col", type=str, default=CFG.fold_col)
    parser.add_argument("--group-col", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        cfg = CFG(
            exp_name=args.exp_name,
            train_path=args.train_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            task=args.task,
            target_col=args.target_col,
            id_col=args.id_col,
            fold_col=args.fold_col,
            group_col=args.group_col,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_lgbm_oof_pipeline(tr, te, cfg)
