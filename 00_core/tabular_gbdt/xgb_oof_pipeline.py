#!/usr/bin/env python3
"""
xgb_oof_pipeline.py
===================
Standalone XGBoost (`tree_method="hist"`, `enable_categorical=True`) Out-Of-Fold (OOF) trainer
supporting `binary`, `multiclass`, and `regression`.
Includes graceful fallback to `sklearn.ensemble.HistGradientBoosting*` if `xgboost` is not installed.
Outputs standardized competition artifacts:
  - `oof_<exp_name>.npy`
  - `test_<exp_name>.npy`
  - `sub_<exp_name>.csv`

Usage:
  python3 00_core/tabular_gbdt/xgb_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name xgb_v1
  python3 00_core/tabular_gbdt/xgb_oof_pipeline.py --smoke-test
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
from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    xgb = None


@dataclass
class CFG:
    exp_name: str = "xgb_baseline"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = "test.parquet"
    output_dir: str = "."
    task: str = "binary"  # binary | multiclass | regression
    target_col: str = "target"
    id_col: str = "id"
    fold_col: str = "fold"
    drop_cols: List[str] = field(default_factory=list)
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def prepare_xgb_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    exclude = {cfg.target_col, cfg.id_col, cfg.fold_col} | set(cfg.drop_cols)
    feature_cols = [c for c in train_df.columns if c not in exclude]
    X_tr = train_df[feature_cols].copy()
    X_te = test_df[feature_cols].copy() if test_df is not None else None

    for col in feature_cols:
        if pd.api.types.is_object_dtype(X_tr[col]) or pd.api.types.is_string_dtype(X_tr[col]) or isinstance(X_tr[col].dtype, pd.CategoricalDtype):
            combined = pd.concat([X_tr[col], X_te[col]], axis=0) if X_te is not None else X_tr[col]
            categories = pd.Index(combined.fillna("__MISSING__").astype(str).unique())
            if xgb is not None:
                X_tr[col] = pd.Categorical(X_tr[col].fillna("__MISSING__").astype(str), categories=categories)
                if X_te is not None:
                    X_te[col] = pd.Categorical(X_te[col].fillna("__MISSING__").astype(str), categories=categories)
            else:
                X_tr[col] = pd.Categorical(X_tr[col].fillna("__MISSING__").astype(str), categories=categories).codes.astype(np.float32)
                if X_te is not None:
                    X_te[col] = pd.Categorical(X_te[col].fillna("__MISSING__").astype(str), categories=categories).codes.astype(np.float32)
        else:
            X_tr[col] = X_tr[col].astype(np.float32)
            if X_te is not None:
                X_te[col] = X_te[col].astype(np.float32)
    return X_tr, X_te, feature_cols


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
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, oof_preds)))}


def run_xgb_oof_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    X_tr, X_te, feature_cols = prepare_xgb_features(train_df, test_df, cfg)
    y = train_df[cfg.target_col].to_numpy()
    folds = train_df[cfg.fold_col].to_numpy(dtype=int)
    unique_folds = sorted([int(f) for f in np.unique(folds) if f >= 0])
    n_splits = len(unique_folds)
    task = cfg.task.lower()
    n_classes = len(np.unique(y)) if task == "multiclass" else 1

    if task == "multiclass":
        oof_preds = np.zeros((len(train_df), n_classes), dtype=np.float64)
        test_preds = np.zeros((len(test_df), n_classes), dtype=np.float64) if test_df is not None else None
    else:
        oof_preds = np.zeros(len(train_df), dtype=np.float64)
        test_preds = np.zeros(len(test_df), dtype=np.float64) if test_df is not None else None

    for fold in unique_folds:
        tr_mask = folds != fold
        va_mask = folds == fold
        X_train, y_train = X_tr.iloc[tr_mask], y[tr_mask]
        X_val, y_val = X_tr.iloc[va_mask], y[va_mask]

        if xgb is not None:
            params: Dict[str, Any] = {
                "n_estimators": cfg.n_estimators,
                "learning_rate": cfg.learning_rate,
                "max_depth": cfg.max_depth,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "reg_alpha": cfg.reg_alpha,
                "reg_lambda": cfg.reg_lambda,
                "tree_method": "hist",
                "enable_categorical": True,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                "random_state": cfg.seed + fold,
                "n_jobs": -1,
            }
            if task == "binary":
                model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                oof_preds[va_mask] = model.predict_proba(X_val)[:, 1]
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict_proba(X_te)[:, 1] / n_splits
            elif task == "multiclass":
                model = xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                oof_preds[va_mask] = model.predict_proba(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict_proba(X_te) / n_splits
            else:
                model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                oof_preds[va_mask] = model.predict(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict(X_te) / n_splits
        else:
            if task in {"binary", "multiclass"}:
                model = HistGradientBoostingClassifier(
                    max_iter=min(cfg.n_estimators, 200),
                    learning_rate=cfg.learning_rate,
                    max_depth=cfg.max_depth,
                    l2_regularization=cfg.reg_lambda,
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
                    max_depth=cfg.max_depth,
                    l2_regularization=cfg.reg_lambda,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_train, y_train)
                oof_preds[va_mask] = model.predict(X_val)
                if X_te is not None and test_preds is not None:
                    test_preds += model.predict(X_te) / n_splits

    metrics = evaluate_predictions(y, oof_preds, task)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / f"oof_{cfg.exp_name}.npy"
    np.save(oof_path, oof_preds)

    test_path_out = None
    sub_path_out = None
    if test_df is not None and test_preds is not None:
        test_path_out = out_dir / f"test_{cfg.exp_name}.npy"
        np.save(test_path_out, test_preds)

        sub_df = pd.DataFrame()
        sub_df[cfg.id_col] = test_df[cfg.id_col].values if cfg.id_col in test_df.columns else np.arange(len(test_df))
        sub_df[cfg.target_col] = np.argmax(test_preds, axis=1) if task == "multiclass" else test_preds
        sub_path_out = out_dir / f"sub_{cfg.exp_name}.csv"
        sub_df.to_csv(sub_path_out, index=False)

    print("=" * 72)
    print(f"[XGBOOST OOF PIPELINE] exp={cfg.exp_name} | task={task} | backend={'XGBoost(hist)' if xgb else 'HistGBDT-Fallback'}")
    print(f"OOF Metrics: {metrics}")
    print("=" * 72)

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "metrics": metrics,
        "oof_path": str(oof_path),
        "test_path": str(test_path_out) if test_path_out else None,
        "sub_path": str(sub_path_out) if sub_path_out else None,
    }


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 200, 60
    x1 = rng.normal(0, 1, size=n_tr)
    x2 = rng.normal(0, 1, size=n_tr)
    cat = rng.choice(["tier_1", "tier_2", "tier_3"], size=n_tr)
    y = (x1 - 0.6 * x2 + (cat == "tier_1") * 0.9 + rng.normal(0, 0.5, size=n_tr) > 0.1).astype(int)

    tr_df = pd.DataFrame({"id": np.arange(n_tr), "fold": np.arange(n_tr) % 5, "x1": x1, "x2": x2, "tier": cat, "target": y})
    te_df = pd.DataFrame({
        "id": np.arange(n_tr, n_tr + n_te),
        "x1": rng.normal(0, 1, size=n_te),
        "x2": rng.normal(0, 1, size=n_te),
        "tier": rng.choice(["tier_1", "tier_2", "tier_3"], size=n_te),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = CFG(exp_name="xgb_smoke", output_dir=tmpdir, task="binary", n_estimators=60)
        res = run_xgb_oof_pipeline(tr_df, te_df, cfg)
        assert res["oof_preds"].shape == (n_tr,)
        assert res["test_preds"].shape == (n_te,)
        assert os.path.exists(res["oof_path"])
        assert os.path.exists(res["test_path"])
        assert os.path.exists(res["sub_path"])
        print("[SMOKE TEST PASSED] xgb_oof_pipeline.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost hist OOF trainer.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--task", type=str, default=CFG.task, choices=["binary", "multiclass", "regression"])
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--fold-col", type=str, default=CFG.fold_col)
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
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_xgb_oof_pipeline(tr, te, cfg)
