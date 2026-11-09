#!/usr/bin/env python3
"""
catboost_oof_pipeline.py
========================
Standalone CatBoost Out-Of-Fold (OOF) trainer supporting native `cat_features` and `text_features`
(for mixed tabular + text datasets), covering `binary`, `multiclass`, and `regression`.
Includes graceful fallback (TF-IDF + SVD on `text_features` + HistGradientBoosting) if `catboost`
is not installed in the current environment.
Outputs standardized competition artifacts:
  - `oof_<exp_name>.npy`
  - `test_<exp_name>.npy`
  - `sub_<exp_name>.csv`

Usage:
  python3 00_core/tabular_gbdt/catboost_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name cb_v1
  python3 00_core/tabular_gbdt/catboost_oof_pipeline.py --smoke-test
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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class CFG:
    exp_name: str = "catboost_baseline"
    train_path: str = "train_folds.parquet"
    test_path: Optional[str] = "test.parquet"
    output_dir: str = "."
    task: str = "binary"  # binary | multiclass | regression
    target_col: str = "target"
    id_col: str = "id"
    fold_col: str = "fold"
    cat_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)
    iterations: int = 1000
    learning_rate: float = 0.04
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 50
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


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


def run_catboost_oof_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    exclude = {cfg.target_col, cfg.id_col, cfg.fold_col} | set(cfg.drop_cols)
    feature_cols = [c for c in train_df.columns if c not in exclude]

    text_cols = [c for c in cfg.text_cols if c in feature_cols]
    cat_cols = [c for c in cfg.cat_cols if c in feature_cols and c not in text_cols]
    for col in feature_cols:
        if col not in text_cols and col not in cat_cols:
            if pd.api.types.is_object_dtype(train_df[col]) or pd.api.types.is_string_dtype(train_df[col]) or isinstance(train_df[col].dtype, pd.CategoricalDtype):
                cat_cols.append(col)

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

    if HAS_CATBOOST:
        X_tr = train_df[feature_cols].copy()
        X_te = test_df[feature_cols].copy() if test_df is not None else None
        for c in cat_cols + text_cols:
            X_tr[c] = X_tr[c].fillna("__MISSING__").astype(str)
            if X_te is not None:
                X_te[c] = X_te[c].fillna("__MISSING__").astype(str)

        for fold in unique_folds:
            tr_mask = folds != fold
            va_mask = folds == fold
            tr_pool = Pool(
                X_tr.iloc[tr_mask],
                y[tr_mask],
                cat_features=cat_cols,
                text_features=text_cols,
            )
            va_pool = Pool(
                X_tr.iloc[va_mask],
                y[va_mask],
                cat_features=cat_cols,
                text_features=text_cols,
            )
            te_pool = (
                Pool(X_te, cat_features=cat_cols, text_features=text_cols)
                if X_te is not None
                else None
            )

            params = {
                "iterations": cfg.iterations,
                "learning_rate": cfg.learning_rate,
                "depth": cfg.depth,
                "l2_leaf_reg": cfg.l2_leaf_reg,
                "random_seed": cfg.seed + fold,
                "verbose": False,
            }
            if task in {"binary", "multiclass"}:
                loss_fn = "Logloss" if task == "binary" else "MultiClass"
                model = CatBoostClassifier(loss_function=loss_fn, **params)
                model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cfg.early_stopping_rounds, verbose=False)
                proba_val = model.predict_proba(va_pool)
                oof_preds[va_mask] = proba_val[:, 1] if task == "binary" else proba_val
                if te_pool is not None and test_preds is not None:
                    proba_te = model.predict_proba(te_pool)
                    test_preds += (proba_te[:, 1] if task == "binary" else proba_te) / n_splits
            else:
                model = CatBoostRegressor(loss_function="RMSE", **params)
                model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cfg.early_stopping_rounds, verbose=False)
                oof_preds[va_mask] = model.predict(va_pool)
                if te_pool is not None and test_preds is not None:
                    test_preds += model.predict(te_pool) / n_splits
    else:
        # Fallback: encode cat_cols + TF-IDF SVD on text_cols + HistGradientBoosting
        num_cols = [c for c in feature_cols if c not in cat_cols and c not in text_cols]
        parts_tr: List[pd.DataFrame] = [train_df[num_cols].astype(np.float32)] if num_cols else []
        parts_te: List[pd.DataFrame] = [test_df[num_cols].astype(np.float32)] if (test_df is not None and num_cols) else []

        if cat_cols:
            cat_tr = pd.DataFrame(index=train_df.index)
            cat_te = pd.DataFrame(index=test_df.index) if test_df is not None else None
            for c in cat_cols:
                combined = pd.concat([train_df[c], test_df[c]], axis=0) if test_df is not None else train_df[c]
                cats = pd.Index(combined.fillna("__MISSING__").astype(str).unique())
                cat_tr[c] = pd.Categorical(train_df[c].fillna("__MISSING__").astype(str), categories=cats).codes.astype(np.float32)
                if cat_te is not None and test_df is not None:
                    cat_te[c] = pd.Categorical(test_df[c].fillna("__MISSING__").astype(str), categories=cats).codes.astype(np.float32)
            parts_tr.append(cat_tr)
            if cat_te is not None:
                parts_te.append(cat_te)

        for tcol in text_cols:
            vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
            tr_txt = train_df[tcol].fillna("").astype(str).tolist()
            X_tf_tr = vec.fit_transform(tr_txt)
            n_comp = min(6, max(1, X_tf_tr.shape[1] - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed)
            svd_tr = pd.DataFrame(
                svd.fit_transform(X_tf_tr),
                index=train_df.index,
                columns=[f"{tcol}_svd_{i}" for i in range(n_comp)],
            )
            parts_tr.append(svd_tr)
            if test_df is not None:
                te_txt = test_df[tcol].fillna("").astype(str).tolist()
                X_tf_te = vec.transform(te_txt)
                svd_te = pd.DataFrame(
                    svd.transform(X_tf_te),
                    index=test_df.index,
                    columns=[f"{tcol}_svd_{i}" for i in range(n_comp)],
                )
                parts_te.append(svd_te)

        X_tr_enc = pd.concat(parts_tr, axis=1)
        X_te_enc = pd.concat(parts_te, axis=1) if parts_te else None

        for fold in unique_folds:
            tr_mask = folds != fold
            va_mask = folds == fold
            X_train, y_train = X_tr_enc.iloc[tr_mask], y[tr_mask]
            X_val = X_tr_enc.iloc[va_mask]

            if task in {"binary", "multiclass"}:
                model = HistGradientBoostingClassifier(
                    max_iter=min(cfg.iterations, 200),
                    learning_rate=cfg.learning_rate,
                    max_depth=cfg.depth,
                    l2_regularization=cfg.l2_leaf_reg,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_train, y_train)
                proba_val = model.predict_proba(X_val)
                oof_preds[va_mask] = proba_val[:, 1] if task == "binary" else proba_val
                if X_te_enc is not None and test_preds is not None:
                    proba_te = model.predict_proba(X_te_enc)
                    test_preds += (proba_te[:, 1] if task == "binary" else proba_te) / n_splits
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=min(cfg.iterations, 200),
                    learning_rate=cfg.learning_rate,
                    max_depth=cfg.depth,
                    l2_regularization=cfg.l2_leaf_reg,
                    random_state=cfg.seed + fold,
                )
                model.fit(X_train, y_train)
                oof_preds[va_mask] = model.predict(X_val)
                if X_te_enc is not None and test_preds is not None:
                    test_preds += model.predict(X_te_enc) / n_splits

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
    print(f"[CATBOOST OOF PIPELINE] exp={cfg.exp_name} | task={task} | backend={'CatBoost' if HAS_CATBOOST else 'HistGBDT+SVD-Fallback'}")
    print(f"Cat cols: {cat_cols} | Text cols: {text_cols}")
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
    f1 = rng.normal(0, 1, size=n_tr)
    cat = rng.choice(["jakarta", "surabaya", "bandung"], size=n_tr)
    txt = [f"review barang kualitas bagus pengiriman {c} nomor {i % 7}" for i, c in enumerate(cat)]
    y = (f1 + (cat == "jakarta") * 0.8 + rng.normal(0, 0.4, size=n_tr) > 0.2).astype(int)

    tr_df = pd.DataFrame({
        "id": np.arange(n_tr),
        "fold": np.arange(n_tr) % 5,
        "f1": f1,
        "city": cat,
        "review_text": txt,
        "target": y,
    })
    te_df = pd.DataFrame({
        "id": np.arange(n_tr, n_tr + n_te),
        "f1": rng.normal(0, 1, size=n_te),
        "city": rng.choice(["jakarta", "surabaya", "bandung"], size=n_te),
        "review_text": [f"review barang pengiriman cepat {i}" for i in range(n_te)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = CFG(
            exp_name="cb_smoke",
            output_dir=tmpdir,
            task="binary",
            cat_cols=["city"],
            text_cols=["review_text"],
            iterations=60,
        )
        res = run_catboost_oof_pipeline(tr_df, te_df, cfg)
        assert res["oof_preds"].shape == (n_tr,)
        assert res["test_preds"].shape == (n_te,)
        assert os.path.exists(res["oof_path"])
        assert os.path.exists(res["test_path"])
        assert os.path.exists(res["sub_path"])
        print("[SMOKE TEST PASSED] catboost_oof_pipeline.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost OOF pipeline with native cat_features and text_features.")
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--task", type=str, default=CFG.task, choices=["binary", "multiclass", "regression"])
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--fold-col", type=str, default=CFG.fold_col)
    parser.add_argument("--cat-cols", type=str, nargs="*", default=[])
    parser.add_argument("--text-cols", type=str, nargs="*", default=[])
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
            cat_cols=args.cat_cols,
            text_cols=args.text_cols,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_catboost_oof_pipeline(tr, te, cfg)
