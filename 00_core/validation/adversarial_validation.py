#!/usr/bin/env python3
"""
adversarial_validation.py
=========================
Adversarial Validation (Train vs. Test Classifier) for covariate shift & leakage triage.
Pipeline:
  1. Concatenates train (`is_test = 0`) and test (`is_test = 1`), dropping target/id/fold columns.
  2. Automatically encodes categorical columns and optionally extracts TF-IDF + TruncatedSVD features
     from text columns.
  3. Trains a 5-fold GBDT (`LightGBM` if installed, else `sklearn.ensemble.HistGradientBoostingClassifier`).
  4. Reports OOF ROC-AUC (warns if AUC > 0.60 indicating covariate shift).
  5. Prints top drifting features ranked by importance.
  6. Computes density-ratio importance weights w_i = p(test|x_i) / (1 - p(test|x_i)) for train samples
     to select a test-like validation subset or sample-weighted training set.

Usage:
  python3 00_core/validation/adversarial_validation.py --train-path train.parquet --test-path test.parquet --target-col target
  python3 00_core/validation/adversarial_validation.py --smoke-test
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


@dataclass
class CFG:
    train_path: str = "train.parquet"
    test_path: str = "test.parquet"
    target_col: str = "target"
    id_col: str = "id"
    text_cols: List[str] = field(default_factory=list)
    drop_cols: List[str] = field(default_factory=list)
    n_splits: int = 5
    svd_components: int = 8
    auc_warn_threshold: float = 0.60
    val_subset_fraction: float = 0.20
    output_weights_path: Optional[str] = None
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def build_adversarial_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    y_adv = np.concatenate([np.zeros(n_train, dtype=int), np.ones(len(test_df), dtype=int)])

    exclude = {cfg.target_col, cfg.id_col, "fold", "is_test"} | set(cfg.drop_cols)
    shared_cols = [
        c for c in train_df.columns
        if c in test_df.columns and c not in exclude
    ]

    features_list: List[pd.DataFrame] = []
    tab_cols = [c for c in shared_cols if c not in cfg.text_cols]
    if tab_cols:
        tab_df = combined[tab_cols].copy()
        for col in tab_df.columns:
            if not pd.api.types.is_numeric_dtype(tab_df[col]):
                codes, _ = pd.factorize(tab_df[col].fillna("__MISSING__").astype(str))
                tab_df[col] = codes.astype(np.float32)
            else:
                tab_df[col] = tab_df[col].astype(np.float32)
        features_list.append(tab_df)

    for tcol in cfg.text_cols:
        if tcol in combined.columns:
            texts = combined[tcol].fillna("").astype(str).tolist()
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
            X_tfidf = vec.fit_transform(texts)
            n_comp = min(cfg.svd_components, max(1, X_tfidf.shape[1] - 1))
            if n_comp >= 1 and X_tfidf.shape[1] > 1:
                svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed)
                X_svd = svd.fit_transform(X_tfidf)
                svd_df = pd.DataFrame(
                    X_svd,
                    columns=[f"{tcol}_svd_{i}" for i in range(n_comp)],
                )
                features_list.append(svd_df)

    if not features_list:
        raise ValueError("No usable shared features found between train and test.")
    X_all = pd.concat(features_list, axis=1)
    return X_all, y_adv, n_train


def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
) -> Dict[str, Any]:
    X_all, y_adv, n_train = build_adversarial_matrix(train_df, test_df, cfg)
    feature_names = list(X_all.columns)
    X_np = X_all.to_numpy(dtype=np.float32)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    oof_preds = np.zeros(len(X_all), dtype=np.float64)
    importances = np.zeros(len(feature_names), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_adv)):
        X_tr, y_tr = X_np[tr_idx], y_adv[tr_idx]
        X_va, y_va = X_np[val_idx], y_adv[val_idx]

        if lgb is not None:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample_for_bin=10000,
                random_state=cfg.seed + fold,
                verbosity=-1,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
            importances += model.feature_importances_ / cfg.n_splits
        else:
            model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=cfg.seed + fold,
            )
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
            perm = permutation_importance(
                model, X_va, y_va, n_repeats=3, random_state=cfg.seed + fold
            )
            importances += np.maximum(perm.importances_mean, 0.0) / cfg.n_splits

    oof_auc = float(roc_auc_score(y_adv, oof_preds))
    p_train = np.clip(oof_preds[:n_train], 1e-4, 1.0 - 1e-4)
    # Density ratio importance weight: w_i = p(test|x_i) / (1 - p(test|x_i))
    importance_weights = p_train / (1.0 - p_train)

    n_val_subset = max(1, int(round(n_train * cfg.val_subset_fraction)))
    top_test_like_train_indices = np.argsort(importance_weights)[::-1][:n_val_subset]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    drift_detected = bool(oof_auc > cfg.auc_warn_threshold)
    print("=" * 72)
    print(f"[ADVERSARIAL VALIDATION] OOF ROC-AUC: {oof_auc:.5f} (Backend: {'LightGBM' if lgb else 'HistGBDT'})")
    if drift_detected:
        print(f"  [WARNING] OOF AUC {oof_auc:.4f} > {cfg.auc_warn_threshold:.2f}! Significant covariate shift detected.")
        print("  -> Inspect top drifting features below; consider dropping pure ID/time proxies or using")
        print("     importance weights w_i = p(test|x)/(1 - p(test|x)) for validation selection.")
    else:
        print("  [OK] Train and Test distributions are indistinguishable (low covariate shift).")
    print("-" * 72)
    print("Top Drifting Features:")
    print(imp_df.head(10).to_string(index=False))
    print("-" * 72)
    print(f"Importance weights w_i summary on Train: min={importance_weights.min():.4f}, "
          f"median={np.median(importance_weights):.4f}, max={importance_weights.max():.4f}")
    print("=" * 72)

    if cfg.output_weights_path:
        out_df = pd.DataFrame({
            "train_index": np.arange(n_train),
            "p_is_test": p_train,
            "importance_weight": importance_weights,
            "is_top_test_like_val": np.isin(np.arange(n_train), top_test_like_train_indices),
        })
        Path(cfg.output_weights_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(cfg.output_weights_path, index=False)

    return {
        "oof_auc": oof_auc,
        "drift_detected": drift_detected,
        "feature_importance": imp_df,
        "train_importance_weights": importance_weights,
        "top_test_like_train_indices": top_test_like_train_indices,
    }


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 250, 150
    tr_df = pd.DataFrame({
        "id": np.arange(n_tr),
        "stable_f1": rng.normal(0.0, 1.0, size=n_tr),
        "stable_f2": rng.uniform(-1.0, 1.0, size=n_tr),
        "drifted_feature": rng.normal(0.0, 1.0, size=n_tr),
        "text": [f"normal sample text {i % 10}" for i in range(n_tr)],
        "target": rng.integers(0, 2, size=n_tr),
    })
    te_df = pd.DataFrame({
        "id": np.arange(n_tr, n_tr + n_te),
        "stable_f1": rng.normal(0.0, 1.0, size=n_te),
        "stable_f2": rng.uniform(-1.0, 1.0, size=n_te),
        "drifted_feature": rng.normal(2.5, 1.0, size=n_te),
        "text": [f"normal sample text {i % 10}" for i in range(n_te)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_csv = os.path.join(tmpdir, "adv_weights.csv")
        cfg = CFG(
            target_col="target",
            id_col="id",
            text_cols=["text"],
            output_weights_path=weights_csv,
        )
        res = run_adversarial_validation(tr_df, te_df, cfg)
        assert res["oof_auc"] > 0.80, f"Expected high AUC on shifted data, got {res['oof_auc']}"
        assert res["drift_detected"] is True
        assert res["feature_importance"].iloc[0]["feature"] == "drifted_feature"
        assert len(res["train_importance_weights"]) == n_tr
        assert os.path.exists(weights_csv)
        print("[SMOKE TEST PASSED] adversarial_validation.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial validation (train vs test drift detector).")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--text-cols", type=str, nargs="*", default=[])
    parser.add_argument("--drop-cols", type=str, nargs="*", default=[])
    parser.add_argument("--output-weights-path", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        cfg = CFG(
            train_path=args.train_path,
            test_path=args.test_path,
            target_col=args.target_col,
            id_col=args.id_col,
            text_cols=args.text_cols,
            drop_cols=args.drop_cols,
            output_weights_path=args.output_weights_path,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path)
        run_adversarial_validation(tr, te, cfg)
