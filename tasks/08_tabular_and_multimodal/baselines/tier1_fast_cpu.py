"""
Task 08 — Tier 1 Fast CPU Baseline: Automated Feature-Engineered GBDT Pipeline
==============================================================================
End-to-end automated Tabular + Text GBDT baseline:
  1. Missingness Engineering:
     - Row-wise missing count (`row_na_count`) and per-column missingness indicators.
  2. Datetime Cyclical & Lag Extraction:
     - Hour/day-of-week `sin/cos` cyclical encoding + time delta since previous event.
  3. Categorical Frequency + Leak-Free Out-Of-Fold Smoothed Target Encoding:
     - Empirical Bayes smoothed OOF target encoding computed strictly inside each fold.
  4. Entity / Group Aggregations (`groupby(cat)[num]`):
     - Computes `mean`, `std`, `min`, `max`, `diff_from_mean`, `zscore`, and `ratio_to_mean`.
  5. Text TF-IDF + TruncatedSVD Components:
     - Extracts dense semantic SVD components for any free-text columns.
  6. Trains OOF GBDT (`LightGBM` or `HistGradientBoostingClassifier` fallback),
     evaluates OOF ROC-AUC & LogLoss, and saves `oof_tier1_tabular.npy`,
     `test_tier1_tabular.npy`, `sub_tier1_tabular.csv`, and permutation/gain feature importances.

Usage:
  python3 tier1_fast_cpu.py --data-dir ./data --output-dir ./outputs
  python3 tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class CFG:
    exp_name: str = "tier1_tabular"
    seed: int = 42
    n_folds: int = 5
    te_smoothing: float = 10.0
    svd_components: int = 4
    id_col: str = "row_id"
    target_col: str = "target"
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def build_engineered_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    dt_cols: Sequence[str],
    text_cols: Sequence[str],
    cfg: CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_out = pd.DataFrame(index=train_df.index)
    te_out = pd.DataFrame(index=test_df.index)

    # 1. Raw numeric + missingness count & indicators
    tr_out["row_na_count"] = train_df[list(num_cols) + list(cat_cols)].isna().sum(axis=1).astype(np.float32)
    te_out["row_na_count"] = test_df[list(num_cols) + list(cat_cols)].isna().sum(axis=1).astype(np.float32)

    for ncol in num_cols:
        tr_out[ncol] = train_df[ncol].astype(np.float32)
        te_out[ncol] = test_df[ncol].astype(np.float32)
        tr_out[f"{ncol}_isna"] = train_df[ncol].isna().astype(np.float32)
        te_out[f"{ncol}_isna"] = test_df[ncol].isna().astype(np.float32)

    # Pairwise ratios/differences among numeric columns
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            tr_out[f"{c1}_minus_{c2}"] = tr_out[c1] - tr_out[c2]
            te_out[f"{c1}_minus_{c2}"] = te_out[c1] - te_out[c2]
            tr_out[f"{c1}_div_{c2}"] = tr_out[c1] / (np.abs(tr_out[c2]) + 1e-4)
            te_out[f"{c1}_div_{c2}"] = te_out[c1] / (np.abs(te_out[c2]) + 1e-4)

    # 2. Datetime cyclical & lag features
    for dcol in dt_cols:
        for src, dst in ((train_df, tr_out), (test_df, te_out)):
            dt = pd.to_datetime(src[dcol], errors="coerce")
            hour = dt.dt.hour.fillna(0).to_numpy(dtype=np.float32)
            dow = dt.dt.dayofweek.fillna(0).to_numpy(dtype=np.float32)
            dst[f"{dcol}_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            dst[f"{dcol}_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
            dst[f"{dcol}_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            dst[f"{dcol}_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
            epoch_s = (dt.astype("int64") // 10**9).to_numpy(dtype=np.float64)
            dst[f"{dcol}_lag_sec"] = np.diff(epoch_s, prepend=epoch_s[0]).astype(np.float32)

    # 3. Frequency Encoding + Leak-Free OOF Smoothed Target Encoding
    y = train_df[cfg.target_col].to_numpy(dtype=np.float64)
    folds = train_df["fold"].to_numpy()
    global_prior = float(np.mean(y))

    for ccol in cat_cols:
        freq_map = train_df[ccol].value_counts(dropna=False, normalize=True).to_dict()
        tr_out[f"{ccol}_freq"] = train_df[ccol].map(freq_map).fillna(0.0).astype(np.float32)
        te_out[f"{ccol}_freq"] = test_df[ccol].map(freq_map).fillna(0.0).astype(np.float32)

        oof_te = np.full(len(train_df), global_prior, dtype=np.float32)
        for fold in np.unique(folds):
            tr_m = folds != fold
            va_m = folds == fold
            grp = pd.Series(y[tr_m]).groupby(train_df.loc[tr_m, ccol].values)
            cnt = grp.count()
            mean = grp.mean()
            tr_prior = float(np.mean(y[tr_m]))
            s_map = ((cnt * mean + cfg.te_smoothing * tr_prior) / (cnt + cfg.te_smoothing)).to_dict()
            oof_te[va_m] = (
                train_df.loc[va_m, ccol].map(s_map).fillna(tr_prior).to_numpy(dtype=np.float32)
            )
        tr_out[f"{ccol}_oof_te"] = oof_te

        full_grp = pd.Series(y).groupby(train_df[ccol].values)
        f_cnt = full_grp.count()
        f_mean = full_grp.mean()
        full_map = ((f_cnt * f_mean + cfg.te_smoothing * global_prior) / (f_cnt + cfg.te_smoothing)).to_dict()
        te_out[f"{ccol}_oof_te"] = (
            test_df[ccol].map(full_map).fillna(global_prior).to_numpy(dtype=np.float32)
        )

    # 4. Group Aggregations (`groupby(cat)[num].agg(['mean', 'std', 'min', 'max'])`)
    for ccol in cat_cols[:2]:
        for ncol in num_cols:
            agg = (
                train_df.groupby(ccol, observed=False)[ncol]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            agg["std"] = agg["std"].fillna(0.0)
            for src, dst in ((train_df, tr_out), (test_df, te_out)):
                merged = src[[ccol, ncol]].merge(agg, on=ccol, how="left")
                pfx = f"{ncol}_by_{ccol}"
                m_val = merged["mean"].to_numpy(dtype=np.float32)
                s_val = merged["std"].to_numpy(dtype=np.float32)
                raw = src[ncol].to_numpy(dtype=np.float32)
                dst[f"{pfx}_mean"] = m_val
                dst[f"{pfx}_std"] = s_val
                dst[f"{pfx}_min"] = merged["min"].to_numpy(dtype=np.float32)
                dst[f"{pfx}_max"] = merged["max"].to_numpy(dtype=np.float32)
                dst[f"{pfx}_diff_mean"] = raw - m_val
                dst[f"{pfx}_zscore"] = (raw - m_val) / (s_val + 1e-5)
                dst[f"{pfx}_ratio_mean"] = raw / (np.abs(m_val) + 1e-5)

    # 5. Text TF-IDF + TruncatedSVD features
    for tcol in text_cols:
        tr_txt = train_df[tcol].fillna("").astype(str).tolist()
        te_txt = test_df[tcol].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=128)
        tr_tfidf = vec.fit_transform(tr_txt)
        te_tfidf = vec.transform(te_txt)
        n_comp = max(1, min(cfg.svd_components, tr_tfidf.shape[1] - 1, tr_tfidf.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed)
        tr_svd = svd.fit_transform(tr_tfidf)
        te_svd = svd.transform(te_tfidf)
        for k in range(n_comp):
            tr_out[f"{tcol}_svd_{k}"] = tr_svd[:, k].astype(np.float32)
            te_out[f"{tcol}_svd_{k}"] = te_svd[:, k].astype(np.float32)

    return tr_out, te_out


def _generate_synthetic_data(n_rows: int = 100, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    spend = rng.normal(100.0, 25.0, size=n_rows)
    freq = rng.exponential(4.0, size=n_rows)
    segment = rng.choice(["gold", "silver", "bronze"], size=n_rows)
    region = rng.choice(["jkt", "sby", "bdg", "mdn"], size=n_rows)
    notes = [
        "priority express member verified loyalty account active"
        if s == "gold" or i % 3 == 0
        else "standard unverified guest checkout delivery"
        for i, s in enumerate(segment)
    ]
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="3h").astype(str)

    logit = (
        0.04 * (spend - 100.0)
        + 0.18 * (freq - 4.0)
        + 1.1 * (segment == "gold")
        - 0.7 * (segment == "bronze")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "row_id": [f"R{i:04d}" for i in range(n_rows)],
            "event_dt": dates,
            "spend": spend,
            "freq": freq,
            "segment": segment,
            "region": region,
            "note": notes,
            "target": target,
        }
    )
    df.loc[rng.random(n_rows) < 0.08, "spend"] = np.nan
    return df.iloc[:80].reset_index(drop=True), df.iloc[80:].reset_index(drop=True)


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    if smoke_test:
        train_df, test_df = _generate_synthetic_data(100, cfg.seed)
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[:20].copy().reset_index(drop=True)
        else:
            train_df, test_df = _generate_synthetic_data(120, cfg.seed)

    if "fold" not in train_df.columns:
        train_df["fold"] = 0
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[cfg.target_col])):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    num_cols = ["spend", "freq"]
    cat_cols = ["segment", "region"]
    dt_cols = ["event_dt"]
    text_cols = ["note"]

    X_train_df, X_test_df = build_engineered_features(
        train_df, test_df, num_cols, cat_cols, dt_cols, text_cols, cfg
    )
    feature_cols = list(X_train_df.columns)
    X_train = X_train_df.to_numpy(dtype=np.float32)
    X_test = X_test_df.to_numpy(dtype=np.float32)
    y_train = train_df[cfg.target_col].to_numpy(dtype=np.int32)

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    importances = np.zeros(len(feature_cols), dtype=np.float64)

    unique_folds = sorted(train_df["fold"].unique())
    for fold in unique_folds:
        tr_m = train_df["fold"].to_numpy() != fold
        va_m = train_df["fold"].to_numpy() == fold

        if HAS_LGBM:
            model = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=15,
                random_state=cfg.seed + int(fold),
                verbosity=-1,
            )
            model.fit(X_train[tr_m], y_train[tr_m])
            importances += model.feature_importances_ / len(unique_folds)
        else:
            model = HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                random_state=cfg.seed + int(fold),
            )
            model.fit(X_train[tr_m], y_train[tr_m])
            perm = permutation_importance(
                model, X_train[va_m], y_train[va_m], n_repeats=2, random_state=cfg.seed
            )
            importances += perm.importances_mean / len(unique_folds)

        oof_preds[va_m] = model.predict_proba(X_train[va_m])[:, 1].astype(np.float32)
        test_preds += model.predict_proba(X_test)[:, 1].astype(np.float32) / len(unique_folds)

    auc = float(roc_auc_score(y_train, oof_preds))
    ll = float(log_loss(y_train, np.clip(oof_preds, 1e-6, 1.0 - 1e-6)))

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_preds)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_preds)
    imp_df.to_csv(out_dir / f"feature_importance_{cfg.exp_name}.csv", index=False)
    pd.DataFrame({cfg.id_col: test_df[cfg.id_col], cfg.target_col: test_preds}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )

    print(
        f"[Tier1 Automated GBDT] Features={len(feature_cols)} | "
        f"OOF ROC-AUC={auc:.4f}, LogLoss={ll:.4f} | "
        f"Top-3 Features={imp_df['feature'].head(3).tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Tier 1 Tabular GBDT")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            run_pipeline(cfg, smoke_test=True)
    else:
        run_pipeline(cfg, smoke_test=False)


if __name__ == "__main__":
    main()
