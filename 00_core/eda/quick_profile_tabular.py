#!/usr/bin/env python3
"""
quick_profile_tabular.py
========================
Fast 15-minute competition tabular EDA & train-test drift profiler.
Audits:
  - Shape, memory footprint, and dtypes
  - Missingness % and missingness correlation with target
  - Cardinality, constant (n_unique <= 1), and quasi-constant columns (top freq >= 99%)
  - ID uniqueness/overlap and timestamp monotonicity checks
  - Numeric outlier quantiles (p01, p05, p50, p95, p99)
  - Target distribution (class balance or regression skewness/quantiles)
  - Train-Test distribution mismatch:
      * 2-sample Kolmogorov-Smirnov (KS) statistic & p-value for numeric columns
      * Total Variation Distance (TVD), Chi-square test, and unseen test category count for categoricals

Usage:
  python3 00_core/eda/quick_profile_tabular.py --train-path train.parquet --test-path test.parquet --target-col target
  python3 00_core/eda/quick_profile_tabular.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    train_path: str = "train.parquet"
    test_path: Optional[str] = "test.parquet"
    target_col: Optional[str] = "target"
    id_col: Optional[str] = "id"
    time_col: Optional[str] = None
    quasi_constant_threshold: float = 0.99
    ks_drift_warn_threshold: float = 0.10
    tvd_drift_warn_threshold: float = 0.10
    max_cat_cardinality_for_tvd: int = 500
    quantiles: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.50, 0.95, 0.99])
    output_json: Optional[str] = None
    seed: int = 42


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def audit_column_stats(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    n_rows = len(df)
    records: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        n_miss = int(s.isna().sum())
        miss_pct = 100.0 * n_miss / max(n_rows, 1)
        nunique = int(s.nunique(dropna=True))
        top_freq_ratio = 0.0
        if n_rows - n_miss > 0:
            vc = s.value_counts(dropna=True)
            if len(vc) > 0:
                top_freq_ratio = float(vc.iloc[0] / (n_rows - n_miss))

        is_constant = nunique <= 1
        is_quasi_constant = (not is_constant) and (top_freq_ratio >= cfg.quasi_constant_threshold)

        row: Dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": n_miss,
            "missing_pct": round(miss_pct, 3),
            "n_unique": nunique,
            "top_val_ratio": round(top_freq_ratio, 4),
            "is_constant": is_constant,
            "is_quasi_constant": is_quasi_constant,
        }

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            clean = s.dropna().astype(float)
            if len(clean) > 0:
                q_vals = np.quantile(clean, cfg.quantiles)
                for q, val in zip(cfg.quantiles, q_vals):
                    q_name = f"p{int(round(q * 100)):02d}"
                    row[q_name] = float(np.round(val, 5))
                row["mean"] = float(np.round(clean.mean(), 5))
                row["std"] = float(np.round(clean.std(ddof=0), 5))
            else:
                for q in cfg.quantiles:
                    row[f"p{int(round(q * 100)):02d}"] = np.nan
                row["mean"] = np.nan
                row["std"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def check_id_and_timestamp(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if cfg.id_col and cfg.id_col in train_df.columns:
        tr_ids = train_df[cfg.id_col]
        report["id_col"] = cfg.id_col
        report["train_id_unique"] = bool(tr_ids.is_unique)
        if pd.api.types.is_numeric_dtype(tr_ids):
            report["train_id_monotonic_increasing"] = bool(tr_ids.is_monotonic_increasing)
        if test_df is not None and cfg.id_col in test_df.columns:
            te_ids = test_df[cfg.id_col]
            report["test_id_unique"] = bool(te_ids.is_unique)
            overlap = len(set(tr_ids.dropna()).intersection(set(te_ids.dropna())))
            report["train_test_id_overlap_count"] = int(overlap)

    if cfg.time_col and cfg.time_col in train_df.columns:
        tr_t = pd.to_datetime(train_df[cfg.time_col], errors="coerce")
        report["time_col"] = cfg.time_col
        report["train_time_monotonic_increasing"] = bool(tr_t.is_monotonic_increasing)
        report["train_time_min"] = str(tr_t.min())
        report["train_time_max"] = str(tr_t.max())
        if test_df is not None and cfg.time_col in test_df.columns:
            te_t = pd.to_datetime(test_df[cfg.time_col], errors="coerce")
            report["test_time_monotonic_increasing"] = bool(te_t.is_monotonic_increasing)
            report["test_time_min"] = str(te_t.min())
            report["test_time_max"] = str(te_t.max())
            report["temporal_split_clean"] = bool(tr_t.max() <= te_t.min())
    return report


def profile_target(train_df: pd.DataFrame, cfg: CFG) -> Dict[str, Any]:
    if not cfg.target_col or cfg.target_col not in train_df.columns:
        return {}
    y = train_df[cfg.target_col]
    n_unique = int(y.nunique(dropna=True))
    info: Dict[str, Any] = {
        "target_col": cfg.target_col,
        "dtype": str(y.dtype),
        "missing_count": int(y.isna().sum()),
        "n_unique": n_unique,
    }
    if n_unique <= 30 or not pd.api.types.is_numeric_dtype(y):
        vc = y.value_counts(dropna=False)
        vc_pct = (vc / max(len(y), 1) * 100.0).round(2)
        info["task_hint"] = "classification"
        info["class_distribution"] = {
            str(k): {"count": int(vc[k]), "pct": float(vc_pct[k])} for k in vc.index
        }
    else:
        clean = y.dropna().astype(float)
        info["task_hint"] = "regression"
        info["mean"] = float(clean.mean())
        info["std"] = float(clean.std(ddof=0))
        info["skew"] = float(stats.skew(clean))
        info["kurtosis"] = float(stats.kurtosis(clean))
        q_vals = np.quantile(clean, cfg.quantiles)
        info["quantiles"] = {
            f"p{int(round(q * 100)):02d}": float(np.round(v, 5))
            for q, v in zip(cfg.quantiles, q_vals)
        }
    return info


def compare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: CFG,
) -> Dict[str, Any]:
    tr_cols = set(train_df.columns)
    te_cols = set(test_df.columns)
    only_in_train = sorted(list(tr_cols - te_cols - {cfg.target_col}))
    only_in_test = sorted(list(te_cols - tr_cols))
    shared_cols = [
        c for c in train_df.columns
        if c in te_cols and c not in {cfg.target_col, cfg.id_col, cfg.time_col}
    ]

    drift_rows: List[Dict[str, Any]] = []
    for col in shared_cols:
        s_tr = train_df[col].dropna()
        s_te = test_df[col].dropna()
        if len(s_tr) == 0 or len(s_te) == 0:
            continue

        if pd.api.types.is_numeric_dtype(s_tr) and not pd.api.types.is_bool_dtype(s_tr):
            ks_stat, ks_pval = stats.ks_2samp(s_tr.astype(float), s_te.astype(float))
            flagged = bool(ks_stat >= cfg.ks_drift_warn_threshold)
            drift_rows.append({
                "column": col,
                "type": "numeric",
                "metric": "ks_stat",
                "statistic": round(float(ks_stat), 5),
                "p_value": float(ks_pval),
                "unseen_in_test": 0,
                "drift_flagged": flagged,
            })
        else:
            tr_str = s_tr.astype(str)
            te_str = s_te.astype(str)
            tr_set = set(tr_str.unique())
            te_set = set(te_str.unique())
            unseen = len(te_set - tr_set)

            if len(tr_set | te_set) <= cfg.max_cat_cardinality_for_tvd:
                tr_freq = tr_str.value_counts(normalize=True)
                te_freq = te_str.value_counts(normalize=True)
                all_cats = sorted(list(tr_set | te_set))
                p = np.array([tr_freq.get(c, 0.0) for c in all_cats], dtype=float)
                q = np.array([te_freq.get(c, 0.0) for c in all_cats], dtype=float)
                tvd = 0.5 * float(np.abs(p - q).sum())

                # Chi-square homogeneity test on contingency counts
                tr_cnt = tr_str.value_counts()
                te_cnt = te_str.value_counts()
                contingency = np.array([
                    [tr_cnt.get(c, 0) for c in all_cats],
                    [te_cnt.get(c, 0) for c in all_cats],
                ], dtype=float)
                if contingency.shape[1] >= 2 and contingency.sum() > 0:
                    _, chi2_p, _, _ = stats.chi2_contingency(contingency)
                else:
                    chi2_p = 1.0

                drift_rows.append({
                    "column": col,
                    "type": "categorical",
                    "metric": "tvd",
                    "statistic": round(tvd, 5),
                    "p_value": float(chi2_p),
                    "unseen_in_test": int(unseen),
                    "drift_flagged": bool(tvd >= cfg.tvd_drift_warn_threshold or unseen > 0),
                })

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values("statistic", ascending=False).reset_index(drop=True)

    return {
        "only_in_train_excluding_target": only_in_train,
        "only_in_test": only_in_test,
        "drift_table": drift_df.to_dict(orient="records") if not drift_df.empty else [],
    }


def run_profile(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    col_stats_df = audit_column_stats(train_df, cfg)
    constant_cols = col_stats_df.loc[col_stats_df["is_constant"], "column"].tolist()
    quasi_constant_cols = col_stats_df.loc[col_stats_df["is_quasi_constant"], "column"].tolist()
    id_time_report = check_id_and_timestamp(train_df, test_df, cfg)
    target_report = profile_target(train_df, cfg)
    mismatch_report = compare_train_test(train_df, test_df, cfg) if test_df is not None else {}

    summary = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape) if test_df is not None else None,
        "train_memory_mb": round(float(train_df.memory_usage(deep=True).sum() / (1024 ** 2)), 3),
        "constant_columns": constant_cols,
        "quasi_constant_columns": quasi_constant_cols,
        "id_and_timestamp": id_time_report,
        "target_profile": target_report,
        "column_stats": col_stats_df.to_dict(orient="records"),
        "train_test_comparison": mismatch_report,
    }

    print("=" * 72)
    print(f"[TABULAR PROFILE] Train shape: {train_df.shape} | Test shape: {test_df.shape if test_df is not None else 'N/A'}")
    print(f"Memory (Train): {summary['train_memory_mb']:.2f} MB")
    print(f"Constant cols ({len(constant_cols)}): {constant_cols}")
    print(f"Quasi-constant cols ({len(quasi_constant_cols)}): {quasi_constant_cols}")
    if id_time_report:
        print(f"ID / Time checks: {id_time_report}")
    if target_report:
        print(f"Target ({cfg.target_col}): task_hint={target_report.get('task_hint')} | n_unique={target_report.get('n_unique')}")
    print("-" * 72)
    print("Top Columns by Missingness / Quantiles:")
    print(col_stats_df.head(10).to_string(index=False))
    if mismatch_report and mismatch_report.get("drift_table"):
        print("-" * 72)
        print("Train-Test Drift Audit (KS for numeric, TVD + Chi2 for categorical):")
        print(pd.DataFrame(mismatch_report["drift_table"]).head(10).to_string(index=False))
    print("=" * 72)

    if cfg.output_json:
        Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n_tr, n_te = 300, 150
    with tempfile.TemporaryDirectory() as tmpdir:
        tr_df = pd.DataFrame({
            "id": np.arange(n_tr),
            "timestamp": pd.date_range("2025-01-01", periods=n_tr, freq="h").astype(str),
            "num_stable": rng.normal(0.0, 1.0, size=n_tr),
            "num_drifted": rng.normal(0.0, 1.0, size=n_tr),
            "const_col": np.ones(n_tr),
            "quasi_const_col": np.where(rng.random(n_tr) < 0.995, 1, 0),
            "cat_col": rng.choice(["A", "B", "C"], size=n_tr, p=[0.6, 0.3, 0.1]),
            "target": rng.choice([0, 1], size=n_tr, p=[0.75, 0.25]),
        })
        tr_df.loc[rng.choice(n_tr, 15, replace=False), "num_stable"] = np.nan

        te_df = pd.DataFrame({
            "id": np.arange(n_tr, n_tr + n_te),
            "timestamp": pd.date_range("2025-02-01", periods=n_te, freq="h").astype(str),
            "num_stable": rng.normal(0.0, 1.0, size=n_te),
            "num_drifted": rng.normal(1.2, 1.5, size=n_te),
            "const_col": np.ones(n_te),
            "quasi_const_col": np.ones(n_te, dtype=int),
            "cat_col": rng.choice(["A", "B", "D_unseen"], size=n_te, p=[0.2, 0.5, 0.3]),
        })

        tr_path = os.path.join(tmpdir, "train.parquet")
        te_path = os.path.join(tmpdir, "test.parquet")
        out_json = os.path.join(tmpdir, "profile.json")
        tr_df.to_parquet(tr_path, index=False)
        te_df.to_parquet(te_path, index=False)

        cfg = CFG(
            train_path=tr_path,
            test_path=te_path,
            target_col="target",
            id_col="id",
            time_col="timestamp",
            output_json=out_json,
        )
        summary = run_profile(load_dataframe(tr_path), load_dataframe(te_path), cfg)
        assert "const_col" in summary["constant_columns"]
        assert "quasi_const_col" in summary["quasi_constant_columns"]
        assert summary["id_and_timestamp"]["train_id_monotonic_increasing"] is True
        assert summary["id_and_timestamp"]["temporal_split_clean"] is True
        assert os.path.exists(out_json)
        print("[SMOKE TEST PASSED] quick_profile_tabular.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast tabular EDA and train-test drift profiler.")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--id-col", type=str, default=CFG.id_col)
    parser.add_argument("--time-col", type=str, default=CFG.time_col)
    parser.add_argument("--output-json", type=str, default=None)
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
            time_col=args.time_col,
            output_json=args.output_json,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_profile(tr, te, cfg)
