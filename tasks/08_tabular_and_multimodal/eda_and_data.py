"""
Task 08 — Tabular & Multimodal: Automatic Schema Detector & Feature Diagnostics
===============================================================================
Comprehensive exploratory diagnostics for structured tabular + text datasets:
  1. Automatic Multimodal Schema Detector:
     Classifies columns into `id_cols`, `datetime_cols`, `text_cols`,
     `cat_low_card`, `cat_high_card`, and `num_cols`.
  2. Missingness Correlation Matrix:
     Identifies pairs of features whose missingness patterns (`isna()`) co-occur
     (often indicating data-source joins or sensor/form section skips).
  3. Target Association Ranking:
     Computes Pearson `r`, Spearman `rho`, and Mutual Information against `target`.
  4. High-Collinearity Pair Detector:
     Flags redundant numeric pairs with `|corr| >= 0.95`.
  5. Exports `train_folds.parquet` with leak-free `StratifiedKFold` / `KFold`.

Usage:
  python3 eda_and_data.py --data-dir ./data --output-dir ./outputs
  python3 eda_and_data.py --smoke-test
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
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class CFG:
    exp_name: str = "eda_tabular_multimodal"
    seed: int = 42
    n_folds: int = 5
    target_col: str = "target"
    high_card_threshold: int = 20
    text_min_avg_words: float = 4.0
    collinearity_threshold: float = 0.95
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def detect_multimodal_schema(
    df: pd.DataFrame, cfg: CFG
) -> Dict[str, List[str]]:
    """
    Automatically classifies DataFrame columns into:
      - `id_cols`: monotonic / unique identifier columns
      - `datetime_cols`: datetime64 or parseable ISO timestamp columns
      - `text_cols`: string columns with average word count >= `text_min_avg_words`
      - `cat_low_card`: categorical/string/integer columns with `<= high_card_threshold` unique values
      - `cat_high_card`: categorical/string columns with `> high_card_threshold` unique values
      - `num_cols`: continuous numerical columns
    """
    schema: Dict[str, List[str]] = {
        "id_cols": [],
        "datetime_cols": [],
        "text_cols": [],
        "cat_low_card": [],
        "cat_high_card": [],
        "num_cols": [],
    }
    n_rows = max(len(df), 1)

    for col in df.columns:
        if col in (cfg.target_col, "fold"):
            continue
        series = df[col]
        n_unique = series.nunique(dropna=True)

        # 1. Check explicit ID column name and near-100% uniqueness
        if col.lower() in ("id", "row_id", "sample_id") or (
            col.lower().endswith("_id") and n_unique >= 0.95 * n_rows and series.dtype == object
        ):
            schema["id_cols"].append(col)
            continue

        # 2. Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            schema["datetime_cols"].append(col)
            continue
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            sample_vals = series.dropna().astype(str).head(20)
            if len(sample_vals) > 0 and ("date" in col.lower() or "time" in col.lower()):
                try:
                    pd.to_datetime(sample_vals, errors="raise")
                    schema["datetime_cols"].append(col)
                    continue
                except Exception:
                    pass

            # 3. Distinguish free text vs categorical
            avg_words = float(sample_vals.str.split().str.len().mean()) if len(sample_vals) > 0 else 0.0
            if avg_words >= cfg.text_min_avg_words:
                schema["text_cols"].append(col)
            elif n_unique <= cfg.high_card_threshold:
                schema["cat_low_card"].append(col)
            else:
                schema["cat_high_card"].append(col)
            continue

        # 4. Numeric or boolean columns
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            if pd.api.types.is_integer_dtype(series) and n_unique <= 10 and n_rows > 30:
                schema["cat_low_card"].append(col)
            else:
                schema["num_cols"].append(col)

    return schema


def compute_missingness_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise correlation between binary missingness indicators (`df.isna()`)."""
    miss_cols = [c for c in df.columns if df[c].isna().sum() > 0]
    if len(miss_cols) < 2:
        return pd.DataFrame()
    return df[miss_cols].isna().astype(np.float32).corr()


def rank_target_correlations(
    df: pd.DataFrame, num_cols: Sequence[str], target_col: str, seed: int = 42
) -> pd.DataFrame:
    """Ranks numerical features by Pearson r, Spearman rho, and Mutual Information."""
    if not num_cols or target_col not in df.columns:
        return pd.DataFrame()

    y = df[target_col].to_numpy(dtype=np.float64)
    X_filled = df[list(num_cols)].copy()
    for c in num_cols:
        X_filled[c] = X_filled[c].fillna(X_filled[c].median())

    is_binary_or_int = len(np.unique(y[~np.isnan(y)])) <= 10
    if is_binary_or_int:
        mi = mutual_info_classif(
            X_filled.to_numpy(dtype=np.float64), y.astype(int), random_state=seed
        )
    else:
        mi = mutual_info_regression(
            X_filled.to_numpy(dtype=np.float64), y, random_state=seed
        )

    rows = []
    for idx, col in enumerate(num_cols):
        x = X_filled[col].to_numpy(dtype=np.float64)
        if np.std(x) < 1e-12:
            p_r, s_r = 0.0, 0.0
        else:
            p_r = float(pearsonr(x, y)[0])
            s_r = float(spearmanr(x, y)[0])
        rows.append(
            {
                "feature": col,
                "pearson_r": p_r,
                "abs_pearson": abs(p_r),
                "spearman_rho": s_r,
                "mutual_info": float(mi[idx]),
                "missing_rate": float(df[col].isna().mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def detect_collinear_pairs(
    df: pd.DataFrame, num_cols: Sequence[str], threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    if len(num_cols) < 2:
        return []
    corr = df[list(num_cols)].corr().abs()
    pairs: List[Tuple[str, str, float]] = []
    cols = list(num_cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr.iloc[i, j])
            if val >= threshold:
                pairs.append((cols[i], cols[j], val))
    return sorted(pairs, key=lambda x: -x[2])


def generate_synthetic_multimodal_data(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num1 = rng.normal(50.0, 12.0, size=n_rows)
    num2 = rng.exponential(10.0, size=n_rows)
    # Deliberately collinear feature to test detector
    num3_collinear = num1 * 1.02 + rng.normal(0.0, 0.2, size=n_rows)

    # Inject correlated missingness
    miss_mask = rng.random(n_rows) < 0.12
    num1_with_na = num1.copy()
    num2_with_na = num2.copy()
    num1_with_na[miss_mask] = np.nan
    num2_with_na[miss_mask] = np.nan

    cats_low = rng.choice(["tier_A", "tier_B", "tier_C"], size=n_rows)
    cats_high = [f"merchant_{i % 30:02d}" for i in range(n_rows)]
    texts = [
        "customer requested express delivery with fragile packaging protection"
        if i % 2 == 0
        else "standard economy shipping order for regular household items"
        for i in range(n_rows)
    ]
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="6h").astype(str)
    logits = 0.08 * (num1 - 50.0) - 0.05 * num2 + (cats_low == "tier_A") * 0.9
    probs = 1.0 / (1.0 + np.exp(-logits))
    target = (rng.random(n_rows) < probs).astype(int)

    return pd.DataFrame(
        {
            "row_id": [f"R{i:04d}" for i in range(n_rows)],
            "event_time": dates,
            "num_spend": num1_with_na,
            "num_latency": num2_with_na,
            "num_spend_dup": num3_collinear,
            "segment": cats_low,
            "merchant_code": cats_high,
            "review_note": texts,
            "target": target,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Tabular & Multimodal EDA")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)

    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            df = generate_synthetic_multimodal_data(80, seed=cfg.seed)
            schema = detect_multimodal_schema(df, cfg)
            miss_corr = compute_missingness_correlations(df)
            target_rank = rank_target_correlations(df, schema["num_cols"], cfg.target_col, cfg.seed)
            collinear = detect_collinear_pairs(
                df, schema["num_cols"], threshold=cfg.collinearity_threshold
            )

            df["fold"] = 0
            skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
            for fold, (_, val_idx) in enumerate(skf.split(df, df[cfg.target_col])):
                df.iloc[val_idx, df.columns.get_loc("fold")] = fold

            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / "train_folds.parquet", index=False)

            oof_dummy = df[cfg.target_col].to_numpy(dtype=np.float32)
            np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_dummy)
            np.save(out_dir / f"test_{cfg.exp_name}.npy", oof_dummy[:10])
            pd.DataFrame(
                {"row_id": df["row_id"].iloc[:10], "target": oof_dummy[:10]}
            ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

            print("=== Detected Multimodal Schema ===")
            for k, v in schema.items():
                print(f"  {k:<16}: {v}")
            print(
                f"[Smoke Test OK] Folds shape={df.shape} | "
                f"Top feature={target_rank.iloc[0]['feature']} | "
                f"Collinear pairs={collinear} | "
                f"Missingness corr shape={miss_corr.shape}"
            )
        return

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        train_path = data_dir / "train.csv"
    df = (
        pd.read_parquet(train_path)
        if train_path.exists() and train_path.suffix == ".parquet"
        else (
            pd.read_csv(train_path)
            if train_path.exists()
            else generate_synthetic_multimodal_data(100, cfg.seed)
        )
    )
    schema = detect_multimodal_schema(df, cfg)
    print("Schema:", schema)
    df["fold"] = 0
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold
    df.to_parquet(out_dir / "train_folds.parquet", index=False)


if __name__ == "__main__":
    main()
