"""
Task 08 — Reusable Tabular & Multimodal Feature Engineering Modules
===================================================================
Leak-free, competition-grade feature engineering transformers:
  1. `OOFTargetEncoder`: Out-Of-Fold Empirical Bayes smoothed target encoding
     for binary, continuous, and multiclass targets.
  2. `GroupAggregationTransformer`: Computes `groupby(cat)[num].agg(['mean', 'std', 'min', 'max'])`
     plus within-group `z_score` (`(x - mean) / std`) and `ratio_to_mean` (`x / mean`).
  3. `NumericInteractionGenerator`: Builds pairwise ratios, differences, and products
     among top numerical features.
  4. `DatetimeFeatureExtractor`: Extracts cyclical `sin/cos` calendar features,
     epoch timestamps, and inter-event time deltas.
  5. `RankGaussScaler`: `QuantileTransformer(output_distribution='normal')` wrapper
     essential for training Neural Tabular models alongside GBDTs.

Usage:
  python3 feature_engineering.py --smoke-test
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
from sklearn.preprocessing import QuantileTransformer


@dataclass
class CFG:
    exp_name: str = "tabular_fe_modules"
    seed: int = 42
    smoothing_m: float = 10.0


class OOFTargetEncoder:
    """
    Leak-free Out-Of-Fold Empirical Bayes Smoothed Target Encoder.
    Formula: `TE(c) = (n_c * mean_c + m * global_prior) / (n_c + m)`
    Supports binary/continuous 1D targets as well as multiclass targets (one-vs-rest columns).
    """

    def __init__(self, cols: Sequence[str], smoothing: float = 10.0, multiclass: bool = False) -> None:
        self.cols = list(cols)
        self.smoothing = smoothing
        self.multiclass = multiclass
        self.global_priors_: Dict[str, float] = {}
        self.mapping_: Dict[Tuple[str, str], Dict[object, float]] = {}
        self.classes_: List = []

    def fit_transform_oof(
        self, df: pd.DataFrame, target: Sequence | np.ndarray, fold_col: str = "fold"
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        y = np.asarray(target)
        folds = df[fold_col].to_numpy()
        unique_folds = np.unique(folds)

        if self.multiclass:
            self.classes_ = sorted(np.unique(y).tolist())
            target_cols = {f"cls_{c}": (y == c).astype(np.float64) for c in self.classes_}
        else:
            target_cols = {"target": y.astype(np.float64)}

        for t_name, y_vec in target_cols.items():
            global_prior = float(np.mean(y_vec))
            self.global_priors_[t_name] = global_prior

            for col in self.cols:
                feat_name = f"{col}_te" if not self.multiclass else f"{col}_te_{t_name}"
                oof_vals = np.full(len(df), global_prior, dtype=np.float32)

                # Compute OOF encoding strictly inside each training fold split
                for f in unique_folds:
                    tr_mask = folds != f
                    va_mask = folds == f
                    tr_s = pd.Series(y_vec[tr_mask]).groupby(df.loc[tr_mask, col].values)
                    cnts = tr_s.count()
                    means = tr_s.mean()
                    tr_prior = float(np.mean(y_vec[tr_mask]))
                    smooth_map = (
                        (cnts * means + self.smoothing * tr_prior) / (cnts + self.smoothing)
                    ).to_dict()
                    oof_vals[va_mask] = (
                        df.loc[va_mask, col].map(smooth_map).fillna(tr_prior).to_numpy(dtype=np.float32)
                    )

                out[feat_name] = oof_vals

                # Fit full-dataset mapping for test transform
                full_s = pd.Series(y_vec).groupby(df[col].values)
                f_cnts = full_s.count()
                f_means = full_s.mean()
                self.mapping_[(col, t_name)] = (
                    (f_cnts * f_means + self.smoothing * global_prior)
                    / (f_cnts + self.smoothing)
                ).to_dict()

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        t_names = [f"cls_{c}" for c in self.classes_] if self.multiclass else ["target"]
        for t_name in t_names:
            prior = self.global_priors_[t_name]
            for col in self.cols:
                feat_name = f"{col}_te" if not self.multiclass else f"{col}_te_{t_name}"
                s_map = self.mapping_.get((col, t_name), {})
                out[feat_name] = df[col].map(s_map).fillna(prior).astype(np.float32)
        return out


class GroupAggregationTransformer:
    """
    Computes `groupby(group_col)[num_col]` statistics (`mean`, `std`, `min`, `max`)
    plus entity-relative `zscore` and `ratio_to_mean` (1st-place AmEx / Home Credit pattern).
    """

    def __init__(self, group_cols: Sequence[str], num_cols: Sequence[str]) -> None:
        self.group_cols = list(group_cols)
        self.num_cols = list(num_cols)
        self.stats_tables_: Dict[Tuple[str, str], pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame) -> "GroupAggregationTransformer":
        for gcol in self.group_cols:
            for ncol in self.num_cols:
                agg = (
                    df.groupby(gcol, observed=False)[ncol]
                    .agg(["mean", "std", "min", "max"])
                    .reset_index()
                )
                agg["std"] = agg["std"].fillna(0.0)
                self.stats_tables_[(gcol, ncol)] = agg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for (gcol, ncol), agg in self.stats_tables_.items():
            merged = df[[gcol, ncol]].merge(agg, on=gcol, how="left")
            prefix = f"{ncol}_by_{gcol}"
            mean_v = merged["mean"].to_numpy(dtype=np.float32)
            std_v = merged["std"].to_numpy(dtype=np.float32)
            raw_v = df[ncol].to_numpy(dtype=np.float32)

            out[f"{prefix}_mean"] = mean_v
            out[f"{prefix}_std"] = std_v
            out[f"{prefix}_min"] = merged["min"].to_numpy(dtype=np.float32)
            out[f"{prefix}_max"] = merged["max"].to_numpy(dtype=np.float32)
            out[f"{prefix}_zscore"] = (raw_v - mean_v) / (std_v + 1e-5)
            out[f"{prefix}_ratio"] = raw_v / (np.abs(mean_v) + 1e-5)
        return out


class NumericInteractionGenerator:
    """Generates pairwise ratios, differences, and products for numerical columns."""

    def __init__(self, cols: Sequence[str]) -> None:
        self.cols = list(cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for i in range(len(self.cols)):
            for j in range(i + 1, len(self.cols)):
                c1, c2 = self.cols[i], self.cols[j]
                v1 = df[c1].to_numpy(dtype=np.float32)
                v2 = df[c2].to_numpy(dtype=np.float32)
                out[f"{c1}_minus_{c2}"] = v1 - v2
                out[f"{c1}_times_{c2}"] = v1 * v2
                out[f"{c1}_div_{c2}"] = v1 / (np.abs(v2) + 1e-5)
        return out


class DatetimeFeatureExtractor:
    """Extracts cyclical calendar encodings (`sin`/`cos`), epoch time, and event lags."""

    def __init__(self, datetime_cols: Sequence[str]) -> None:
        self.datetime_cols = list(datetime_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.datetime_cols:
            dt = pd.to_datetime(df[col], errors="coerce")
            hour = dt.dt.hour.fillna(0).to_numpy(dtype=np.float32)
            dow = dt.dt.dayofweek.fillna(0).to_numpy(dtype=np.float32)
            month = dt.dt.month.fillna(1).to_numpy(dtype=np.float32)

            out[f"{col}_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            out[f"{col}_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
            out[f"{col}_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            out[f"{col}_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
            out[f"{col}_is_weekend"] = (dow >= 5).astype(np.float32)
            out[f"{col}_month"] = month
            epoch_sec = (dt.astype("int64") // 10**9).to_numpy(dtype=np.float64)
            out[f"{col}_diff_prev_sec"] = np.diff(epoch_sec, prepend=epoch_sec[0]).astype(
                np.float32
            )
        return out


class RankGaussScaler:
    """
    Applies median imputation + missing indicator + `QuantileTransformer(output_distribution='normal')`
    (RankGauss), transforming skewed/heavy-tailed tabular features into smooth Gaussians for Neural Nets.
    """

    def __init__(self, n_quantiles: int = 100, random_state: int = 42) -> None:
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.medians_: np.ndarray = np.empty(0)
        self.qt_: QuantileTransformer | None = None

    def fit(self, X: np.ndarray) -> "RankGaussScaler":
        X_arr = np.asarray(X, dtype=np.float32)
        self.medians_ = np.nanmedian(X_arr, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        X_filled = np.where(np.isnan(X_arr), self.medians_[None, :], X_arr)
        n_q = max(5, min(self.n_quantiles, len(X_filled)))
        self.qt_ = QuantileTransformer(
            n_quantiles=n_q, output_distribution="normal", random_state=self.random_state
        )
        self.qt_.fit(X_filled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_filled = np.where(np.isnan(X_arr), self.medians_[None, :], X_arr)
        assert self.qt_ is not None
        return self.qt_.transform(X_filled).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 60
    df = pd.DataFrame(
        {
            "cat_a": rng.choice(["A", "B", "C"], size=n),
            "num_1": rng.normal(100.0, 20.0, size=n),
            "num_2": rng.exponential(5.0, size=n),
            "event_dt": pd.date_range("2025-03-01", periods=n, freq="4h").astype(str),
            "fold": np.arange(n) % 5,
            "target": rng.integers(0, 2, size=n),
        }
    )

    te = OOFTargetEncoder(cols=["cat_a"], smoothing=cfg.smoothing_m)
    te_oof = te.fit_transform_oof(df, df["target"], fold_col="fold")
    te_test = te.transform(df.iloc[:10])

    grp = GroupAggregationTransformer(group_cols=["cat_a"], num_cols=["num_1", "num_2"]).fit(df)
    grp_feats = grp.transform(df)

    inter_feats = NumericInteractionGenerator(cols=["num_1", "num_2"]).transform(df)
    dt_feats = DatetimeFeatureExtractor(datetime_cols=["event_dt"]).transform(df)
    rg = RankGaussScaler(n_quantiles=30, random_state=cfg.seed)
    scaled_num = rg.fit_transform(df[["num_1", "num_2"]].to_numpy())

    all_feats = pd.concat([te_oof, grp_feats, inter_feats, dt_feats], axis=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", te_oof["cat_a_te"].to_numpy())
        np.save(out_dir / f"test_{cfg.exp_name}.npy", te_test["cat_a_te"].to_numpy())
        pd.DataFrame({"id": range(10), "target": te_test["cat_a_te"].to_numpy()}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] Engineered features shape={all_feats.shape}, "
        f"RankGauss mean={float(np.mean(scaled_num)):.3f}, std={float(np.std(scaled_num)):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Feature Engineering Modules")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
