"""
Task 08 — Datetime Cyclical, Business-Day & Entity Time-Gap Lag Feature Extractor
=================================================================================
Standalone temporal feature transformer `DatetimeFeatureExtractor`:
  - Cyclical `sin/cos` encodings for hour-of-day, day-of-week, and month-of-year
  - Business-day / weekend binary flags and elapsed epoch seconds
  - Global and per-entity (`groupby(entity_col)`) inter-event time-gap lag deltas (`diff_prev_sec`, `diff_next_sec`)

Usage:
  python3 datetime_features.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


@dataclass
class CFG:
    exp_name: str = "datetime_features"
    seed: int = 42


class DatetimeFeatureExtractor:
    """
    Extracts cyclical calendar encodings (`sin`/`cos`), business-day/weekend flags,
    epoch timestamps, and global + per-entity inter-event time-gap lag features.
    """

    def __init__(
        self, datetime_cols: Sequence[str], entity_col: str | None = None
    ) -> None:
        self.datetime_cols = list(datetime_cols)
        self.entity_col = entity_col

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
            out[f"{col}_month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
            out[f"{col}_month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
            out[f"{col}_is_weekend"] = (dow >= 5).astype(np.float32)
            out[f"{col}_is_business_day"] = (dow < 5).astype(np.float32)

            epoch_sec = (dt.astype("int64") // 10**9).to_numpy(dtype=np.float64)
            out[f"{col}_diff_prev_sec"] = np.diff(
                epoch_sec, prepend=epoch_sec[0]
            ).astype(np.float32)

            if self.entity_col is not None and self.entity_col in df.columns:
                tmp = pd.DataFrame(
                    {"entity": df[self.entity_col].values, "epoch": epoch_sec},
                    index=df.index,
                )
                entity_prev = (
                    tmp.groupby("entity", sort=False)["epoch"].diff(1).fillna(0.0)
                )
                entity_next = (
                    -tmp.groupby("entity", sort=False)["epoch"].diff(-1).fillna(0.0)
                )
                out[f"{col}_entity_lag_prev_sec"] = entity_prev.to_numpy(
                    dtype=np.float32
                )
                out[f"{col}_entity_lag_next_sec"] = entity_next.to_numpy(
                    dtype=np.float32
                )

        return out


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 40
    df = pd.DataFrame(
        {
            "user_id": rng.choice(["U1", "U2", "U3"], size=n),
            "event_dt": pd.date_range("2025-03-01", periods=n, freq="6h").astype(str),
        }
    )

    dt_ext = DatetimeFeatureExtractor(
        datetime_cols=["event_dt"], entity_col="user_id"
    )
    dt_feats = dt_ext.transform(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = dt_feats.to_numpy(dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr[:10])
        dt_feats.head(10).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] DatetimeFeatureExtractor shape={dt_feats.shape} | "
        f"columns={list(dt_feats.columns)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Datetime Features Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
