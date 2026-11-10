"""
Task 08 — Group Aggregation Transformer (`mean`, `std`, `min`, `max`, `zscore`, `diff_from_mean`, `ratio_to_mean`)
==================================================================================================================
Standalone tabular feature transformer `GroupAggregationTransformer`:
  - Aggregates numerical columns grouped by categorical entities (`customer_id`, `merchant_id`, `category`)
  - Computes `mean`, `std`, `min`, `max`, `diff_from_mean`, `zscore`, and `ratio_to_mean`
    (1st-place AmEx Default Prediction & Home Credit Default Risk feature pattern).

Usage:
  python3 group_aggregations.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

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
    exp_name: str = "group_aggregations"
    seed: int = 42


class GroupAggregationTransformer:
    """
    Computes `groupby(group_col)[num_col]` statistics (`mean`, `std`, `min`, `max`)
    plus entity-relative `diff_from_mean`, `zscore`, and `ratio_to_mean`.
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
            out[f"{prefix}_diff_from_mean"] = raw_v - mean_v
            out[f"{prefix}_zscore"] = (raw_v - mean_v) / (std_v + 1e-5)
            out[f"{prefix}_ratio_to_mean"] = raw_v / (np.abs(mean_v) + 1e-5)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 50
    df = pd.DataFrame(
        {
            "entity_id": rng.choice(["E1", "E2", "E3", "E4"], size=n),
            "spend": rng.normal(250.0, 45.0, size=n),
            "balance": rng.exponential(100.0, size=n),
        }
    )

    grp = GroupAggregationTransformer(
        group_cols=["entity_id"], num_cols=["spend", "balance"]
    )
    feats = grp.fit_transform(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = feats.to_numpy(dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr[:10])
        feats.head(10).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] GroupAggregationTransformer output shape={feats.shape} | "
        f"columns={list(feats.columns[:4])}..."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Group Aggregations Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
