"""
Task 05 — Stage-2 GBDT Query-Contextualized Relative Features
=============================================================
Standalone feature builder `build_query_context_features` that enriches `(query_id, doc_id)`
candidate tables with query-relative statistics:
  - Per-query z-score (`<col>_q_zscore`)
  - Per-query min-max scaled score (`<col>_q_minmax`)
  - Within-query rank (`<col>_q_rank`) and reciprocal rank (`<col>_q_rr`)
  - Score gap to top-1 (`<col>_diff_top1`), top-3 mean (`<col>_diff_top3_mean`), and query mean (`<col>_diff_mean`)

Usage:
  python3 query_context_features.py --smoke-test
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
    exp_name: str = "query_context_features"
    seed: int = 42
    query_col: str = "query_id"


def build_query_context_features(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    query_col: str = "query_id",
) -> pd.DataFrame:
    """
    Enriches a query-candidate DataFrame with query-contextualized relative features:
      - `<col>_q_zscore`: within-query standardized score
      - `<col>_q_minmax`: within-query [0, 1] scaled score
      - `<col>_q_rank`: within-query descending rank (1 = highest)
      - `<col>_q_rr`: within-query reciprocal rank `1 / rank`
      - `<col>_diff_mean`: difference from query mean score
      - `<col>_diff_top1`: difference from best candidate of the same query
      - `<col>_diff_top3_mean`: difference from mean of top-3 candidates of the same query
    """
    out = df.copy()
    grouped = out.groupby(query_col, sort=False)

    for col in score_cols:
        q_mean = grouped[col].transform("mean")
        q_std = grouped[col].transform("std").fillna(0.0)
        q_min = grouped[col].transform("min")
        q_max = grouped[col].transform("max")

        out[f"{col}_q_zscore"] = (out[col] - q_mean) / (q_std + 1e-6)
        out[f"{col}_q_minmax"] = (out[col] - q_min) / (q_max - q_min + 1e-6)

        ranks = grouped[col].rank(ascending=False, method="min")
        out[f"{col}_q_rank"] = ranks
        out[f"{col}_q_rr"] = 1.0 / ranks
        out[f"{col}_diff_mean"] = out[col] - q_mean
        out[f"{col}_diff_top1"] = out[col] - q_max

        # Top-3 mean per query
        top3_means = (
            out[[query_col, col]]
            .sort_values([query_col, col], ascending=[True, False])
            .groupby(query_col, sort=False)
            .head(3)
            .groupby(query_col, sort=False)[col]
            .mean()
        )
        out[f"{col}_diff_top3_mean"] = out[col] - out[query_col].map(top3_means)

    return out


def _run_smoke_test(cfg: CFG) -> None:
    sample_df = pd.DataFrame(
        {
            "query_id": ["Q0", "Q0", "Q0", "Q0", "Q1", "Q1", "Q1", "Q1"],
            "doc_id": ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"],
            "bm25_score": [3.2, 2.5, 1.8, 0.4, 5.1, 4.9, 2.2, 1.0],
            "tfidf_cos": [0.75, 0.60, 0.42, 0.11, 0.88, 0.80, 0.45, 0.25],
        }
    )
    feat_df = build_query_context_features(
        sample_df, ["bm25_score", "tfidf_cos"], query_col=cfg.query_col
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        oof_arr = feat_df["bm25_score_q_zscore"].to_numpy(dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", oof_arr)
        feat_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Query context features shape={feat_df.shape} | "
        f"Added columns={feat_df.shape[1] - sample_df.shape[1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Query Context Features Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
