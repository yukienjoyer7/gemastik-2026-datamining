#!/usr/bin/env python3
"""
memory_reduction.py
===================
Safe Pandas DataFrame memory reduction utility (`reduce_mem_usage`) for large tabular joins.
Features:
  - Downcasts integer columns to the smallest safe integer type (`int8`, `int16`, `int32`, `int64`).
  - Downcasts float64 columns to `float32` (never `float16`, which lacks support in many GBDT/Parquet engines
    and causes catastrophic cancellation in rolling aggregations).
  - Never touches columns listed in `exclude_cols` (such as target, ID, or high-precision timestamps).
  - Optionally converts low-cardinality object/string columns (`n_unique / n_rows <= cat_ratio_threshold`)
    to `category` dtype.

Usage:
  python3 00_core/tracking_and_utils/memory_reduction.py --input-path train.parquet --output-path train_slim.parquet
  python3 00_core/tracking_and_utils/memory_reduction.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    input_path: str = "train.parquet"
    output_path: Optional[str] = None
    exclude_cols: List[str] = field(default_factory=lambda: ["id", "target", "fold"])
    convert_objects_to_category: bool = True
    cat_ratio_threshold: float = 0.50
    verbose: bool = True


def reduce_mem_usage(
    df: pd.DataFrame,
    exclude_cols: Optional[Sequence[str]] = None,
    convert_objects_to_category: bool = True,
    cat_ratio_threshold: float = 0.50,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely downcasts numeric and low-cardinality string/object columns in `df`.
    Returns `(downcast_df, report_dict)`.
    """
    out = df.copy()
    excluded = set(exclude_cols or [])
    start_mem = float(out.memory_usage(deep=True).sum() / (1024 ** 2))
    n_rows = max(len(out), 1)
    changes: List[Dict[str, str]] = []

    for col in out.columns:
        if col in excluded:
            continue
        s = out[col]
        orig_dtype = str(s.dtype)

        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            c_min, c_max = int(s.min()), int(s.max())
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                out[col] = s.astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                out[col] = s.astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                out[col] = s.astype(np.int32)
            else:
                out[col] = s.astype(np.int64)

        elif pd.api.types.is_float_dtype(s):
            c_min, c_max = float(s.min()), float(s.max())
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                out[col] = s.astype(np.float32)

        elif convert_objects_to_category and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            nunique = int(s.nunique(dropna=False))
            if (nunique / n_rows) <= cat_ratio_threshold:
                out[col] = s.astype("category")

        new_dtype = str(out[col].dtype)
        if new_dtype != orig_dtype:
            changes.append({"column": col, "from": orig_dtype, "to": new_dtype})

    end_mem = float(out.memory_usage(deep=True).sum() / (1024 ** 2))
    saved_pct = 100.0 * (start_mem - end_mem) / max(start_mem, 1e-9)

    report = {
        "start_mem_mb": round(start_mem, 4),
        "end_mem_mb": round(end_mem, 4),
        "saved_pct": round(saved_pct, 2),
        "columns_converted": len(changes),
        "changes": changes,
    }

    if verbose:
        print("=" * 72)
        print(
            f"[MEMORY REDUCTION] {start_mem:.3f} MB -> {end_mem:.3f} MB "
            f"({saved_pct:.1f}% reduction | {len(changes)} cols downcast)"
        )
        print("=" * 72)

    return out, report


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    df = pd.DataFrame({
        "id": np.arange(n, dtype=np.int64),
        "small_flag": rng.integers(0, 5, size=n, dtype=np.int64),
        "med_int": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "float_feat": rng.normal(0.0, 1.0, size=n).astype(np.float64),
        "city_cat": rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan"], size=n),
        "target": rng.uniform(0.0, 1.0, size=n).astype(np.float64),
    })

    slim_df, rep = reduce_mem_usage(df, exclude_cols=["id", "target"], verbose=True)
    assert slim_df["id"].dtype == np.int64, "Excluded id column should remain int64"
    assert slim_df["target"].dtype == np.float64, "Excluded target column should remain float64"
    assert slim_df["small_flag"].dtype == np.int8
    assert slim_df["med_int"].dtype == np.int16
    assert slim_df["float_feat"].dtype == np.float32
    assert isinstance(slim_df["city_cat"].dtype, pd.CategoricalDtype)
    assert rep["saved_pct"] > 50.0
    print("[SMOKE TEST PASSED] memory_reduction.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe Pandas DataFrame memory reduction.")
    parser.add_argument("--input-path", type=str, default=CFG.input_path)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--exclude-cols", type=str, nargs="*", default=["id", "target", "fold"])
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        p = Path(args.input_path)
        df_in = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        df_out, _ = reduce_mem_usage(df_in, exclude_cols=args.exclude_cols, verbose=True)
        if args.output_path:
            out_p = Path(args.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() in {".parquet", ".pq"}:
                df_out.to_parquet(out_p, index=False)
            else:
                df_out.to_csv(out_p, index=False)
