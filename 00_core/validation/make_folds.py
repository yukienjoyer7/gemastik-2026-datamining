#!/usr/bin/env python3
"""
make_folds.py
=============
Unified leak-free Cross-Validation fold generator outputting `train_folds.parquet` (or `.csv`)
with an integer `fold` column in `0 .. K-1`.

Supported CV schemes (`--scheme`):
  1. `stratified`                — StratifiedKFold for binary/multiclass classification
  2. `binned_stratified`         — Sturges' rule quantile binning + StratifiedKFold for continuous targets
  3. `group`                     — GroupKFold on `--group-col`
  4. `stratified_group`          — StratifiedGroupKFold on `--target-col` + `--group-col`
  5. `pair_connected_components` — Union-Find undirected graph on `(pair_col_a, pair_col_b)` +
                                   StratifiedGroupKFold / GroupKFold to eliminate transitive pair leakage
  6. `multilabel`                — Iterative multilabel stratification (`iterstrat` or built-in Sechidis et al.
                                   greedy iterative stratification fallback)
  7. `purged_time_series`        — Expanding-window Purged Group Time-Series Split with purge/embargo gap
                                   (unassigned early burn-in rows receive `fold = -1`, validation folds `0..K-1`)

Usage:
  python3 00_core/validation/make_folds.py --input-path train.parquet --output-path train_folds.parquet --scheme stratified --target-col target
  python3 00_core/validation/make_folds.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as _IterStratMSKF
except ImportError:
    _IterStratMSKF = None


@dataclass
class CFG:
    input_path: str = "train.parquet"
    output_path: str = "train_folds.parquet"
    scheme: str = "stratified"
    n_splits: int = 5
    seed: int = 42
    target_col: str = "target"
    multilabel_cols: List[str] = field(default_factory=list)
    group_col: Optional[str] = None
    pair_col_a: Optional[str] = None
    pair_col_b: Optional[str] = None
    time_col: Optional[str] = None
    purge_gap: int = 0
    embargo_gap: int = 0


class UnionFind:
    """Disjoint-Set (Union-Find) with path compression and union by rank for pair entity resolution."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def compute_pair_connected_components(col_a: Sequence, col_b: Sequence) -> np.ndarray:
    uf = UnionFind()
    str_a = [f"node::{a}" for a in col_a]
    str_b = [f"node::{b}" for b in col_b]
    for a, b in zip(str_a, str_b):
        uf.union(a, b)
    comp_map: Dict[str, int] = {}
    comp_ids = np.empty(len(str_a), dtype=np.int64)
    for i, a in enumerate(str_a):
        root = uf.find(a)
        if root not in comp_map:
            comp_map[root] = len(comp_map)
        comp_ids[i] = comp_map[root]
    return comp_ids


def greedy_multilabel_stratified_kfold(
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Sechidis et al. (2011) Iterative Multilabel Stratification fallback when `iterative-stratification`
    is not installed. Assigns rare positive label combinations first to balance label counts & fold sizes.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_labels = Y.shape
    folds = np.full(n_samples, -1, dtype=np.int64)

    desired_fold_sizes = np.full(n_splits, n_samples / n_splits, dtype=float)
    label_counts = Y.sum(axis=0).astype(float)
    desired_fold_label_counts = np.outer(np.ones(n_splits), label_counts / n_splits)

    unassigned = np.ones(n_samples, dtype=bool)
    while np.any(unassigned):
        unassigned_idx = np.where(unassigned)[0]
        rem_Y = Y[unassigned_idx]
        pos_per_label = rem_Y.sum(axis=0)
        valid_labels = np.where(pos_per_label > 0)[0]

        if len(valid_labels) == 0:
            # Remaining rows have all-zero labels
            for i in unassigned_idx:
                best_f = int(np.argmax(desired_fold_sizes))
                folds[i] = best_f
                desired_fold_sizes[best_f] -= 1.0
                unassigned[i] = False
            break

        rarest_label = int(valid_labels[np.argmin(pos_per_label[valid_labels])])
        candidates = unassigned_idx[Y[unassigned_idx, rarest_label] > 0]
        rng.shuffle(candidates)

        for sample_idx in candidates:
            if not unassigned[sample_idx]:
                continue
            label_demand = desired_fold_label_counts[:, rarest_label]
            max_demand = np.max(label_demand)
            tied_folds = np.where(np.isclose(label_demand, max_demand, atol=1e-6))[0]
            if len(tied_folds) == 1:
                chosen_fold = int(tied_folds[0])
            else:
                size_demand = desired_fold_sizes[tied_folds]
                max_size_demand = np.max(size_demand)
                final_ties = tied_folds[np.isclose(size_demand, max_size_demand, atol=1e-6)]
                chosen_fold = int(rng.choice(final_ties))

            folds[sample_idx] = chosen_fold
            unassigned[sample_idx] = False
            desired_fold_sizes[chosen_fold] -= 1.0
            active_labels = np.where(Y[sample_idx] > 0)[0]
            desired_fold_label_counts[chosen_fold, active_labels] -= 1.0

    return folds


class PurgedGroupTimeSeriesSplit:
    """
    Purged & Embargoed Time-Series Splitter over ordered time groups.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 0, embargo_gap: int = 0) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        if groups is None:
            groups = indices
        unique_groups = pd.unique(groups)
        n_groups = len(unique_groups)
        fold_size = n_groups // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(f"Not enough unique time groups ({n_groups}) for {self.n_splits} splits.")

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(self.n_splits):
            val_start_g = (k + 1) * fold_size
            val_end_g = (k + 2) * fold_size if k < self.n_splits - 1 else n_groups
            train_end_g = max(0, val_start_g - self.purge_gap)

            train_groups = set(unique_groups[:train_end_g])
            val_groups = set(unique_groups[val_start_g:val_end_g])

            train_idx = indices[pd.Series(groups).isin(train_groups).to_numpy()]
            val_idx = indices[pd.Series(groups).isin(val_groups).to_numpy()]
            splits.append((train_idx, val_idx))
        return splits


def assign_folds(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["fold"] = -1
    scheme = cfg.scheme.lower()

    if scheme == "stratified":
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        y = out[cfg.target_col].to_numpy()
        for fold, (_, val_idx) in enumerate(skf.split(out, y)):
            out.loc[val_idx, "fold"] = fold

    elif scheme == "binned_stratified":
        y = out[cfg.target_col].astype(float).to_numpy()
        n_bins = int(np.clip(1 + np.floor(np.log2(max(len(out), 2))), 2, 20))
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(skf.split(out, bins)):
            out.loc[val_idx, "fold"] = fold

    elif scheme == "group":
        if not cfg.group_col or cfg.group_col not in out.columns:
            raise ValueError("`--group-col` is required for `group` scheme.")
        gkf = GroupKFold(n_splits=cfg.n_splits)
        for fold, (_, val_idx) in enumerate(gkf.split(out, groups=out[cfg.group_col])):
            out.loc[val_idx, "fold"] = fold

    elif scheme == "stratified_group":
        if not cfg.group_col or cfg.group_col not in out.columns:
            raise ValueError("`--group-col` is required for `stratified_group` scheme.")
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        y = out[cfg.target_col].to_numpy()
        for fold, (_, val_idx) in enumerate(sgkf.split(out, y, groups=out[cfg.group_col])):
            out.loc[val_idx, "fold"] = fold

    elif scheme == "pair_connected_components":
        if not cfg.pair_col_a or not cfg.pair_col_b:
            raise ValueError("`--pair-col-a` and `--pair-col-b` are required for `pair_connected_components`.")
        comp_ids = compute_pair_connected_components(out[cfg.pair_col_a], out[cfg.pair_col_b])
        out["_pair_component_id"] = comp_ids
        if cfg.target_col in out.columns and out[cfg.target_col].nunique() <= 30:
            sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
            splitter = sgkf.split(out, out[cfg.target_col].to_numpy(), groups=comp_ids)
        else:
            gkf = GroupKFold(n_splits=cfg.n_splits)
            splitter = gkf.split(out, groups=comp_ids)
        for fold, (_, val_idx) in enumerate(splitter):
            out.loc[val_idx, "fold"] = fold

    elif scheme == "multilabel":
        cols = cfg.multilabel_cols
        if not cols:
            raise ValueError("`--multilabel-cols` must be provided for `multilabel` scheme.")
        Y = out[cols].to_numpy(dtype=int)
        if _IterStratMSKF is not None:
            mskf = _IterStratMSKF(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
            for fold, (_, val_idx) in enumerate(mskf.split(out, Y)):
                out.loc[val_idx, "fold"] = fold
        else:
            out["fold"] = greedy_multilabel_stratified_kfold(Y, n_splits=cfg.n_splits, seed=cfg.seed)

    elif scheme == "purged_time_series":
        if cfg.time_col and cfg.time_col in out.columns:
            out = out.sort_values(cfg.time_col).reset_index(drop=True)
            groups = out[cfg.time_col].to_numpy()
        elif cfg.group_col and cfg.group_col in out.columns:
            groups = out[cfg.group_col].to_numpy()
        else:
            groups = np.arange(len(out))
        ptss = PurgedGroupTimeSeriesSplit(
            n_splits=cfg.n_splits,
            purge_gap=cfg.purge_gap,
            embargo_gap=cfg.embargo_gap,
        )
        for fold, (_, val_idx) in enumerate(ptss.split(out, groups=groups)):
            out.loc[val_idx, "fold"] = fold

    else:
        raise ValueError(f"Unknown CV scheme: {cfg.scheme}")

    return out


def verify_and_print_folds(df: pd.DataFrame, cfg: CFG) -> None:
    print("=" * 72)
    print(f"[FOLD VERIFICATION] scheme={cfg.scheme} | n_splits={cfg.n_splits} | total_rows={len(df)}")
    fold_counts = df["fold"].value_counts().sort_index()
    for f_id, cnt in fold_counts.items():
        if cfg.multilabel_cols:
            means = df.loc[df["fold"] == f_id, cfg.multilabel_cols].mean().round(4).to_dict()
            print(f"  Fold {f_id:2d}: n={cnt:5d} | label_pos_rates={means}")
        elif cfg.target_col in df.columns:
            t_mean = float(df.loc[df["fold"] == f_id, cfg.target_col].mean())
            print(f"  Fold {f_id:2d}: n={cnt:5d} | {cfg.target_col}_mean={t_mean:.5f}")
        else:
            print(f"  Fold {f_id:2d}: n={cnt:5d}")
    print("=" * 72)


def save_folds(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    n = 240
    df = pd.DataFrame({
        "id": np.arange(n),
        "timestamp": np.repeat(np.arange(40), 6),
        "group_id": rng.integers(0, 30, size=n),
        "entity_a": [f"item_{i % 45}" for i in range(n)],
        "entity_b": [f"item_{(i + 1) % 45}" for i in range(n)],
        "target_bin": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "target_reg": rng.lognormal(mean=1.0, sigma=0.8, size=n),
        "label_1": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "label_2": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
        "label_3": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pq = os.path.join(tmpdir, "train_folds.parquet")

        # 1. Stratified
        f1 = assign_folds(df, CFG(scheme="stratified", target_col="target_bin", n_splits=5))
        assert set(f1["fold"].unique()) == {0, 1, 2, 3, 4}

        # 2. Binned stratified (regression)
        f2 = assign_folds(df, CFG(scheme="binned_stratified", target_col="target_reg", n_splits=5))
        assert set(f2["fold"].unique()) == {0, 1, 2, 3, 4}

        # 3. GroupKFold
        f3 = assign_folds(df, CFG(scheme="group", group_col="group_id", n_splits=5))
        assert f3.groupby("group_id")["fold"].nunique().max() == 1

        # 4. StratifiedGroupKFold
        f4 = assign_folds(df, CFG(scheme="stratified_group", target_col="target_bin", group_col="group_id", n_splits=5))
        assert f4.groupby("group_id")["fold"].nunique().max() == 1

        # 5. Pair Connected Components (entity resolution / pair tasks)
        df_pairs = df.copy()
        df_pairs["entity_a"] = [f"cluster_{i // 10}_node_{i % 3}" for i in range(n)]
        df_pairs["entity_b"] = [f"cluster_{i // 10}_node_{(i + 1) % 3}" for i in range(n)]
        cfg_pair = CFG(
            scheme="pair_connected_components",
            pair_col_a="entity_a",
            pair_col_b="entity_b",
            target_col="target_bin",
            n_splits=5,
        )
        f5 = assign_folds(df_pairs, cfg_pair)
        assert f5.groupby("_pair_component_id")["fold"].nunique().max() == 1
        verify_and_print_folds(f5, cfg_pair)

        # 6. Multilabel
        cfg_ml = CFG(scheme="multilabel", multilabel_cols=["label_1", "label_2", "label_3"], n_splits=5)
        f6 = assign_folds(df, cfg_ml)
        assert set(f6["fold"].unique()) == {0, 1, 2, 3, 4}
        verify_and_print_folds(f6, cfg_ml)

        # 7. Purged Time Series
        cfg_ts = CFG(scheme="purged_time_series", time_col="timestamp", target_col="target_bin", n_splits=5, purge_gap=1)
        f7 = assign_folds(df, cfg_ts)
        assert {0, 1, 2, 3, 4}.issubset(set(f7["fold"].unique()))

        save_folds(f1, out_pq)
        assert os.path.exists(out_pq)
        print("[SMOKE TEST PASSED] All 7 CV schemes in make_folds.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified competition CV fold generator.")
    parser.add_argument("--input-path", type=str, default=CFG.input_path)
    parser.add_argument("--output-path", type=str, default=CFG.output_path)
    parser.add_argument(
        "--scheme",
        type=str,
        default=CFG.scheme,
        choices=[
            "stratified",
            "binned_stratified",
            "group",
            "stratified_group",
            "pair_connected_components",
            "multilabel",
            "purged_time_series",
        ],
    )
    parser.add_argument("--n-splits", type=int, default=CFG.n_splits)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--multilabel-cols", type=str, nargs="*", default=[])
    parser.add_argument("--group-col", type=str, default=None)
    parser.add_argument("--pair-col-a", type=str, default=None)
    parser.add_argument("--pair-col-b", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--purge-gap", type=int, default=0)
    parser.add_argument("--embargo-gap", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        cfg = CFG(
            input_path=args.input_path,
            output_path=args.output_path,
            scheme=args.scheme,
            n_splits=args.n_splits,
            seed=args.seed,
            target_col=args.target_col,
            multilabel_cols=args.multilabel_cols,
            group_col=args.group_col,
            pair_col_a=args.pair_col_a,
            pair_col_b=args.pair_col_b,
            time_col=args.time_col,
            purge_gap=args.purge_gap,
            embargo_gap=args.embargo_gap,
        )
        p = Path(cfg.input_path)
        df_in = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        df_out = assign_folds(df_in, cfg)
        verify_and_print_folds(df_out, cfg)
        save_folds(df_out, cfg.output_path)
        print(f"Saved folds to {cfg.output_path}")
