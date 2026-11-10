"""
Task 10 — Leak-Free Out-Of-Fold Relational Neighbor Aggregator & Label Propagation
==================================================================================
Standalone module `RelationalNeighborAggregator` for 2nd-stage Graph Stacking:
  - Given 1st-stage Out-Of-Fold node probability predictions `oof_probs` `(N, C)`
    and relational links `(id, parent_id, user_id)`:
      1. `parent_prob`: 1st-stage OOF probability of node `i`'s parent
      2. `root_prob`: 1st-stage OOF probability of node `i`'s thread root
      3. `children_mean_prob`: mean 1st-stage OOF probability of direct replies to `i`
      4. `siblings_mean_prob`: leave-one-out mean 1st-stage OOF probability of sibling replies
      5. `same_user_loo_prob`: **strictly leave-one-out** mean 1st-stage OOF probability
         across other posts authored by `user_id` (`(sum_u - p_i) / (n_u - 1)`), preventing
         target self-leakage!

Usage:
  python3 graph_label_propagation.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    exp_name: str = "graph_label_propagation"
    seed: int = 42
    num_classes: int = 3


class RelationalNeighborAggregator:
    """
    Computes leak-free relational neighbor probability features from 1st-stage OOF
    or test probability matrices `(N, C)`.
    """

    def __init__(
        self,
        id_col: str = "id",
        parent_col: str = "parent_id",
        user_col: str = "user_id",
    ) -> None:
        self.id_col = id_col
        self.parent_col = parent_col
        self.user_col = user_col

    def transform(self, df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
        n_samples, n_classes = probs.shape
        ids = df[self.id_col].astype(str).tolist()
        id_to_idx = {nid: i for i, nid in enumerate(ids)}
        parents = (
            df[self.parent_col].fillna("").astype(str).tolist()
            if self.parent_col in df.columns
            else [""] * n_samples
        )
        users = (
            df[self.user_col].fillna("u0").astype(str).tolist()
            if self.user_col in df.columns
            else ["u0"] * n_samples
        )

        parent_idx: Dict[int, int] = {}
        children_idx: Dict[int, List[int]] = defaultdict(list)
        for i, pid in enumerate(parents):
            if pid in id_to_idx and id_to_idx[pid] != i:
                p_i = id_to_idx[pid]
                parent_idx[i] = p_i
                children_idx[p_i].append(i)

        root_idx = np.arange(n_samples, dtype=np.int32)
        for i in range(n_samples):
            curr = i
            seen = {curr}
            while curr in parent_idx:
                nxt = parent_idx[curr]
                if nxt in seen:
                    break
                seen.add(nxt)
                curr = nxt
            root_idx[i] = curr

        user_sum: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(n_classes, dtype=np.float64)
        )
        user_cnt: Dict[str, int] = defaultdict(int)
        for i, u in enumerate(users):
            user_sum[u] += probs[i]
            user_cnt[u] += 1

        global_prior = np.mean(probs, axis=0)
        cols: Dict[str, np.ndarray] = {}
        for c in range(n_classes):
            cols[f"self_p{c}"] = np.zeros(n_samples, dtype=np.float32)
            cols[f"parent_p{c}"] = np.zeros(n_samples, dtype=np.float32)
            cols[f"root_p{c}"] = np.zeros(n_samples, dtype=np.float32)
            cols[f"children_mean_p{c}"] = np.zeros(n_samples, dtype=np.float32)
            cols[f"siblings_loo_p{c}"] = np.zeros(n_samples, dtype=np.float32)
            cols[f"user_loo_p{c}"] = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            p_self = probs[i]
            p_par = probs[parent_idx[i]] if i in parent_idx else p_self
            p_root = probs[int(root_idx[i])]

            ch = children_idx.get(i, [])
            p_ch = np.mean(probs[ch], axis=0) if ch else p_self

            if i in parent_idx:
                sibs = [s for s in children_idx[parent_idx[i]] if s != i]
                p_sib = np.mean(probs[sibs], axis=0) if sibs else p_par
            else:
                p_sib = p_self

            u = users[i]
            if user_cnt[u] > 1:
                p_user = (user_sum[u] - p_self) / float(user_cnt[u] - 1)
            else:
                p_user = global_prior

            for c in range(n_classes):
                cols[f"self_p{c}"][i] = float(p_self[c])
                cols[f"parent_p{c}"][i] = float(p_par[c])
                cols[f"root_p{c}"][i] = float(p_root[c])
                cols[f"children_mean_p{c}"][i] = float(p_ch[c])
                cols[f"siblings_loo_p{c}"][i] = float(p_sib[c])
                cols[f"user_loo_p{c}"][i] = float(p_user[c])

        return pd.DataFrame(cols, index=df.index)


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    df = pd.DataFrame(
        {
            "id": ["N0", "N1", "N2", "N3", "N4", "N5"],
            "parent_id": ["", "N0", "N0", "N1", "N1", "N2"],
            "user_id": ["U1", "U2", "U1", "U2", "U3", "U1"],
        }
    )
    probs = rng.dirichlet(np.ones(cfg.num_classes), size=len(df)).astype(np.float32)

    agg = RelationalNeighborAggregator()
    feat_df = agg.transform(df, probs)

    # Verify leave-one-out user aggregation for N0 (user U1 has posts N0, N2, N5)
    expected_u1_for_n0 = 0.5 * (probs[2] + probs[5])
    actual_u1_for_n0 = np.array(
        [feat_df.loc[0, f"user_loo_p{c}"] for c in range(cfg.num_classes)]
    )
    assert np.allclose(expected_u1_for_n0, actual_u1_for_n0, atol=1e-5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = feat_df.to_numpy(dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr)
        feat_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] RelationalNeighborAggregator shape={feat_df.shape} | "
        f"Leave-one-out user_id leakage check=PASSED"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 10 Graph Label Propagation & Relational Aggregator Module"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
