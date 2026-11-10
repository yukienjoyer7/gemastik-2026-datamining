"""
Task 10 — Reply-Tree & Conversational Graph Topology Feature Extractor
======================================================================
Standalone feature engineering module `TreeTopologyFeatureExtractor`:
  - Computes structural & temporal features per node in a conversational thread:
      * `tree_depth` & `normalized_depth` (depth / max_thread_depth)
      * `is_root` & `is_leaf` binary flags
      * `direct_out_degree` (number of direct child replies)
      * `total_descendants` (total subtree size rooted at node `i`)
      * `sibling_count` & `sibling_rank` (chronological order among replies to the same parent)
      * `log_latency_from_parent` & `log_latency_from_root`
      * `parent_child_jaccard_overlap` & `root_child_jaccard_overlap` (lexical shift)

Usage:
  python3 tree_topology_features.py --smoke-test
"""

from __future__ import annotations

import argparse
import re
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

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


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class CFG:
    exp_name: str = "tree_topology_features"
    id_col: str = "id"
    parent_col: str = "parent_id"
    text_col: str = "text"
    timestamp_col: str = "timestamp"


def _token_set(text: str) -> Set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


class TreeTopologyFeatureExtractor:
    """Extracts structural, temporal, and lexical-shift features from reply trees."""

    def __init__(
        self,
        id_col: str = "id",
        parent_col: str = "parent_id",
        text_col: str = "text",
        timestamp_col: str = "timestamp",
    ) -> None:
        self.id_col = id_col
        self.parent_col = parent_col
        self.text_col = text_col
        self.timestamp_col = timestamp_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        ids = df[self.id_col].astype(str).tolist()
        parents = (
            df[self.parent_col].fillna("").astype(str).tolist()
            if self.parent_col in df.columns
            else [""] * n
        )
        texts = (
            df[self.text_col].fillna("").astype(str).tolist()
            if self.text_col in df.columns
            else [""] * n
        )
        ts = (
            df[self.timestamp_col].fillna(0).to_numpy(dtype=np.float64)
            if self.timestamp_col in df.columns
            else np.arange(n, dtype=np.float64)
        )

        id_to_idx = {nid: i for i, nid in enumerate(ids)}
        parent_idx: Dict[int, int] = {}
        children_idx: Dict[int, List[int]] = defaultdict(list)

        for i, pid in enumerate(parents):
            if pid in id_to_idx and id_to_idx[pid] != i:
                p_i = id_to_idx[pid]
                parent_idx[i] = p_i
                children_idx[p_i].append(i)

        # Sort each parent's children by timestamp
        sibling_rank = np.zeros(n, dtype=np.float32)
        sibling_count = np.zeros(n, dtype=np.float32)
        for p_i, ch_list in children_idx.items():
            ch_sorted = sorted(ch_list, key=lambda idx: ts[idx])
            children_idx[p_i] = ch_sorted
            for rank_pos, c_i in enumerate(ch_sorted):
                sibling_rank[c_i] = float(rank_pos)
                sibling_count[c_i] = float(len(ch_sorted))

        depths = np.zeros(n, dtype=np.float32)
        root_idx = np.arange(n, dtype=np.int32)
        for i in range(n):
            curr = i
            d = 0
            seen = {curr}
            while curr in parent_idx:
                nxt = parent_idx[curr]
                if nxt in seen:
                    break
                seen.add(nxt)
                curr = nxt
                d += 1
            depths[i] = float(d)
            root_idx[i] = curr

        desc_memo: Dict[int, int] = {}

        def _desc(u: int) -> int:
            if u in desc_memo:
                return desc_memo[u]
            res = sum(1 + _desc(c) for c in children_idx.get(u, []))
            desc_memo[u] = res
            return res

        total_desc = np.array([_desc(i) for i in range(n)], dtype=np.float32)
        out_deg = np.array([len(children_idx.get(i, [])) for i in range(n)], dtype=np.float32)

        thread_max_depth: Dict[int, float] = defaultdict(float)
        thread_size: Dict[int, int] = defaultdict(int)
        for i in range(n):
            r = int(root_idx[i])
            thread_size[r] += 1
            if depths[i] > thread_max_depth[r]:
                thread_max_depth[r] = float(depths[i])

        tok_sets = [_token_set(t) for t in texts]

        rows = []
        for i in range(n):
            r_i = int(root_idx[i])
            p_i = parent_idx.get(i, i)
            max_d = max(thread_max_depth[r_i], 1.0)
            dt_par = max(0.0, float(ts[i] - ts[p_i])) if i in parent_idx else 0.0
            dt_root = max(0.0, float(ts[i] - ts[r_i]))

            rows.append(
                {
                    "tree_depth": float(depths[i]),
                    "normalized_depth": float(depths[i] / max_d),
                    "is_root": 1.0 if i not in parent_idx else 0.0,
                    "is_leaf": 1.0 if out_deg[i] == 0 else 0.0,
                    "direct_out_degree": float(out_deg[i]),
                    "total_descendants": float(total_desc[i]),
                    "sibling_count": float(sibling_count[i]),
                    "sibling_rank": float(sibling_rank[i]),
                    "thread_size": float(thread_size[r_i]),
                    "log_latency_from_parent": float(np.log1p(dt_par)),
                    "log_latency_from_root": float(np.log1p(dt_root)),
                    "parent_child_jaccard": _jaccard(tok_sets[i], tok_sets[p_i]),
                    "root_child_jaccard": _jaccard(tok_sets[i], tok_sets[r_i]),
                }
            )

        return pd.DataFrame(rows, index=df.index)


def _run_smoke_test(cfg: CFG) -> None:
    df = pd.DataFrame(
        {
            "id": ["N0", "N1", "N2", "N3", "N4"],
            "parent_id": ["", "N0", "N0", "N1", "N3"],
            "timestamp": [1000, 1060, 1120, 1200, 1500],
            "text": [
                "kebijakan subsidi energi baru pemerintah pusat",
                "kebijakan subsidi ini sangat membantu masyarakat",
                "apakah subsidi energi berlaku untuk semua daerah",
                "benar sangat membantu masyarakat kecil di daerah",
                "semoga bantuan subsidi tepat sasaran",
            ],
        }
    )

    ext = TreeTopologyFeatureExtractor()
    feats = ext.transform(df)
    assert feats.loc[4, "tree_depth"] == 3.0
    assert feats.loc[0, "total_descendants"] == 4.0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = feats.to_numpy(dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr)
        feats.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] TreeTopologyFeatureExtractor shape={feats.shape} | "
        f"root_descendants={feats.loc[0, 'total_descendants']:.0f}, max_depth={feats['tree_depth'].max():.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10 Tree Topology Features Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
