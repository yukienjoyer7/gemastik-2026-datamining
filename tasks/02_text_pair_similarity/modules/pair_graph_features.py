#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Magic Pair-Graph Topological Feature Extractor
==============================================================================
Atomic standalone graph feature builder (`PairGraphFeatureExtractor`) for pair
classification & similarity competitions (Quora Question Pairs, Shopee, Entity Matching):
  - Union-Find connected components (`comp_id`, `comp_size`, `log_comp_size`)
  - Node degrees (`deg_a`, `deg_b`, `max_deg`, `min_deg`, `deg_diff`, `deg_ratio`)
  - Neighborhood overlap (`shared_neighbors`, `neighbor_jaccard`, `adamic_adar`)
  - Power-iteration PageRank (`pagerank_a`, `pagerank_b`, `pagerank_sum`)
  - 2-hop Personalized PageRank mutual transition affinity (`ppr_mutual_affinity`).

Usage:
    python3 tasks/02_text_pair_similarity/modules/pair_graph_features.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np


@dataclass
class CFG:
    """Configuration for PairGraphFeatureExtractor Smoke Test."""

    damping: float = 0.85
    n_iters: int = 20
    smoke_test: bool = False


class PairGraphFeatureExtractor:
    """
    Extracts 15 label-free topological graph features from the undirected graph of
    all pairs `(text_a, text_b)`.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "deg_a",
        "deg_b",
        "max_deg",
        "min_deg",
        "deg_diff",
        "deg_ratio",
        "shared_neighbors",
        "neighbor_jaccard",
        "adamic_adar",
        "comp_size",
        "log_comp_size",
        "pagerank_a",
        "pagerank_b",
        "pagerank_sum",
        "ppr_mutual_affinity",
    )

    def __init__(self, damping: float = 0.85, n_iters: int = 20) -> None:
        self.damping = damping
        self.n_iters = n_iters
        self.node_to_id: Dict[str, int] = {}
        self.neighbors: List[set] = []
        self.degrees: np.ndarray = np.empty(0, dtype=np.float64)
        self.comp_id_by_node: np.ndarray = np.empty(0, dtype=np.int64)
        self.comp_size_by_node: np.ndarray = np.empty(0, dtype=np.float64)
        self.pagerank: np.ndarray = np.empty(0, dtype=np.float64)

    def fit(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> "PairGraphFeatureExtractor":
        node_to_id: Dict[str, int] = {}
        for t in list(texts_a) + list(texts_b):
            s = str(t)
            if s not in node_to_id:
                node_to_id[s] = len(node_to_id)

        n_nodes = len(node_to_id)
        neighbors: List[set] = [set() for _ in range(n_nodes)]

        # Union-Find (Disjoint Set Union) with path compression and union by size
        parent = list(range(n_nodes))
        sz = [1] * n_nodes

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            if sz[ra] < sz[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            sz[ra] += sz[rb]

        for a, b in zip(texts_a, texts_b):
            u, v = node_to_id[str(a)], node_to_id[str(b)]
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
                _union(u, v)

        degrees = np.array([len(nb) for nb in neighbors], dtype=np.float64)
        comp_id_by_node = np.array([_find(i) for i in range(n_nodes)], dtype=np.int64)
        comp_size_by_node = np.array([float(sz[comp_id_by_node[i]]) for i in range(n_nodes)], dtype=np.float64)

        # Power-iteration PageRank
        pr = np.full(n_nodes, 1.0 / max(n_nodes, 1), dtype=np.float64)
        teleport = (1.0 - self.damping) / max(n_nodes, 1)
        for _ in range(self.n_iters):
            new_pr = np.full(n_nodes, teleport, dtype=np.float64)
            dangling_mass = pr[degrees == 0].sum() * self.damping / max(n_nodes, 1)
            new_pr += dangling_mass
            for u in range(n_nodes):
                deg_u = degrees[u]
                if deg_u > 0:
                    share = self.damping * pr[u] / deg_u
                    for v in neighbors[u]:
                        new_pr[v] += share
            pr = new_pr

        self.node_to_id = node_to_id
        self.neighbors = neighbors
        self.degrees = degrees
        self.comp_id_by_node = comp_id_by_node
        self.comp_size_by_node = comp_size_by_node
        self.pagerank = pr * n_nodes
        return self

    def get_pair_component_ids(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
        """Returns the connected-component ID for each pair (for leak-free GroupKFold splitting)."""
        out = np.zeros(len(texts_a), dtype=np.int64)
        for i, (a, b) in enumerate(zip(texts_a, texts_b)):
            u = self.node_to_id.get(str(a), -1)
            v = self.node_to_id.get(str(b), -1)
            if u >= 0:
                out[i] = self.comp_id_by_node[u]
            elif v >= 0:
                out[i] = self.comp_id_by_node[v]
            else:
                out[i] = -i - 1
        return out

    def transform(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
        n_rows = len(texts_a)
        feats = np.zeros((n_rows, len(self.FEATURE_NAMES)), dtype=np.float32)

        for i, (raw_a, raw_b) in enumerate(zip(texts_a, texts_b)):
            u = self.node_to_id.get(str(raw_a), -1)
            v = self.node_to_id.get(str(raw_b), -1)
            if u < 0 or v < 0:
                continue

            nb_u = self.neighbors[u]
            nb_v = self.neighbors[v]
            deg_u = float(self.degrees[u])
            deg_v = float(self.degrees[v])
            max_d = max(deg_u, deg_v)
            min_d = min(deg_u, deg_v)

            common = nb_u & nb_v
            shared_cnt = float(len(common))
            union_cnt = float(len(nb_u | nb_v))
            jaccard_nb = shared_cnt / max(union_cnt, 1.0)
            adamic_adar = sum(
                1.0 / math.log(max(float(self.degrees[w]), 2.0)) for w in common
            )

            c_size = float(max(self.comp_size_by_node[u], self.comp_size_by_node[v]))
            pr_u = float(self.pagerank[u])
            pr_v = float(self.pagerank[v])

            direct_edge = 1.0 if v in nb_u else 0.0
            two_hop_mass = sum(1.0 / max(float(self.degrees[w]), 1.0) for w in common)
            ppr_u_to_v = (
                self.damping * direct_edge / max(deg_u, 1.0)
                + (self.damping**2) * two_hop_mass / max(deg_u, 1.0)
            )
            ppr_v_to_u = (
                self.damping * direct_edge / max(deg_v, 1.0)
                + (self.damping**2) * two_hop_mass / max(deg_v, 1.0)
            )
            ppr_affinity = 0.5 * (ppr_u_to_v + ppr_v_to_u)

            feats[i, 0] = deg_u
            feats[i, 1] = deg_v
            feats[i, 2] = max_d
            feats[i, 3] = min_d
            feats[i, 4] = abs(deg_u - deg_v)
            feats[i, 5] = min_d / max(max_d, 1.0)
            feats[i, 6] = shared_cnt
            feats[i, 7] = jaccard_nb
            feats[i, 8] = adamic_adar
            feats[i, 9] = c_size
            feats[i, 10] = math.log1p(c_size)
            feats[i, 11] = pr_u
            feats[i, 12] = pr_v
            feats[i, 13] = pr_u + pr_v
            feats[i, 14] = ppr_affinity

        return feats


def run_smoke_test(cfg: CFG) -> None:
    texts_a = ["Q1", "Q2", "Q1", "Q3", "Q4", "Q5"]
    texts_b = ["Q2", "Q3", "Q3", "Q4", "Q5", "Q1"]
    extractor = PairGraphFeatureExtractor(damping=cfg.damping, n_iters=cfg.n_iters).fit(
        texts_a, texts_b
    )
    g_feats = extractor.transform(texts_a, texts_b)
    comp_ids = extractor.get_pair_component_ids(texts_a, texts_b)

    assert g_feats.shape == (len(texts_a), len(PairGraphFeatureExtractor.FEATURE_NAMES))
    assert g_feats[0, 6] >= 1.0  # (Q1, Q2) share Q3
    assert len(np.unique(comp_ids)) == 1  # all 5 nodes form 1 connected component

    print("[SMOKE TEST] Task 02 PairGraphFeatureExtractor (`pair_graph_features.py`)")
    print(f"  Graph features shape : {g_feats.shape}")
    print(f"  Row 0 (Q1, Q2)       : shared_neighbors={g_feats[0, 6]:.1f}, adamic_adar={g_feats[0, 8]:.3f}, ppr={g_feats[0, 14]:.3f}")
    print("[SUCCESS] PairGraphFeatureExtractor checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 PairGraphFeatureExtractor")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
