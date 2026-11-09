#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Bi-Encoder Interaction, Losses & Graph Features
===============================================================================
Standalone reusable components for semantic similarity and pair classification:
  1. `BiEncoderInteractionHead`: Classical Sentence-BERT / InferSent interaction layer
     (`concat([u, v, |u - v|, u * v, cos(u, v)])` + LayerNorm + MLP).
  2. `CoSENTLoss`: Cosine Sentence Ranking Loss (Su, 2022) that directly optimizes
     Spearman rank correlation on cosine similarities of continuous or binary pairs.
  3. `PearsonCorrelationLoss`: Differentiable `1 - r(y_pred, y_true)` loss used by every
     top solution in Kaggle U.S. Patent Phrase to Phrase Matching.
  4. `PairGraphFeatureExtractor`: Computes "Magic" pair-graph features on the combined
     `(train + test)` undirected graph:
       - `deg(A)`, `deg(B)`, `max_deg`, `min_deg`, `deg_ratio`
       - Common neighbors count `|N(A) ∩ N(B)|` & Jaccard neighbor overlap
       - Adamic-Adar index `sum_{w in N(A) ∩ N(B)} 1 / log(deg(w))`
       - Connected component size `|CC(A)|`
       - Power-iteration PageRank scores `PR(A)`, `PR(B)`, and Personalized PageRank affinity.

Usage:
    python modules/pair_features_and_biencoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import site
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    """Configuration for testing Bi-Encoder interaction, losses, and pair graph features."""

    batch_size: int = 16
    embed_dim: int = 64
    hidden_dim: int = 64
    out_dim: int = 1
    cosent_scale: float = 20.0
    pagerank_damping: float = 0.85
    pagerank_iters: int = 20
    smoke_test: bool = False


# =============================================================================
# 1. Bi-Encoder Interaction Head & Similarity Losses
# =============================================================================


class BiEncoderInteractionHead(nn.Module):
    """
    Combines two dense sentence embeddings `u` and `v` via:
      `[u, v, |u - v|, u * v]`
    followed by LayerNorm, Dropout, and a 2-layer GELU MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = embed_dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(u - v)
        prod = u * v
        features = torch.cat([u, v, diff, prod], dim=-1)
        return self.net(features)


class PearsonCorrelationLoss(nn.Module):
    """
    Differentiable `1 - PearsonR(preds, targets)` loss.
    Invariant to shift and scale; standard primary loss for Pearson correlation metrics.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = preds.view(-1).float()
        y = targets.view(-1).float()
        vx = x - x.mean()
        vy = y - y.mean()
        denom = torch.sqrt((vx.pow(2).sum() + self.eps) * (vy.pow(2).sum() + self.eps))
        corr = (vx * vy).sum() / denom
        return 1.0 - corr.clamp(-1.0, 1.0)


class CoSENTLoss(nn.Module):
    """
    CoSENT (Cosine Sentence) Ranking Loss:
      `log(1 + sum_{y_i > y_j} exp(scale * (cos(u_j, v_j) - cos(u_i, v_i))))`
    Directly optimizes pairwise ranking order of cosine similarities for any continuous
    or binary ground-truth labels `y`.
    """

    def __init__(self, scale: float = 20.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self, u: torch.Tensor, v: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        u_norm = F.normalize(u, p=2, dim=-1)
        v_norm = F.normalize(v, p=2, dim=-1)
        cos_sim = (u_norm * v_norm).sum(dim=-1) * self.scale  # (B,)

        y = targets.view(-1).float()
        # Pairwise differences: cos_diff[i, j] = cos_sim[j] - cos_sim[i]
        cos_diff = cos_sim.unsqueeze(0) - cos_sim.unsqueeze(1)  # (B, B)
        # Mask pairs where y[i] > y[j]
        order_mask = (y.unsqueeze(1) - y.unsqueeze(0)) > 1e-6  # (B, B)

        if not order_mask.any():
            return cos_sim.sum() * 0.0

        valid_diffs = cos_diff[order_mask]
        # log(1 + sum(exp(valid_diffs))) computed stably via logsumexp with a 0 anchor
        zero_anchor = torch.zeros(1, device=valid_diffs.device, dtype=valid_diffs.dtype)
        return torch.logsumexp(torch.cat([zero_anchor, valid_diffs], dim=0), dim=0)


# =============================================================================
# 2. Magic Pair-Graph Feature Extractor
# =============================================================================


class PairGraphFeatureExtractor:
    """
    Extracts topological graph features from the undirected graph of all pairs `(text_a, text_b)`.
    Notice that in transductive pair competitions (e.g., Quora Question Pairs, Shopee),
    building the unlabeled adjacency graph across `concat([train, test])` is completely
    label-free and yields +0.03 to +0.08 leaderboard gain.
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
        for a, b in zip(texts_a, texts_b):
            u, v = node_to_id[str(a)], node_to_id[str(b)]
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)

        degrees = np.array([len(nb) for nb in neighbors], dtype=np.float64)

        # Connected components via BFS
        visited = np.zeros(n_nodes, dtype=bool)
        comp_size_by_node = np.ones(n_nodes, dtype=np.float64)
        for start in range(n_nodes):
            if visited[start]:
                continue
            queue = [start]
            visited[start] = True
            comp_nodes = [start]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                for nxt in neighbors[curr]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        queue.append(nxt)
                        comp_nodes.append(nxt)
            c_size = float(len(comp_nodes))
            for node in comp_nodes:
                comp_size_by_node[node] = c_size

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
        self.comp_size_by_node = comp_size_by_node
        self.pagerank = pr * n_nodes  # normalized so mean PageRank == 1.0
        return self

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

            # 1-hop + 2-hop Personalized PageRank transition affinity between u and v
            direct_edge = 1.0 if v in nb_u else 0.0
            two_hop_mass = sum(
                1.0 / max(float(self.degrees[w]), 1.0) for w in common
            )
            ppr_affinity = (
                self.damping * direct_edge / max(deg_u, 1.0)
                + (self.damping**2) * two_hop_mass / max(deg_u, 1.0)
            )

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


# =============================================================================
# 3. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    B, D = cfg.batch_size, cfg.embed_dim
    u = torch.randn(B, D, requires_grad=True)
    v = torch.randn(B, D, requires_grad=True)
    y = torch.rand(B)

    head = BiEncoderInteractionHead(embed_dim=D, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim)
    logits = head(u, v)
    assert logits.shape == (B, cfg.out_dim)

    pearson_loss_fn = PearsonCorrelationLoss()
    cosent_loss_fn = CoSENTLoss(scale=cfg.cosent_scale)

    p_loss = pearson_loss_fn(logits.squeeze(-1), y)
    c_loss = cosent_loss_fn(u, v, y)
    total_loss = p_loss + c_loss
    total_loss.backward()

    # Test PairGraphFeatureExtractor on a triangle + star graph
    texts_a = ["Q1", "Q2", "Q1", "Q3", "Q4", "Q5"]
    texts_b = ["Q2", "Q3", "Q3", "Q4", "Q5", "Q1"]
    extractor = PairGraphFeatureExtractor().fit(texts_a, texts_b)
    g_feats = extractor.transform(texts_a, texts_b)
    assert g_feats.shape == (len(texts_a), len(PairGraphFeatureExtractor.FEATURE_NAMES))
    # (Q1, Q2) share Q3 as common neighbor -> shared_neighbors >= 1
    assert g_feats[0, 6] >= 1.0, "Expected common neighbor Q3 for pair (Q1, Q2)!"

    print("[SMOKE TEST] Task 02 Bi-Encoder & Pair Graph Modules")
    print(f"  BiEncoderInteractionHead shape : {tuple(logits.shape)}")
    print(f"  PearsonCorrelationLoss         : {float(p_loss.item()):.4f}")
    print(f"  CoSENTLoss                     : {float(c_loss.item()):.4f}")
    print(f"  PairGraphFeatureExtractor      : {g_feats.shape} ({len(extractor.FEATURE_NAMES)} features)")
    print("[SUCCESS] All Task 02 module checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Bi-Encoder & Pair Graph Modules")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
