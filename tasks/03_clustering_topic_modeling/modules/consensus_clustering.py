#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Hungarian Alignment & Consensus Clustering
=================================================================================
Atomic standalone ensemble clusterer (`ConsensusClusterer` & `align_cluster_labels_hungarian`)
that combines multiple clustering runs across different seeds, embeddings, or UMAP
projections using:
  1. **Hungarian Label Alignment (`scipy.optimize.linear_sum_assignment`)** to the medoid
     partition with highest average NMI.
  2. **Co-Association Matrix Hierarchical Clustering** on $D_{ij} = 1 - C_{ij}$ where
     $C_{ij} = \frac{1}{M}\sum_{m=1}^M \mathbb{I}[c_i^{(m)} == c_j^{(m)}]$.

Usage:
    python3 tasks/03_clustering_topic_modeling/modules/consensus_clustering.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple

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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class CFG:
    """Configuration for ConsensusClusterer Smoke Test."""

    n_samples: int = 120
    n_clusters: int = 4
    embed_dim: int = 16
    n_runs: int = 5
    seed: int = 42
    smoke_test: bool = False


def align_cluster_labels_hungarian(
    reference_labels: np.ndarray, target_labels: np.ndarray
) -> np.ndarray:
    """
    Permutes `target_labels` to maximize overlap with `reference_labels` using the
    Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
    """
    ref = np.asarray(reference_labels, dtype=np.int64)
    tgt = np.asarray(target_labels, dtype=np.int64)
    k = int(max(ref.max(), tgt.max()) + 1)

    cost_matrix = np.zeros((k, k), dtype=np.int64)
    for r, t in zip(ref, tgt):
        if 0 <= r < k and 0 <= t < k:
            cost_matrix[t, r] -= 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(t), int(t)) for t in tgt], dtype=np.int64)


class ConsensusClusterer:
    """
    Ensembles multiple clustering partitions into a single high-stability partition
    via Hungarian alignment (`mode="hungarian"`) or Co-Association hierarchical
    clustering (`mode="coassociation"`).
    """

    def __init__(self, n_clusters: int, mode: str = "coassociation") -> None:
        self.n_clusters = n_clusters
        self.mode = mode
        self.labels_: np.ndarray = np.empty(0, dtype=np.int64)
        self.probs_: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def fit_predict(self, partitions: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        part_list = [np.asarray(p, dtype=np.int64).ravel() for p in partitions]
        m_runs = len(part_list)
        n_samples = len(part_list[0])

        avg_nmis = np.zeros(m_runs, dtype=np.float64)
        for i in range(m_runs):
            for j in range(m_runs):
                if i != j:
                    avg_nmis[i] += normalized_mutual_info_score(part_list[i], part_list[j])
        anchor_idx = int(np.argmax(avg_nmis))
        anchor = part_list[anchor_idx]

        aligned = [
            align_cluster_labels_hungarian(anchor, p) if idx != anchor_idx else anchor
            for idx, p in enumerate(part_list)
        ]

        vote_counts = np.zeros((n_samples, self.n_clusters), dtype=np.float32)
        for p in aligned:
            for i, cid in enumerate(p):
                if 0 <= cid < self.n_clusters:
                    vote_counts[i, cid] += 1.0
        probs = vote_counts / max(float(m_runs), 1.0)

        if self.mode == "coassociation" and n_samples <= 2500:
            coassoc = np.zeros((n_samples, n_samples), dtype=np.float32)
            for p in part_list:
                coassoc += (p[:, None] == p[None, :]).astype(np.float32)
            coassoc /= max(float(m_runs), 1.0)
            dist = np.clip(1.0 - coassoc, 0.0, 1.0)
            agg = AgglomerativeClustering(
                n_clusters=self.n_clusters, metric="precomputed", linkage="average"
            )
            raw_labels = agg.fit_predict(dist)
            labels = align_cluster_labels_hungarian(np.argmax(probs, axis=1), raw_labels)
        else:
            labels = np.argmax(probs, axis=1).astype(np.int64)

        self.labels_ = labels
        self.probs_ = probs
        return labels, probs


def run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    centers = rng.normal(0, 3.0, size=(cfg.n_clusters, cfg.embed_dim))
    true_labels = np.arange(cfg.n_samples) % cfg.n_clusters
    X = centers[true_labels] + rng.normal(0, 0.6, size=(cfg.n_samples, cfg.embed_dim))

    partitions = [
        KMeans(n_clusters=cfg.n_clusters, n_init=1, random_state=cfg.seed + s).fit_predict(X)
        for s in range(cfg.n_runs)
    ]
    ens = ConsensusClusterer(n_clusters=cfg.n_clusters, mode="coassociation")
    labels, probs = ens.fit_predict(partitions)
    ari = float(adjusted_rand_score(true_labels, labels))
    assert labels.shape == (cfg.n_samples,)
    assert probs.shape == (cfg.n_samples, cfg.n_clusters)
    assert ari >= 0.85

    print("[SMOKE TEST] Task 03 ConsensusClusterer (`consensus_clustering.py`)")
    print(f"  Ensemble ARI across {cfg.n_runs} runs : {ari:.4f}")
    print(f"  Output shapes             : labels={labels.shape}, probs={probs.shape}")
    print("[SUCCESS] ConsensusClusterer checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 ConsensusClusterer")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
