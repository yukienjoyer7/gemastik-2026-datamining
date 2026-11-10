#!/usr/bin/env python3
"""
Task 03 — Clustering & Topic Modeling: HDBSCAN `-1` Outlier Reassignment
========================================================================
Atomic standalone utility (`reassign_hdbscan_outliers`) that eliminates HDBSCAN/DBSCAN
`-1` noise labels by assigning every outlier to its nearest cluster via L2-normalized
centroid cosine similarity or distance-weighted $k$-NN, returning both contiguous
hard labels (`0..K-1`) and soft cluster membership probabilities.

Usage:
    python3 tasks/03_clustering_topic_modeling/modules/hdbscan_outlier_reassignment.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Tuple

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


@dataclass
class CFG:
    """Configuration for HDBSCAN Outlier Reassignment Smoke Test."""

    n_samples: int = 120
    n_clusters: int = 4
    embed_dim: int = 16
    seed: int = 42
    smoke_test: bool = False


def reassign_hdbscan_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "centroid_cosine",
    n_neighbors: int = 5,
    temperature: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reassigns all `-1` outlier points produced by HDBSCAN/DBSCAN so every sample receives
    a valid cluster ID in `0..K-1`.
    Returns `(clean_labels, soft_cluster_probs)`.
    """
    X = normalize(np.asarray(embeddings, dtype=np.float64), norm="l2")
    y = np.asarray(labels, dtype=np.int64).copy()

    valid_mask = y >= 0
    unique_clusters = np.sort(np.unique(y[valid_mask]))

    if len(unique_clusters) == 0:
        y[:] = 0
        return y, np.ones((len(y), 1), dtype=np.float32)

    remap = {int(c): i for i, c in enumerate(unique_clusters)}
    for idx in np.where(valid_mask)[0]:
        y[idx] = remap[int(y[idx])]

    k_clusters = len(unique_clusters)
    centroids = np.zeros((k_clusters, X.shape[1]), dtype=np.float64)
    for c in range(k_clusters):
        centroids[c] = X[y == c].mean(axis=0)
    centroids = normalize(centroids, norm="l2")

    cos_sim = X @ centroids.T  # (N, K)
    scaled_logits = cos_sim / max(temperature, 1e-4)
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    soft_probs = (
        exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    ).astype(np.float32)

    outlier_idx = np.where(~valid_mask)[0]
    if len(outlier_idx) == 0:
        return y, soft_probs

    if method == "knn" and valid_mask.sum() >= 2:
        k_eff = min(n_neighbors, int(valid_mask.sum()))
        knn = KNeighborsClassifier(n_neighbors=k_eff, weights="distance", metric="cosine")
        knn.fit(X[valid_mask], y[valid_mask])
        y[outlier_idx] = knn.predict(X[outlier_idx])
    else:
        y[outlier_idx] = np.argmax(cos_sim[outlier_idx], axis=1)

    return y, soft_probs


def run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    centers = rng.normal(0, 3.0, size=(cfg.n_clusters, cfg.embed_dim))
    true_labels = np.arange(cfg.n_samples) % cfg.n_clusters
    X = centers[true_labels] + rng.normal(0, 0.5, size=(cfg.n_samples, cfg.embed_dim))

    noisy_labels = true_labels.copy()
    noisy_labels[::5] = -1  # 20% noise outliers across all 4 clusters

    reassigned_cos, soft_probs = reassign_hdbscan_outliers(
        X, noisy_labels, method="centroid_cosine"
    )
    reassigned_knn, _ = reassign_hdbscan_outliers(X, noisy_labels, method="knn")

    assert (reassigned_cos >= 0).all() and (reassigned_knn >= 0).all()
    assert soft_probs.shape == (cfg.n_samples, cfg.n_clusters)
    acc = float((reassigned_cos == true_labels).mean())
    assert acc >= 0.90

    print("[SMOKE TEST] Task 03 HDBSCAN Outlier Reassignment (`hdbscan_outlier_reassignment.py`)")
    print(f"  Outliers reassigned : {int((noisy_labels == -1).sum())} -> 0 remaining")
    print(f"  Recovery accuracy   : {acc:.4f}, soft_probs={soft_probs.shape}")
    print("[SUCCESS] HDBSCAN outlier reassignment checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 HDBSCAN Outlier Reassignment")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
