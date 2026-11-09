#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Outlier Reassignment, Consensus & c-TF-IDF
=================================================================================
Standalone reusable clustering utilities for competitive unsupervised/topic tasks:
  1. `reassign_hdbscan_outliers(embeddings, labels, method="centroid_cosine"|"knn")`:
     Eliminates HDBSCAN `-1` noise labels by assigning every outlier to its most
     similar cluster centroid (cosine similarity) or majority $k$-NN cluster vote.
  2. `ConsensusClusterer`:
     Ensembles multiple clustering runs across different random seeds, feature views,
     or algorithms using either:
       - **Hungarian label alignment** (`scipy.optimize.linear_sum_assignment`) + majority/soft voting
       - **Co-Association Matrix** ($C_{ij} = \text{fraction of runs where } i, j \text{ co-cluster}$)
         followed by Agglomerative Clustering on $1 - C$.
  3. `compute_c_tfidf_topics(texts, labels, top_k=10, ngram_range=(1, 2))`:
     BERTopic class-based TF-IDF ($w_{t,c} = \text{tf}_{t,c} \cdot \log(1 + \frac{A}{\text{tf}_t})$)
     with optional **Maximal Marginal Relevance (MMR)** keyword diversification.

Usage:
    python modules/clustering_tricks.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


@dataclass
class CFG:
    """Configuration for Task 03 Clustering Tricks Smoke Test."""

    n_samples: int = 120
    n_clusters: int = 4
    embed_dim: int = 16
    top_k_words: int = 8
    mmr_diversity: float = 0.35
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Mandatory HDBSCAN -1 Outlier Reassignment
# =============================================================================


def reassign_hdbscan_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "centroid_cosine",
    n_neighbors: int = 5,
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

    # Edge case: all points marked as -1 noise
    if len(unique_clusters) == 0:
        y[:] = 0
        return y, np.ones((len(y), 1), dtype=np.float32)

    # Remap existing positive cluster IDs to contiguous 0..K-1
    remap = {c: i for i, c in enumerate(unique_clusters)}
    for idx in np.where(valid_mask)[0]:
        y[idx] = remap[int(y[idx])]

    k_clusters = len(unique_clusters)
    centroids = np.zeros((k_clusters, X.shape[1]), dtype=np.float64)
    for c in range(k_clusters):
        c_pts = X[y == c]
        centroids[c] = c_pts.mean(axis=0)
    centroids = normalize(centroids, norm="l2")

    # Soft cosine similarity probabilities to all cluster centroids
    cos_sim = X @ centroids.T  # (N, K)
    scaled_logits = cos_sim * 10.0
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    soft_probs = (exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)).astype(
        np.float32
    )

    outlier_idx = np.where(~valid_mask)[0]
    if len(outlier_idx) == 0:
        return y, soft_probs

    if method == "knn" and valid_mask.sum() >= 2:
        k_eff = min(n_neighbors, int(valid_mask.sum()))
        knn = KNeighborsClassifier(n_neighbors=k_eff, weights="distance", metric="cosine")
        knn.fit(X[valid_mask], y[valid_mask])
        y[outlier_idx] = knn.predict(X[outlier_idx])
    else:
        # Default: assign to highest cosine similarity cluster centroid
        y[outlier_idx] = np.argmax(cos_sim[outlier_idx], axis=1)

    return y, soft_probs


# =============================================================================
# 2. Consensus & Co-Association Clustering Ensemble
# =============================================================================


def align_cluster_labels_hungarian(reference_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
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
            cost_matrix[t, r] -= 1  # negative overlap count for minimization

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(t), int(t)) for t in tgt], dtype=np.int64)


class ConsensusClusterer:
    r"""
    Ensembles multiple clustering partitions $P^{(1)}, \dots, P^{(M)}$ (from different
    seeds, UMAP projections, or embedding models) into a single high-stability partition.
    Supports:
      - `mode="hungarian"`: Aligns every partition to the medoid partition via Hungarian
        matching and outputs both hard majority-vote labels and soft membership probabilities.
      - `mode="coassociation"`: Builds the $N \times N$ co-association similarity matrix
        $C_{ij} = \frac{1}{M} \sum_m \mathbb{I}[c_i^{(m)} == c_j^{(m)}]$ and runs average-linkage
        Agglomerative Clustering on distance $D = 1 - C$.
    """

    def __init__(self, n_clusters: int, mode: str = "hungarian") -> None:
        self.n_clusters = n_clusters
        self.mode = mode
        self.labels_: np.ndarray = np.empty(0, dtype=np.int64)
        self.probs_: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def fit_predict(self, partitions: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        part_list = [np.asarray(p, dtype=np.int64).ravel() for p in partitions]
        m_runs = len(part_list)
        n_samples = len(part_list[0])

        # Select the partition with highest average NMI to all other runs as anchor
        avg_nmis = np.zeros(m_runs, dtype=np.float64)
        for i in range(m_runs):
            for j in range(m_runs):
                if i != j:
                    avg_nmis[i] += normalized_mutual_info_score(part_list[i], part_list[j])
        anchor_idx = int(np.argmax(avg_nmis))
        anchor = part_list[anchor_idx]

        # Align all partitions to anchor via Hungarian matching
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


# =============================================================================
# 3. Class-Based TF-IDF (c-TF-IDF) + MMR Topic Keyword Extraction
# =============================================================================


def compute_c_tfidf_topics(
    texts: Sequence[str],
    labels: Sequence[int],
    top_k: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    mmr_diversity: float = 0.3,
) -> Dict[int, List[Tuple[str, float]]]:
    r"""
    Computes BERTopic-style class-based TF-IDF (c-TF-IDF):
      1. Concatenates all documents within each cluster $c$ into a single meta-document.
      2. Computes term frequency $t_{w, c}$ normalized by total words in cluster $c$.
      3. Multiplies by inverse class frequency $\log(1 + \frac{A}{\sum_c \text{tf}_{w, c}})$,
         where $A$ is the average word count per cluster.
      4. Applies Maximal Marginal Relevance (MMR) over character 3-gram word representations
         to avoid redundant morphological variants (e.g., `"bank"`, `"banking"`, `"banks"`).
    """
    df_grouped: Dict[int, List[str]] = {}
    for text, lab in zip(texts, labels):
        cid = int(lab)
        if cid < 0:
            continue
        df_grouped.setdefault(cid, []).append(str(text))

    cluster_ids = sorted(df_grouped.keys())
    cluster_docs = [" ".join(df_grouped[c]) for c in cluster_ids]

    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=20000)
    X_counts = vec.fit_transform(cluster_docs).toarray().astype(np.float64)  # (C, V)
    vocab = np.array(vec.get_feature_names_out())

    words_per_cluster = np.maximum(X_counts.sum(axis=1, keepdims=True), 1.0)
    tf = X_counts / words_per_cluster
    avg_words = float(words_per_cluster.mean())
    total_word_freq = np.maximum(X_counts.sum(axis=0, keepdims=True), 1.0)
    idf = np.log(1.0 + (avg_words / total_word_freq))
    c_tfidf = tf * idf  # (C, V)

    topics: Dict[int, List[Tuple[str, float]]] = {}
    for idx, cid in enumerate(cluster_ids):
        scores = c_tfidf[idx]
        cand_pool_idx = np.argsort(scores)[::-1][: min(len(vocab), top_k * 3)]
        cand_words = [str(vocab[i]) for i in cand_pool_idx]
        cand_scores = [float(scores[i]) for i in cand_pool_idx]

        if mmr_diversity <= 0.0 or len(cand_words) <= top_k:
            topics[cid] = [
                (w, round(s, 4)) for w, s in zip(cand_words[:top_k], cand_scores[:top_k])
            ]
            continue

        # Character 3-gram embeddings of candidate keywords for MMR diversity
        char_vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        W_emb = normalize(char_vec.fit_transform(cand_words).toarray().astype(np.float64), norm="l2")
        sim_mat = W_emb @ W_emb.T

        selected_indices = [0]
        remaining = list(range(1, len(cand_words)))
        norm_rel = np.array(cand_scores) / max(cand_scores[0], 1e-9)

        while len(selected_indices) < top_k and remaining:
            best_mmr = -1e9
            best_r = remaining[0]
            for r in remaining:
                max_sim_to_sel = max(float(sim_mat[r, s]) for s in selected_indices)
                mmr_score = (1.0 - mmr_diversity) * norm_rel[r] - mmr_diversity * max_sim_to_sel
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_r = r
            selected_indices.append(best_r)
            remaining.remove(best_r)

        topics[cid] = [(cand_words[i], round(cand_scores[i], 4)) for i in selected_indices]

    return topics


# =============================================================================
# 4. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    centers = rng.normal(0, 3.0, size=(cfg.n_clusters, cfg.embed_dim))
    true_labels = np.arange(cfg.n_samples) % cfg.n_clusters
    X = centers[true_labels] + rng.normal(0, 0.5, size=(cfg.n_samples, cfg.embed_dim))

    # 1. Test HDBSCAN -1 outlier reassignment
    noisy_labels = true_labels.copy()
    noisy_labels[::5] = -1  # mark 20% as -1 outliers
    reassigned_cos, soft_probs = reassign_hdbscan_outliers(X, noisy_labels, method="centroid_cosine")
    reassigned_knn, _ = reassign_hdbscan_outliers(X, noisy_labels, method="knn")
    assert (reassigned_cos >= 0).all() and (reassigned_knn >= 0).all()
    assert soft_probs.shape == (cfg.n_samples, cfg.n_clusters)

    # 2. Test ConsensusClusterer across 5 KMeans seeds
    partitions = [
        KMeans(n_clusters=cfg.n_clusters, n_init=1, random_state=cfg.seed + s).fit_predict(X)
        for s in range(5)
    ]
    consensus = ConsensusClusterer(n_clusters=cfg.n_clusters, mode="coassociation")
    ens_labels, ens_probs = consensus.fit_predict(partitions)
    ari = float(adjusted_rand_score(true_labels, ens_labels))
    assert ens_labels.shape == (cfg.n_samples,)
    assert ens_probs.shape == (cfg.n_samples, cfg.n_clusters)

    # 3. Test c-TF-IDF + MMR keyword extraction
    topic_vocab = {
        0: "inflation bank interest rate bond fiscal economy",
        1: "football striker goal tournament stadium coach",
        2: "transformer attention neural gradient tensor gpu",
        3: "vaccine hospital patient clinical medical doctor",
    }
    texts = [topic_vocab[int(c)] for c in true_labels]
    topics = compute_c_tfidf_topics(texts, ens_labels, top_k=cfg.top_k_words, mmr_diversity=cfg.mmr_diversity)
    assert len(topics) == cfg.n_clusters

    print("[SMOKE TEST] Task 03 Clustering Tricks Module")
    print(f"  Outlier reassignment (-1 -> 0..K-1) : 100% assigned, soft_probs={soft_probs.shape}")
    print(f"  ConsensusClusterer ARI vs GroundTruth: {ari:.4f}")
    print(f"  c-TF-IDF + MMR Topic 0 Top Keywords : {topics[0][:4]}")
    print("[SUCCESS] All Task 03 module checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 Clustering Tricks Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
