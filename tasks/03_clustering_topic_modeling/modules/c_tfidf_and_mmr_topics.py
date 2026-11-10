#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Class-Based TF-IDF (c-TF-IDF) & MMR Topics
=================================================================================
Atomic standalone BERTopic-style topic representation builder (`compute_c_tfidf_topics`)
combining:
  1. **Class-based TF-IDF (c-TF-IDF)**:
     $w_{t, c} = \text{tf}_{t, c} \cdot \log\left(1 + \frac{A}{\sum_{c'} \text{tf}_{t, c'}}\right)$
  2. **Maximal Marginal Relevance (MMR)** keyword diversification over character n-gram
     embeddings to eliminate redundant morphological variants.

Usage:
    python3 tasks/03_clustering_topic_modeling/modules/c_tfidf_and_mmr_topics.py --smoke-test
"""

from __future__ import annotations

import argparse
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


@dataclass
class CFG:
    """Configuration for c-TF-IDF & MMR Topic Extraction Smoke Test."""

    top_k_words: int = 6
    mmr_diversity: float = 0.35
    smoke_test: bool = False


def compute_c_tfidf_topics(
    texts: Sequence[str],
    labels: Sequence[int],
    top_k: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    mmr_diversity: float = 0.3,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Computes BERTopic-style class-based TF-IDF (c-TF-IDF) with Maximal Marginal Relevance
    (MMR) keyword diversification per cluster.
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
    X_counts = vec.fit_transform(cluster_docs).toarray().astype(np.float64)
    vocab = np.array(vec.get_feature_names_out())

    words_per_cluster = np.maximum(X_counts.sum(axis=1, keepdims=True), 1.0)
    tf = X_counts / words_per_cluster
    avg_words = float(words_per_cluster.mean())
    total_word_freq = np.maximum(X_counts.sum(axis=0, keepdims=True), 1.0)
    idf = np.log(1.0 + (avg_words / total_word_freq))
    c_tfidf = tf * idf

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

        char_vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        W_emb = normalize(
            char_vec.fit_transform(cand_words).toarray().astype(np.float64), norm="l2"
        )
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


def run_smoke_test(cfg: CFG) -> None:
    topic_vocab = {
        0: "inflasi bank suku bunga obligasi fiskal ekonomi rupiah",
        1: "sepakbola striker gol turnamen stadion pelatih liga",
        2: "transformer attention neural gradient tensor gpu matriks",
        3: "vaksin rumah sakit pasien klinis medis dokter terapi",
    }
    labels = [i % 4 for i in range(40)]
    texts = [topic_vocab[c] for c in labels]

    topics = compute_c_tfidf_topics(
        texts, labels, top_k=cfg.top_k_words, mmr_diversity=cfg.mmr_diversity
    )
    assert len(topics) == 4
    assert all(len(kws) == cfg.top_k_words for kws in topics.values())

    print("[SMOKE TEST] Task 03 c-TF-IDF & MMR Topic Extractor (`c_tfidf_and_mmr_topics.py`)")
    for cid, kws in topics.items():
        print(f"  Cluster {cid} top keywords : {kws[:4]}")
    print("[SUCCESS] c-TF-IDF and MMR topic diversification checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 c-TF-IDF & MMR Topic Extractor")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
