#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Tier 2 Dense Embedding + Manifold BERTopic Pipeline
==========================================================================================
Standalone BERTopic-style neural clustering & topic modeling pipeline without heavy
external wrapper dependencies:
  1. **Dense Sentence Embeddings**: Uses `sentence-transformers` or HuggingFace `AutoModel`
     mean-pooled representations (with automatic TF-IDF + SVD fallback for offline `--smoke-test`).
  2. **L2 Normalization** on the unit hypersphere.
  3. **Manifold Dimensionality Reduction**: Uses `umap.UMAP(metric='cosine')` when `umap-learn`
     is installed, with automatic fallback to Cosine `SpectralEmbedding` / `PCA`.
  4. **Density / Centroid Clustering**: Runs `hdbscan.HDBSCAN` when installed (or density-aware
     DBSCAN / KMeans fallback) followed by **Mandatory `-1` Outlier Reassignment** via soft
     cosine similarity to cluster centroids (`reassign_outliers_to_centroids`).
  5. **c-TF-IDF + Maximal Marginal Relevance (MMR)** topic representation per cluster.
  6. Standardized output contract: saves `oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     and `sub_<exp_name>.csv`.

Usage:
    python baselines/tier2_strong_dl.py --model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    python baselines/tier2_strong_dl.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
import tempfile
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
from scipy.special import softmax
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from transformers import AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class CFG:
    """Configuration for Tier 2 Dense Embedding + Manifold Clustering (BERTopic-style)."""

    exp_name: str = "tier2_bertopic"
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "cluster"

    # Manifold & clustering hyperparameters
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    min_cluster_size: int = 10
    n_clusters_fallback: int = 4

    # Topic representation
    top_k_keywords: int = 10
    mmr_diversity: float = 0.35

    batch_size: int = 32
    max_len: int = 256
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Dense Sentence Encoder (HuggingFace Mean-Pool + Offline SVD Fallback)
# =============================================================================


@torch.no_grad()
def encode_texts_dense(texts: Sequence[str], cfg: CFG) -> np.ndarray:
    """
    Encodes documents into L2-normalized dense embeddings using HuggingFace `AutoModel`
    mean pooling, or falls back to Word+Char TF-IDF + SVD when offline / `--smoke-test`.
    """
    if not cfg.smoke_test and HAS_TRANSFORMERS:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model = AutoModel.from_pretrained(cfg.model_name).to(device)
            model.eval()
            all_vecs = []
            for start in range(0, len(texts), cfg.batch_size):
                batch_texts = [str(t) for t in texts[start : start + cfg.batch_size]]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_len,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_vecs.append(pooled.cpu().numpy())
            return normalize(np.vstack(all_vecs), norm="l2")
        except Exception as exc:
            print(f"[WARN] Could not load HF encoder '{cfg.model_name}' ({exc}); using SVD fallback.")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    X_tfidf = vec.fit_transform([str(t) for t in texts])
    n_comp = min(32, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=max(2, n_comp), random_state=cfg.seed)
    return normalize(svd.fit_transform(X_tfidf), norm="l2")


# =============================================================================
# 2. Manifold Reduction (UMAP / Spectral) + Clustering + Outlier Reassignment
# =============================================================================


def reduce_manifold(embeddings: np.ndarray, cfg: CFG) -> np.ndarray:
    """Reduces L2-normalized embeddings onto a low-dimensional manifold via UMAP or SpectralEmbedding."""
    n_comp = min(cfg.umap_n_components, embeddings.shape[0] - 2, embeddings.shape[1])
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=min(cfg.umap_n_neighbors, embeddings.shape[0] - 1),
            min_dist=cfg.umap_min_dist,
            metric="cosine",
            random_state=cfg.seed,
        )
        return np.asarray(reducer.fit_transform(embeddings), dtype=np.float64)

    n_nb = min(cfg.umap_n_neighbors, max(2, embeddings.shape[0] // 4))
    spec = SpectralEmbedding(
        n_components=max(2, n_comp),
        affinity="nearest_neighbors",
        n_neighbors=n_nb,
        random_state=cfg.seed,
    )
    return spec.fit_transform(embeddings)


def cluster_and_reassign_outliers(
    embeddings_hd: np.ndarray, manifold_ld: np.ndarray, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Clusters manifold coordinates via HDBSCAN (or DBSCAN/KMeans fallback), then enforces
    **Mandatory `-1` Outlier Reassignment** using cosine similarity to high-dimensional
    cluster centroids.
    Returns `(clean_labels, soft_probs, num_reassigned_outliers)`.
    """
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        raw_labels = clusterer.fit_predict(manifold_ld)
    else:
        # Run DBSCAN with automatic radius; if too few clusters, use KMeans + distance outlier gate
        km = KMeans(n_clusters=cfg.n_clusters_fallback, n_init=10, random_state=cfg.seed)
        raw_labels = km.fit_predict(manifold_ld)
        dists = np.linalg.norm(manifold_ld - km.cluster_centers_[raw_labels], axis=1)
        outlier_thr = np.percentile(dists, 90)
        raw_labels[dists > outlier_thr] = -1

    num_outliers = int((raw_labels < 0).sum())
    valid_mask = raw_labels >= 0
    unique_c = np.sort(np.unique(raw_labels[valid_mask]))

    if len(unique_c) < 2:
        km = KMeans(n_clusters=cfg.n_clusters_fallback, n_init=10, random_state=cfg.seed)
        raw_labels = km.fit_predict(embeddings_hd)
        valid_mask = raw_labels >= 0
        unique_c = np.sort(np.unique(raw_labels))

    remap = {int(c): i for i, c in enumerate(unique_c)}
    clean_labels = np.array([remap.get(int(y), -1) for y in raw_labels], dtype=np.int64)
    k = len(unique_c)

    centroids = np.zeros((k, embeddings_hd.shape[1]), dtype=np.float64)
    for c in range(k):
        centroids[c] = embeddings_hd[clean_labels == c].mean(axis=0)
    centroids = normalize(centroids, norm="l2")

    cos_sim = embeddings_hd @ centroids.T
    soft_probs = softmax(cos_sim * 10.0, axis=1).astype(np.float32)

    outlier_idx = np.where(clean_labels < 0)[0]
    if len(outlier_idx) > 0:
        clean_labels[outlier_idx] = np.argmax(cos_sim[outlier_idx], axis=1)

    return clean_labels, soft_probs, num_outliers


# =============================================================================
# 3. c-TF-IDF + Maximal Marginal Relevance (MMR) Topic Representation
# =============================================================================


def extract_c_tfidf_mmr_topics(
    texts: Sequence[str],
    labels: np.ndarray,
    top_k: int = 10,
    diversity: float = 0.35,
) -> Dict[int, List[Tuple[str, float]]]:
    """Extracts diversified topic keywords per cluster via c-TF-IDF + MMR."""
    unique_c = np.sort(np.unique(labels))
    grouped_docs = [" ".join(str(texts[i]) for i in np.where(labels == c)[0]) for c in unique_c]

    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=15000)
    X_cnt = vec.fit_transform(grouped_docs).toarray().astype(np.float64)
    vocab = np.array(vec.get_feature_names_out())

    words_per_c = np.maximum(X_cnt.sum(axis=1, keepdims=True), 1.0)
    tf = X_cnt / words_per_c
    idf = np.log(1.0 + float(words_per_c.mean()) / np.maximum(X_cnt.sum(axis=0, keepdims=True), 1.0))
    c_tfidf = tf * idf

    topics: Dict[int, List[Tuple[str, float]]] = {}
    for idx, cid in enumerate(unique_c):
        row = c_tfidf[idx]
        pool_idx = np.argsort(row)[::-1][: min(len(vocab), top_k * 3)]
        cand_words = [str(vocab[j]) for j in pool_idx]
        cand_scores = [float(row[j]) for j in pool_idx]

        if len(cand_words) <= top_k or diversity <= 0.0:
            topics[int(cid)] = [(w, round(s, 4)) for w, s in zip(cand_words[:top_k], cand_scores[:top_k])]
            continue

        c_vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        W_mat = normalize(c_vec.fit_transform(cand_words).toarray().astype(np.float64), norm="l2")
        sim = W_mat @ W_mat.T
        rel = np.array(cand_scores) / max(cand_scores[0], 1e-9)

        selected = [0]
        remaining = list(range(1, len(cand_words)))
        while len(selected) < top_k and remaining:
            best_r = remaining[0]
            best_val = -1e9
            for r in remaining:
                max_s = max(float(sim[r, s]) for s in selected)
                val = (1.0 - diversity) * rel[r] - diversity * max_s
                if val > best_val:
                    best_val = val
                    best_r = r
            selected.append(best_r)
            remaining.remove(best_r)

        topics[int(cid)] = [(cand_words[j], round(cand_scores[j], 4)) for j in selected]

    return topics


# =============================================================================
# 4. End-to-End Pipeline & Smoke Test
# =============================================================================


def run_bertopic_pipeline(
    train_df: pd.DataFrame, test_df: Optional[pd.DataFrame], cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Tuple[str, float]]]]:
    tr_texts = train_df[cfg.text_col].fillna("").astype(str).tolist()
    te_texts = (
        test_df[cfg.text_col].fillna("").astype(str).tolist()
        if test_df is not None
        else tr_texts[: min(20, len(tr_texts))]
    )
    all_texts = tr_texts + te_texts
    n_tr = len(tr_texts)

    embeddings = encode_texts_dense(all_texts, cfg)
    manifold = reduce_manifold(embeddings, cfg)
    labels_all, probs_all, num_reassigned = cluster_and_reassign_outliers(embeddings, manifold, cfg)

    oof_probs = probs_all[:n_tr]
    test_probs = probs_all[n_tr:]
    oof_labels = labels_all[:n_tr]
    test_labels = labels_all[n_tr:]

    sil = float(silhouette_score(embeddings[:n_tr], oof_labels, metric="cosine"))
    topics = extract_c_tfidf_mmr_topics(
        tr_texts, oof_labels, top_k=cfg.top_k_keywords, diversity=cfg.mmr_diversity
    )

    print("=" * 72)
    print("TIER 2 DENSE EMBEDDING + MANIFOLD CLUSTERING (BERTopic Pipeline)")
    print("=" * 72)
    print(f"  Discovered Clusters        : {len(topics)}")
    print(f"  Reassigned -1 Outliers     : {num_reassigned}/{len(all_texts)} (0 remaining -1 labels)")
    print(f"  Cosine Silhouette Score    : {sil:.4f}")
    if "true_topic" in train_df.columns:
        nmi = float(normalized_mutual_info_score(train_df["true_topic"].values, oof_labels))
        ari = float(adjusted_rand_score(train_df["true_topic"].values, oof_labels))
        print(f"  Ground-Truth NMI / ARI     : NMI={nmi:.4f} | ARI={ari:.4f}")

    for cid, kw_list in topics.items():
        formatted = ", ".join(f"{w} ({s:.3f})" for w, s in kw_list[:6])
        print(f"  Topic {cid}: {formatted}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs)
    test_ids = test_df[cfg.id_col].values if test_df is not None else np.arange(len(test_labels))
    pd.DataFrame({cfg.id_col: test_ids, cfg.target_col: test_labels}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )
    return oof_probs, test_probs, topics


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 Tier 2 BERTopic-Style Pipeline")
    parser.add_argument("--model-name", type=str, default=CFG.model_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run offline smoke test")
    args = parser.parse_args()

    cfg = CFG(
        model_name=args.model_name,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(cfg.seed)
        themes = [
            "bank central inflation interest rate bond yield fiscal policy",
            "football striker stadium tournament goal championship match",
            "transformer attention neural network gradient tensor gpu",
            "vaccine hospital clinical patient doctor medical epidemiology",
        ]
        tr_df = pd.DataFrame(
            [
                {
                    "id": f"tr_{i}",
                    "text": " ".join(rng.choice(themes[i % 4].split(), size=18, replace=True)),
                    "true_topic": i % 4,
                }
                for i in range(100)
            ]
        )
        te_df = pd.DataFrame(
            [
                {
                    "id": f"te_{i}",
                    "text": " ".join(rng.choice(themes[i % 4].split(), size=18, replace=True)),
                }
                for i in range(24)
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            oof, test_p, topics = run_bertopic_pipeline(tr_df, te_df, cfg)
            assert oof.shape[0] == 100 and test_p.shape[0] == 24
            assert len(topics) == 4
            print("[SUCCESS] Task 03 Tier 2 BERTopic smoke test passed.")
    else:
        tr_df = load_table(cfg.train_path)
        te_df = load_table(cfg.test_path) if pathlib.Path(cfg.test_path).exists() else None
        run_bertopic_pipeline(tr_df, te_df, cfg)


if __name__ == "__main__":
    main()
