#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Tier 1 Fast CPU LSA + Spherical K-Means + NMF
====================================================================================
Standalone CPU clustering and topic modeling baseline implementing:
  1. FeatureUnion of Word `(1, 2)` TF-IDF + Char-wb `(3, 5)` TF-IDF.
  2. Latent Semantic Analysis (`TruncatedSVD`) + L2 `Normalizer` $\rightarrow$ **Spherical K-Means**
     (K-Means on unit hypersphere where Euclidean distance is monotonic with cosine similarity).
  3. Non-Negative Matrix Factorization (**NMF** with Kullback-Leibler / Frobenius loss) soft
     topic distributions blended with Spherical K-Means cosine similarity probabilities.
  4. Automatic $K$ selection via rank-aggregated Silhouette, Calinski-Harabasz, and
     Davies-Bouldin scan across `k_candidates`.
  5. Top-10 **c-TF-IDF** topic keyword extraction per discovered cluster.
  6. Standardized output contract: saves `oof_tier1_clustering.npy` (soft cluster probs),
     `test_tier1_clustering.npy`, and `sub_tier1_clustering.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
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
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer, normalize


@dataclass
class CFG:
    """Configuration for Tier 1 LSA + Spherical K-Means + NMF Baseline."""

    exp_name: str = "tier1_clustering"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "cluster"

    n_clusters: Optional[int] = None  # If None, auto-selected via metric scan
    k_candidates: Tuple[int, ...] = (3, 4, 5, 6, 8)
    svd_components: int = 32
    max_word_features: int = 30000
    max_char_features: int = 40000
    nmf_blend_weight: float = 0.30
    top_k_keywords: int = 10

    seed: int = 42
    smoke_test: bool = False


def select_optimal_k(
    X_unit: np.ndarray, k_candidates: Sequence[int], seed: int = 42
) -> Tuple[int, pd.DataFrame]:
    """
    Scans candidate cluster counts `K` using Spherical K-Means and ranks each `K` across:
      - Silhouette Score (cosine, higher is better)
      - Calinski-Harabasz Index (higher is better)
      - Davies-Bouldin Index (lower is better)
    Returns `(best_k, scan_results_df)`.
    """
    records = []
    for k in k_candidates:
        if k < 2 or k >= len(X_unit):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X_unit)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(X_unit, labels, metric="cosine"))
        ch = float(calinski_harabasz_score(X_unit, labels))
        db = float(davies_bouldin_score(X_unit, labels))
        records.append({"k": int(k), "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})

    scan_df = pd.DataFrame(records)
    # Composite score: normalized Silhouette + normalized CH - normalized DB
    sil_norm = (scan_df["silhouette"] - scan_df["silhouette"].min()) / max(
        float(np.ptp(scan_df["silhouette"].values)), 1e-9
    )
    ch_norm = (scan_df["calinski_harabasz"] - scan_df["calinski_harabasz"].min()) / max(
        float(np.ptp(scan_df["calinski_harabasz"].values)), 1e-9
    )
    db_norm = (scan_df["davies_bouldin"] - scan_df["davies_bouldin"].min()) / max(
        float(np.ptp(scan_df["davies_bouldin"].values)), 1e-9
    )
    scan_df["composite_score"] = (0.50 * sil_norm + 0.25 * ch_norm + 0.25 * (1.0 - db_norm)).round(4)
    best_row = scan_df.sort_values("composite_score", ascending=False).iloc[0]
    return int(best_row["k"]), scan_df


def extract_c_tfidf_keywords(
    texts: Sequence[str], labels: np.ndarray, top_k: int = 10
) -> Dict[int, List[str]]:
    """Extracts top-k discriminative keywords per cluster using class-based TF-IDF (c-TF-IDF)."""
    unique_c = np.sort(np.unique(labels))
    grouped_docs = [
        " ".join(str(texts[i]) for i in np.where(labels == c)[0]) for c in unique_c
    ]
    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=15000)
    counts = vec.fit_transform(grouped_docs).toarray().astype(np.float64)
    vocab = np.array(vec.get_feature_names_out())

    words_per_c = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    tf = counts / words_per_c
    avg_w = float(words_per_c.mean())
    idf = np.log(1.0 + avg_w / np.maximum(counts.sum(axis=0, keepdims=True), 1.0))
    c_tfidf = tf * idf

    out: Dict[int, List[str]] = {}
    for idx, c in enumerate(unique_c):
        top_idx = np.argsort(c_tfidf[idx])[::-1][:top_k]
        out[int(c)] = [str(vocab[j]) for j in top_idx]
    return out


def align_nmf_to_kmeans(km_probs: np.ndarray, nmf_probs: np.ndarray) -> np.ndarray:
    """Aligns NMF topic columns to Spherical K-Means clusters via correlation matching."""
    from scipy.optimize import linear_sum_assignment

    sim = km_probs.T @ nmf_probs
    row_ind, col_ind = linear_sum_assignment(-sim)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    return nmf_probs[:, perm]


def run_clustering_pipeline(
    train_df: pd.DataFrame, test_df: Optional[pd.DataFrame], cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
    tr_texts = train_df[cfg.text_col].fillna("").astype(str).tolist()
    te_texts = (
        test_df[cfg.text_col].fillna("").astype(str).tolist()
        if test_df is not None
        else tr_texts[: min(20, len(tr_texts))]
    )
    all_texts = tr_texts + te_texts
    n_tr = len(tr_texts)

    # 1. Word (1, 2) + Char-wb (3, 5) TF-IDF for LSA
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=cfg.max_word_features,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    max_features=cfg.max_char_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    X_sparse = union.fit_transform(all_texts)

    n_comp = min(cfg.svd_components, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components=max(2, n_comp), random_state=cfg.seed)
    X_lsa = Normalizer(copy=False).fit_transform(svd.fit_transform(X_sparse))

    # 2. Automatic K selection if n_clusters is None
    if cfg.n_clusters is None:
        best_k, scan_df = select_optimal_k(X_lsa[:n_tr], cfg.k_candidates, seed=cfg.seed)
        print("=" * 72)
        print(f"AUTOMATIC K SELECTION SCAN (Selected K={best_k})")
        print("=" * 72)
        print(scan_df.to_string(index=False))
    else:
        best_k = cfg.n_clusters

    # 3. Spherical K-Means
    km = KMeans(n_clusters=best_k, n_init=15, random_state=cfg.seed)
    km.fit(X_lsa[:n_tr])
    centroids = normalize(km.cluster_centers_, norm="l2")
    km_probs_all = softmax((X_lsa @ centroids.T) * 8.0, axis=1)

    # 4. NMF Topic Modeling on Word TF-IDF
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=cfg.max_word_features, stop_words="english")
    X_word = word_vec.fit_transform(all_texts)
    nmf = NMF(n_components=best_k, init="nndsvd", random_state=cfg.seed, max_iter=300)
    W_all = nmf.fit_transform(X_word)
    nmf_probs_all = W_all / np.maximum(W_all.sum(axis=1, keepdims=True), 1e-12)
    nmf_aligned = align_nmf_to_kmeans(km_probs_all, nmf_probs_all)

    # 5. Blended soft cluster assignment
    w_nmf = cfg.nmf_blend_weight
    probs_all = (1.0 - w_nmf) * km_probs_all + w_nmf * nmf_aligned
    oof_probs = probs_all[:n_tr].astype(np.float32)
    test_probs = probs_all[n_tr:].astype(np.float32)
    oof_labels = np.argmax(oof_probs, axis=1)
    test_labels = np.argmax(test_probs, axis=1)

    # 6. Top-10 c-TF-IDF keywords per cluster
    topics = extract_c_tfidf_keywords(tr_texts, oof_labels, top_k=cfg.top_k_keywords)
    print("\n" + "=" * 72)
    print(f"TOP-{cfg.top_k_keywords} c-TF-IDF TOPIC KEYWORDS PER CLUSTER")
    print("=" * 72)
    for cid, words in topics.items():
        count_c = int((oof_labels == cid).sum())
        print(f"  Cluster {cid} (n={count_c:<3}): {', '.join(words[:8])}")

    if "true_topic" in train_df.columns:
        nmi = float(normalized_mutual_info_score(train_df["true_topic"].values, oof_labels))
        ari = float(adjusted_rand_score(train_df["true_topic"].values, oof_labels))
        print(f"[EVALUATION VS GROUND TRUTH] NMI={nmi:.4f} | ARI={ari:.4f}")

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
    parser = argparse.ArgumentParser(description="Task 03 Tier 1 LSA + Spherical K-Means + NMF")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        n_clusters=args.n_clusters,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(cfg.seed)
        themes = [
            "inflation central bank interest rate fiscal monetary bond yield",
            "football tournament striker goalkeeper stadium championship league",
            "transformer neural network backpropagation gradient tensor gpu",
            "hospital clinical vaccine patient epidemiology diagnosis doctor",
        ]
        tr_df = pd.DataFrame(
            [
                {
                    "id": f"tr_{i}",
                    "text": " ".join(rng.choice(themes[i % 4].split(), size=20, replace=True)),
                    "true_topic": i % 4,
                }
                for i in range(120)
            ]
        )
        te_df = pd.DataFrame(
            [
                {
                    "id": f"te_{i}",
                    "text": " ".join(rng.choice(themes[i % 4].split(), size=20, replace=True)),
                }
                for i in range(32)
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 03 Tier 1 Spherical K-Means + NMF + Auto-K Scan...")
            oof, test_p, topics = run_clustering_pipeline(tr_df, te_df, cfg)
            assert oof.shape[0] == 120 and test_p.shape[0] == 32
            assert len(topics) >= 3
            print("[SUCCESS] Task 03 Tier 1 smoke test passed.")
    else:
        tr_df = load_table(cfg.train_path)
        te_df = load_table(cfg.test_path) if pathlib.Path(cfg.test_path).exists() else None
        run_clustering_pipeline(tr_df, te_df, cfg)


if __name__ == "__main__":
    main()
