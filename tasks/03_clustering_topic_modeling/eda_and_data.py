#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Unsupervised Corpus EDA & Diagnostics
============================================================================
Standalone unsupervised text corpus analyzer providing:
  1. Document length distribution (characters, tokens, empty/ultra-short rate).
  2. Exact duplicate & boilerplate header/footer detection.
  3. Stopword & noise ratio (Indonesian + English core stopword sets + digit/symbol noise).
  4. Vocabulary Zipf power-law check (linear regression slope & $R^2$ of $\log(\text{freq})$ vs $\log(\text{rank})$).
  5. Intrinsic clusterability estimation:
     - **Hopkins Statistic** $H \in [0, 1]$ ($H > 0.70$ indicates strong cluster tendency vs uniform noise $H \approx 0.50$).
     - $k$-Nearest Neighbor cosine similarity quantiles ($p_{10}, p_{50}, p_{90}$).

Usage:
    python eda_and_data.py --data-path train_folds.csv
    python eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import site
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer


# Core bilingual stopword set (Indonesian + English)
STOPWORDS_ID_EN = {
    # Indonesian
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "sebagai", "dalam", "tidak", "akan", "juga", "atau", "ada", "mereka",
    "sudah", "saya", "kita", "kami", "bisa", "oleh", "karena", "tersebut", "lebih",
    "telah", "dapat", "para", "harus", "namun", "secara", "antara", "seperti",
    # English
    "the", "and", "to", "of", "in", "is", "for", "that", "with", "on", "as", "are",
    "this", "by", "be", "at", "or", "from", "an", "it", "not", "have", "has", "was",
    "were", "but", "which", "their", "they", "we", "you", "can", "will", "about",
}


@dataclass
class CFG:
    """Configuration for Unsupervised Corpus EDA & Clusterability Diagnostics."""

    data_path: str = "train_folds.parquet"
    text_col: str = "text"
    svd_components: int = 16
    hopkins_sample_size: int = 80
    knn_k: int = 5
    seed: int = 42
    smoke_test: bool = False


def analyze_lengths_and_duplicates(texts: Sequence[str]) -> Dict[str, Any]:
    """Computes document length quantiles, exact duplicate rates, and boilerplate prefixes/suffixes."""
    clean_texts = [str(t or "").strip() for t in texts]
    n = max(len(clean_texts), 1)
    char_lens = np.array([len(t) for t in clean_texts], dtype=np.float64)
    word_lens = np.array([len(t.split()) for t in clean_texts], dtype=np.float64)

    unique_texts = len(set(clean_texts))
    dup_rate = 1.0 - (unique_texts / n)

    # Detect recurring 3-word prefix or suffix boilerplate (>10% corpus share)
    prefix_counter: Counter[str] = Counter()
    suffix_counter: Counter[str] = Counter()
    for t in clean_texts:
        words = t.lower().split()
        if len(words) >= 4:
            prefix_counter[" ".join(words[:3])] += 1
            suffix_counter[" ".join(words[-3:])] += 1

    top_prefix = prefix_counter.most_common(1)[0] if prefix_counter else ("", 0)
    top_suffix = suffix_counter.most_common(1)[0] if suffix_counter else ("", 0)

    return {
        "num_documents": len(clean_texts),
        "unique_documents": unique_texts,
        "exact_duplicate_ratio": round(float(dup_rate), 4),
        "empty_or_short_lt3_words": int((word_lens < 3).sum()),
        "word_len_mean": round(float(word_lens.mean()), 2),
        "word_len_p50": round(float(np.median(word_lens)), 2),
        "word_len_p90": round(float(np.percentile(word_lens, 90)), 2),
        "top_boilerplate_prefix": f"'{top_prefix[0]}' ({top_prefix[1]}/{n})",
        "top_boilerplate_suffix": f"'{top_suffix[0]}' ({top_suffix[1]}/{n})",
    }


def analyze_stopword_and_zipf(texts: Sequence[str]) -> Dict[str, Any]:
    """Computes Indonesian + English stopword ratio, digit noise ratio, and Zipf power-law fit."""
    token_re = re.compile(r"\b\w+\b")
    vocab_counter: Counter[str] = Counter()
    total_tokens = 0
    stop_tokens = 0
    digit_tokens = 0

    for t in texts:
        tokens = token_re.findall(str(t).lower())
        total_tokens += len(tokens)
        for tok in tokens:
            vocab_counter[tok] += 1
            if tok in STOPWORDS_ID_EN:
                stop_tokens += 1
            elif tok.isdigit():
                digit_tokens += 1

    freqs = np.array(sorted(vocab_counter.values(), reverse=True), dtype=np.float64)
    if len(freqs) >= 5:
        log_ranks = np.log(np.arange(1, len(freqs) + 1, dtype=np.float64))
        log_freqs = np.log(np.maximum(freqs, 1.0))
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        pred = slope * log_ranks + intercept
        ss_res = float(np.sum((log_freqs - pred) ** 2))
        ss_tot = float(np.sum((log_freqs - np.mean(log_freqs)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    else:
        slope, r2 = -1.0, 0.0

    denom = max(total_tokens, 1)
    return {
        "vocab_size": len(vocab_counter),
        "total_tokens": total_tokens,
        "stopword_ratio_id_en": round(stop_tokens / denom, 4),
        "digit_noise_ratio": round(digit_tokens / denom, 4),
        "hapax_legomena_ratio": round(sum(1 for c in freqs if c == 1) / max(len(freqs), 1), 4),
        "zipf_power_law_slope": round(float(slope), 4),
        "zipf_power_law_r2": round(float(r2), 4),
    }


def compute_hopkins_and_knn_clusterability(
    texts: Sequence[str],
    n_components: int = 16,
    sample_size: int = 80,
    knn_k: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    r"""
    Estimates intrinsic clusterability of the corpus in L2-normalized LSA space:
      - Hopkins statistic $H = \frac{\sum u_i^d}{\sum u_i^d + \sum w_i^d}$, where $w_i$ are
        nearest-neighbor distances of real points and $u_i$ are nearest-neighbor distances
        from uniformly sampled points in the bounding box.
      - Nearest-neighbor cosine similarity quantiles.
    """
    rng = np.random.default_rng(seed)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_tfidf = vec.fit_transform([str(t) for t in texts])

    k_svd = min(n_components, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    if k_svd >= 2:
        svd = TruncatedSVD(n_components=k_svd, random_state=seed)
        X_dense = Normalizer(copy=False).fit_transform(svd.fit_transform(X_tfidf))
    else:
        X_dense = Normalizer(copy=False).fit_transform(X_tfidf.toarray())

    n_samples, dim = X_dense.shape
    m = min(sample_size, max(5, n_samples // 2))
    sampled_idx = rng.choice(n_samples, size=m, replace=False)

    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n_samples), metric="euclidean").fit(X_dense)
    real_dists, _ = nn.kneighbors(X_dense[sampled_idx], n_neighbors=2)
    w_dists = real_dists[:, 1]  # distance to nearest non-self neighbor

    mins = X_dense.min(axis=0)
    maxs = X_dense.max(axis=0)
    uniform_pts = rng.uniform(mins, maxs, size=(m, dim))
    uniform_Pts_norm = Normalizer(copy=False).fit_transform(uniform_pts)
    u_dists, _ = nn.kneighbors(uniform_Pts_norm, n_neighbors=1)
    u_dists = u_dists[:, 0]

    # Use effective dimensionality <= 4 in exponent for numerical stability
    eff_d = min(dim, 4)
    u_pow = np.sum(np.power(np.maximum(u_dists, 1e-6), eff_d))
    w_pow = np.sum(np.power(np.maximum(w_dists, 1e-6), eff_d))
    hopkins = float(u_pow / max(u_pow + w_pow, 1e-12))

    # Cosine similarity to k-nearest neighbors
    cos_matrix = X_dense @ X_dense.T
    np.fill_diagonal(cos_matrix, -1.0)
    top1_cos = np.max(cos_matrix, axis=1)

    return {
        "hopkins_statistic": round(hopkins, 4),
        "nn1_cosine_p10": round(float(np.percentile(top1_cos, 10)), 4),
        "nn1_cosine_median": round(float(np.median(top1_cos)), 4),
        "nn1_cosine_p90": round(float(np.percentile(top1_cos, 90)), 4),
    }


def generate_synthetic_corpus(n_docs: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = [
        "kebijakan ekonomi bank sentral inflasi suku bunga obligasi pasar modal investasi",
        "pertandingan sepak bola liga stadion pelatih gol turnamen kejuaraan atlet",
        "arsitektur jaringan saraf tiruan transformer komputasi gpu algoritma data",
        "kesehatan rumah sakit vaksinasi dokter pasien medis nutrisi epidemiologi",
    ]
    rows = []
    for i in range(n_docs):
        t_idx = i % len(topics)
        words = rng.choice(topics[t_idx].split(), size=int(rng.integers(12, 35)), replace=True)
        prefix = "breaking newsindonesia:" if i % 6 == 0 else ""
        text = f"{prefix} {' '.join(words)} yang dan di untuk pada"
        if i % 25 == 0 and i > 0:
            text = rows[i - 1]["text"]  # inject exact duplicates
        rows.append({"id": f"doc_{i}", "text": text, "true_topic": t_idx})
    return pd.DataFrame(rows)


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 Unsupervised Text Corpus EDA")
    parser.add_argument("--data-path", type=str, default=CFG.data_path)
    parser.add_argument("--text-col", type=str, default=CFG.text_col)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(data_path=args.data_path, text_col=args.text_col, smoke_test=args.smoke_test)
    if cfg.smoke_test:
        print("[SMOKE TEST] Generating synthetic multi-topic Indonesian/English corpus...")
        df = generate_synthetic_corpus(seed=cfg.seed)
    else:
        df = load_table(cfg.data_path)

    texts = df[cfg.text_col].fillna("").astype(str).tolist()

    print("=" * 72)
    print("1. DOCUMENT LENGTH & DUPLICATE/BOILERPLATE DIAGNOSTICS")
    print("=" * 72)
    for k, v in analyze_lengths_and_duplicates(texts).items():
        print(f"  {k:<32}: {v}")

    print("\n" + "=" * 72)
    print("2. STOPWORD RATIO & ZIPF POWER-LAW CHECK")
    print("=" * 72)
    for k, v in analyze_stopword_and_zipf(texts).items():
        print(f"  {k:<32}: {v}")

    print("\n" + "=" * 72)
    print("3. INTRINSIC CLUSTERABILITY (HOPKINS & NEAREST-NEIGHBOR COSINE)")
    print("=" * 72)
    cluster_stats = compute_hopkins_and_knn_clusterability(
        texts, n_components=cfg.svd_components, sample_size=cfg.hopkins_sample_size, seed=cfg.seed
    )
    for k, v in cluster_stats.items():
        print(f"  {k:<32}: {v}")
    assert 0.0 <= cluster_stats["hopkins_statistic"] <= 1.0
    print("[SUCCESS] Task 03 Unsupervised EDA completed.")


if __name__ == "__main__":
    main()
