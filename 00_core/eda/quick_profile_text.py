#!/usr/bin/env python3
"""
quick_profile_text.py
=====================
Fast competition NLP / Text EDA profiler.
Audits:
  - Character and word length quantiles (p10, p50, p90, p95, p99)
  - Missing, empty, or whitespace-only string counts
  - Vocabulary overlap and Out-Of-Vocabulary (OOV) token/occurrence rates between train & test
  - Surface noise densities: URL count, HTML tag count, emoji/non-ASCII density, digit density, uppercase ratio
  - Top class-discriminative unigrams & bigrams via Monroe et al. Dirichlet-smoothed log-odds ratio z-score

Usage:
  python3 00_core/eda/quick_profile_text.py --train-path train.parquet --test-path test.parquet --text-col text --target-col label
  python3 00_core/eda/quick_profile_text.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class CFG:
    train_path: str = "train.parquet"
    test_path: Optional[str] = "test.parquet"
    text_col: str = "text"
    target_col: Optional[str] = "target"
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90, 0.95, 0.99])
    top_k_ngrams: int = 10
    max_vocab_for_log_odds: int = 15000
    dirichlet_alpha_scale: float = 100.0
    output_json: Optional[str] = None


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def audit_text_series(series: pd.Series, quantiles: List[float]) -> Dict[str, Any]:
    n_total = len(series)
    n_null = int(series.isna().sum())
    s_str = series.fillna("").astype(str)
    n_empty_or_ws = int((s_str.str.strip() == "").sum()) - n_null

    char_lens = s_str.str.len().to_numpy(dtype=float)
    word_lens = s_str.str.split().map(len).to_numpy(dtype=float)

    char_q = np.quantile(char_lens, quantiles) if n_total > 0 else np.zeros(len(quantiles))
    word_q = np.quantile(word_lens, quantiles) if n_total > 0 else np.zeros(len(quantiles))

    url_has = s_str.map(lambda x: 1 if URL_RE.search(x) else 0).mean()
    html_has = s_str.map(lambda x: 1 if HTML_RE.search(x) else 0).mean()

    def _char_ratios(txt: str) -> Tuple[float, float, float]:
        if not txt:
            return 0.0, 0.0, 0.0
        L = len(txt)
        n_upper = sum(1 for c in txt if c.isupper())
        n_digit = sum(1 for c in txt if c.isdigit())
        n_non_ascii = sum(1 for c in txt if ord(c) > 127)
        return n_upper / L, n_digit / L, n_non_ascii / L

    ratios = np.array([_char_ratios(t) for t in s_str], dtype=float)
    upper_mean = float(ratios[:, 0].mean()) if len(ratios) else 0.0
    digit_mean = float(ratios[:, 1].mean()) if len(ratios) else 0.0
    non_ascii_mean = float(ratios[:, 2].mean()) if len(ratios) else 0.0

    return {
        "n_rows": n_total,
        "null_count": n_null,
        "empty_or_whitespace_only_count": int(max(n_empty_or_ws, 0)),
        "char_len_quantiles": {
            f"p{int(round(q * 100)):02d}": round(float(v), 2) for q, v in zip(quantiles, char_q)
        },
        "word_len_quantiles": {
            f"p{int(round(q * 100)):02d}": round(float(v), 2) for q, v in zip(quantiles, word_q)
        },
        "noise_densities": {
            "rows_with_url_pct": round(float(url_has * 100.0), 3),
            "rows_with_html_pct": round(float(html_has * 100.0), 3),
            "mean_uppercase_char_ratio": round(upper_mean, 5),
            "mean_digit_char_ratio": round(digit_mean, 5),
            "mean_emoji_or_non_ascii_ratio": round(non_ascii_mean, 5),
        },
    }


def compute_vocab_overlap(train_texts: pd.Series, test_texts: pd.Series) -> Dict[str, Any]:
    tr_counter: Counter = Counter()
    te_counter: Counter = Counter()

    for t in train_texts.fillna("").astype(str):
        tr_counter.update(TOKEN_RE.findall(t.lower()))
    for t in test_texts.fillna("").astype(str):
        te_counter.update(TOKEN_RE.findall(t.lower()))

    tr_vocab = set(tr_counter.keys())
    te_vocab = set(te_counter.keys())
    shared_vocab = tr_vocab & te_vocab
    oov_types = te_vocab - tr_vocab

    total_te_tokens = sum(te_counter.values())
    oov_token_occurrences = sum(te_counter[w] for w in oov_types)

    return {
        "train_vocab_size": len(tr_vocab),
        "test_vocab_size": len(te_vocab),
        "shared_vocab_size": len(shared_vocab),
        "vocab_jaccard_overlap": round(len(shared_vocab) / max(len(tr_vocab | te_vocab), 1), 4),
        "test_oov_type_rate": round(len(oov_types) / max(len(te_vocab), 1), 4),
        "test_oov_token_rate": round(oov_token_occurrences / max(total_te_tokens, 1), 4),
    }


def compute_discriminative_ngrams_log_odds(
    texts: pd.Series,
    labels: pd.Series,
    cfg: CFG,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Monroe et al. (2008) Dirichlet-smoothed log-odds ratio with informative background prior.
    Returns top unigrams and bigrams per class ranked by z-score.
    """
    valid_mask = labels.notna()
    clean_texts = texts[valid_mask].fillna("").astype(str).tolist()
    clean_labels = labels[valid_mask].to_numpy()
    unique_classes = np.unique(clean_labels)
    if len(unique_classes) < 2 or len(unique_classes) > 30:
        return {}

    vec = CountVectorizer(
        ngram_range=(1, 2),
        max_features=cfg.max_vocab_for_log_odds,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X = vec.fit_transform(clean_texts)
    except ValueError:
        return {}

    vocab = np.array(vec.get_feature_names_out())
    bg_counts = np.asarray(X.sum(axis=0)).ravel().astype(float)
    alpha = bg_counts / max(bg_counts.sum(), 1.0) * cfg.dirichlet_alpha_scale + 1e-3
    alpha_0 = float(alpha.sum())

    results: Dict[str, List[Dict[str, Any]]] = {}
    for cls in unique_classes:
        cls_mask = clean_labels == cls
        c_i = np.asarray(X[cls_mask].sum(axis=0)).ravel().astype(float)
        c_rest = bg_counts - c_i
        n_i = float(c_i.sum())
        n_rest = float(c_rest.sum())

        p_i = (c_i + alpha) / (n_i + alpha_0)
        p_rest = (c_rest + alpha) / (n_rest + alpha_0)

        log_odds = np.log(p_i / (1.0 - p_i + 1e-12)) - np.log(p_rest / (1.0 - p_rest + 1e-12))
        var = 1.0 / (c_i + alpha) + 1.0 / (c_rest + alpha)
        z_scores = log_odds / np.sqrt(var)

        top_idx = np.argsort(z_scores)[::-1][: cfg.top_k_ngrams]
        results[str(cls)] = [
            {
                "ngram": str(vocab[idx]),
                "z_score": round(float(z_scores[idx]), 3),
                "class_count": int(c_i[idx]),
                "other_count": int(c_rest[idx]),
            }
            for idx in top_idx
        ]
    return results


def run_text_profile(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    if cfg.text_col not in train_df.columns:
        raise KeyError(f"Column '{cfg.text_col}' not found in train_df: {list(train_df.columns)}")

    train_stats = audit_text_series(train_df[cfg.text_col], cfg.quantiles)
    test_stats = (
        audit_text_series(test_df[cfg.text_col], cfg.quantiles)
        if test_df is not None and cfg.text_col in test_df.columns
        else None
    )
    vocab_overlap = (
        compute_vocab_overlap(train_df[cfg.text_col], test_df[cfg.text_col])
        if test_df is not None and cfg.text_col in test_df.columns
        else None
    )
    log_odds_ngrams = (
        compute_discriminative_ngrams_log_odds(train_df[cfg.text_col], train_df[cfg.target_col], cfg)
        if cfg.target_col and cfg.target_col in train_df.columns
        else {}
    )

    report = {
        "text_col": cfg.text_col,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "vocab_overlap": vocab_overlap,
        "top_discriminative_ngrams_by_class": log_odds_ngrams,
    }

    print("=" * 72)
    print(f"[TEXT PROFILE] Column: '{cfg.text_col}'")
    print(f"Train char quantiles: {train_stats['char_len_quantiles']}")
    print(f"Train word quantiles: {train_stats['word_len_quantiles']}")
    print(f"Train empty/whitespace rows: {train_stats['empty_or_whitespace_only_count']}")
    print(f"Train noise densities: {train_stats['noise_densities']}")
    if vocab_overlap:
        print(f"Train-Test Vocab Overlap & OOV: {vocab_overlap}")
    if log_odds_ngrams:
        print("-" * 72)
        print("Top Class-Discriminative Unigrams/Bigrams (Log-Odds z-score):")
        for cls_name, items in log_odds_ngrams.items():
            top_str = ", ".join(f"'{it['ngram']}' (z={it['z_score']})" for it in items[:5])
            print(f"  Class [{cls_name}]: {top_str}")
    print("=" * 72)

    if cfg.output_json:
        Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    pos_Pool = ["excellent", "fast", "delivery", "great", "quality", "recommended", "promo", "5star"]
    neg_Pool = ["broken", "scam", "refund", "terrible", "slow", "damaged", "complaint", "fake"]

    tr_texts, tr_labels = [], []
    for i in range(160):
        y = 1 if i % 2 == 0 else 0
        pool = pos_Pool if y == 1 else neg_Pool
        words = rng.choice(pool, size=rng.integers(5, 14), replace=True).tolist()
        if i % 10 == 0:
            words.append("https://example.com/item")
        if i % 15 == 0:
            words.append("<b>HTML</b>")
        tr_texts.append(" ".join(words))
        tr_labels.append(y)
    tr_texts[0] = "   "  # whitespace-only test case

    te_texts = []
    for i in range(60):
        words = rng.choice(pos_Pool + neg_Pool + ["unseen_test_token_xyz"], size=8, replace=True).tolist()
        te_texts.append(" ".join(words))

    with tempfile.TemporaryDirectory() as tmpdir:
        tr_df = pd.DataFrame({"id": np.arange(len(tr_texts)), "text": tr_texts, "target": tr_labels})
        te_df = pd.DataFrame({"id": np.arange(len(te_texts)), "text": te_texts})
        out_json = os.path.join(tmpdir, "text_profile.json")
        cfg = CFG(text_col="text", target_col="target", output_json=out_json)
        report = run_text_profile(tr_df, te_df, cfg)
        assert report["train_stats"]["empty_or_whitespace_only_count"] == 1
        assert report["vocab_overlap"]["test_oov_type_rate"] > 0.0
        assert "0" in report["top_discriminative_ngrams_by_class"]
        assert "1" in report["top_discriminative_ngrams_by_class"]
        assert os.path.exists(out_json)
        print("[SMOKE TEST PASSED] quick_profile_text.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast competition text profiler.")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=CFG.text_col)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        cfg = CFG(
            train_path=args.train_path,
            test_path=args.test_path,
            text_col=args.text_col,
            target_col=args.target_col,
            output_json=args.output_json,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_text_profile(tr, te, cfg)
