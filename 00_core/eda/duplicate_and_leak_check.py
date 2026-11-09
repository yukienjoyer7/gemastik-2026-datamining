#!/usr/bin/env python3
"""
duplicate_and_leak_check.py
===========================
Exact hash deduplication + character n-gram (3, 5) TF-IDF cosine similarity near-duplicate detector.
Audits:
  1. Train-Train Exact Duplicates:
     - Total duplicate rows, duplicate clusters, and conflicting label groups (noisy labels!)
  2. Train-Test Exact Leaks:
     - Test rows that appear verbatim in Train (direct label lookup opportunity or data leak)
  3. Train-Train Near-Duplicates (char n-gram (3, 5) TF-IDF cosine similarity >= threshold):
     - Near-identical texts/rows in Train and whether their targets conflict
  4. Train-Test Near-Duplicates (char n-gram (3, 5) TF-IDF cosine similarity >= threshold):
     - Test rows with near-identical matches in Train

Usage:
  python3 00_core/eda/duplicate_and_leak_check.py --train-path train.parquet --test-path test.parquet --text-col text --target-col target
  python3 00_core/eda/duplicate_and_leak_check.py --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import warnings
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

WS_RE = re.compile(r"\s+")


@dataclass
class CFG:
    train_path: str = "train.parquet"
    test_path: Optional[str] = "test.parquet"
    text_col: Optional[str] = "text"
    feature_cols: Optional[List[str]] = None
    target_col: Optional[str] = "target"
    id_col: Optional[str] = "id"
    ngram_range: Tuple[int, int] = (3, 5)
    max_tfidf_features: int = 30000
    sim_threshold: float = 0.90
    top_k_neighbors: int = 3
    output_json: Optional[str] = None


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def build_canonical_text(df: pd.DataFrame, cfg: CFG) -> pd.Series:
    if cfg.text_col and cfg.text_col in df.columns:
        return (
            df[cfg.text_col]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(lambda s: WS_RE.sub(" ", s).strip())
        )
    cols = cfg.feature_cols
    if not cols:
        exclude = {cfg.target_col, cfg.id_col, "fold"}
        cols = [c for c in df.columns if c not in exclude]
    return (
        df[cols]
        .fillna("NULL")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
        .map(lambda s: WS_RE.sub(" ", s).strip())
    )


def hash_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: hashlib.sha256(x.encode("utf-8", errors="ignore")).hexdigest())


def check_exact_duplicates_and_leaks(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    tr_canon: pd.Series,
    te_canon: Optional[pd.Series],
    cfg: CFG,
) -> Dict[str, Any]:
    tr_hashes = hash_series(tr_canon)
    dup_mask = tr_hashes.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())

    conflicting_groups = 0
    total_dup_groups = 0
    if n_dup_rows > 0:
        tmp = pd.DataFrame({"hash": tr_hashes[dup_mask]})
        if cfg.target_col and cfg.target_col in train_df.columns:
            tmp["target"] = train_df.loc[dup_mask, cfg.target_col].values
            grp_nunique = tmp.groupby("hash")["target"].nunique(dropna=False)
            total_dup_groups = int(len(grp_nunique))
            conflicting_groups = int((grp_nunique > 1).sum())
        else:
            total_dup_groups = int(tmp["hash"].nunique())

    exact_test_leaks = 0
    exact_leak_test_indices: List[int] = []
    if test_df is not None and te_canon is not None:
        te_hashes = hash_series(te_canon)
        tr_hash_set = set(tr_hashes.unique())
        leak_mask = te_hashes.isin(tr_hash_set)
        exact_test_leaks = int(leak_mask.sum())
        exact_leak_test_indices = np.where(leak_mask)[0][:25].tolist()

    return {
        "train_exact_duplicate_rows": n_dup_rows,
        "train_exact_duplicate_groups": total_dup_groups,
        "train_conflicting_label_groups": conflicting_groups,
        "train_test_exact_leak_rows": exact_test_leaks,
        "sample_exact_leak_test_indices": exact_leak_test_indices,
    }


def check_near_duplicates_char_tfidf(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    tr_canon: pd.Series,
    te_canon: Optional[pd.Series],
    cfg: CFG,
) -> Dict[str, Any]:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_tfidf_features,
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(tr_canon.tolist())

    k_tr = min(cfg.top_k_neighbors + 1, X_tr.shape[0])
    nn = NearestNeighbors(n_neighbors=k_tr, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X_tr)

    # 1) Train-Train near duplicates (excluding self at index i == j and exact similarity ~1.0 if desired, or all >= sim_threshold)
    dists_tr, idxs_tr = nn.kneighbors(X_tr)
    sims_tr = 1.0 - dists_tr

    has_target = bool(cfg.target_col and cfg.target_col in train_df.columns)
    y_tr = train_df[cfg.target_col].to_numpy() if has_target else None

    seen_pairs = set()
    near_tr_pairs: List[Dict[str, Any]] = []
    conflicting_near_tr_pairs = 0

    for i in range(X_tr.shape[0]):
        for rank in range(1, k_tr):
            j = int(idxs_tr[i, rank])
            if i == j:
                continue
            sim = float(sims_tr[i, rank])
            if sim < cfg.sim_threshold:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            conflict = False
            if y_tr is not None:
                conflict = bool(y_tr[pair_key[0]] != y_tr[pair_key[1]])
                if conflict:
                    conflicting_near_tr_pairs += 1

            near_tr_pairs.append({
                "train_idx_a": pair_key[0],
                "train_idx_b": pair_key[1],
                "cosine_sim": round(sim, 5),
                "label_conflict": conflict,
            })

    # 2) Train-Test near duplicates
    near_te_matches: List[Dict[str, Any]] = []
    if test_df is not None and te_canon is not None and len(te_canon) > 0:
        X_te = vec.transform(te_canon.tolist())
        dists_te, idxs_te = nn.kneighbors(X_te, n_neighbors=1)
        sims_te = 1.0 - dists_te[:, 0]
        best_tr_idx = idxs_te[:, 0]

        for te_i, (tr_j, sim) in enumerate(zip(best_tr_idx, sims_te)):
            if float(sim) >= cfg.sim_threshold:
                item: Dict[str, Any] = {
                    "test_idx": int(te_i),
                    "matched_train_idx": int(tr_j),
                    "cosine_sim": round(float(sim), 5),
                }
                if y_tr is not None:
                    item["matched_train_target"] = str(y_tr[int(tr_j)])
                near_te_matches.append(item)

    return {
        "sim_threshold": cfg.sim_threshold,
        "char_ngram_range": list(cfg.ngram_range),
        "train_train_near_duplicate_pairs_count": len(near_tr_pairs),
        "train_train_near_duplicate_label_conflicts": conflicting_near_tr_pairs,
        "sample_train_train_near_duplicates": near_tr_pairs[:15],
        "train_test_near_duplicate_rows_count": len(near_te_matches),
        "sample_train_test_near_duplicates": near_te_matches[:15],
    }


def run_duplicate_and_leak_audit(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: CFG,
) -> Dict[str, Any]:
    tr_canon = build_canonical_text(train_df, cfg)
    te_canon = build_canonical_text(test_df, cfg) if test_df is not None else None

    exact_report = check_exact_duplicates_and_leaks(train_df, test_df, tr_canon, te_canon, cfg)
    near_report = check_near_duplicates_char_tfidf(train_df, test_df, tr_canon, te_canon, cfg)

    report = {
        "exact_audit": exact_report,
        "near_duplicate_audit": near_report,
    }

    print("=" * 72)
    print("[DUPLICATE & LEAKAGE AUDIT]")
    print(f"Train Exact Duplicate Rows:   {exact_report['train_exact_duplicate_rows']} "
          f"(Groups: {exact_report['train_exact_duplicate_groups']}, "
          f"Conflicting Labels: {exact_report['train_conflicting_label_groups']})")
    print(f"Train-Test Exact Leak Rows:   {exact_report['train_test_exact_leak_rows']}")
    print("-" * 72)
    print(f"Char {cfg.ngram_range} TF-IDF Cosine >= {cfg.sim_threshold}:")
    print(f"  Train-Train Near-Duplicate Pairs: {near_report['train_train_near_duplicate_pairs_count']} "
          f"(Label Conflicts: {near_report['train_train_near_duplicate_label_conflicts']})")
    print(f"  Train-Test Near-Duplicate Rows:   {near_report['train_test_near_duplicate_rows_count']}")
    print("=" * 72)

    if cfg.output_json:
        Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def run_smoke_test() -> None:
    tr_texts = [
        "Paket sampai dengan cepat dan kondisi barang sangat bagus sekali!",
        "Paket sampai dengan cepat dan kondisi barang sangat bagus sekali!",  # exact dup, conflicting label
        "Paket sampai dengan cepat dan kondisi barang sangat bagus sekali!!",  # near dup
        "Aplikasi sering force close saat melakukan pembayaran QRIS di kasir.",
        "Pengiriman lambat sekali kurir tidak ramah dan barang penyok.",
        "Harga murah kualitas original mantap jiwa terima kasih seller.",
    ] + [f"Unique training sentence number {i} with distinct vocabulary {i*17}." for i in range(40)]

    tr_labels = [1, 0, 1, 0, 0, 1] + [i % 2 for i in range(40)]

    te_texts = [
        "Aplikasi sering force close saat melakukan pembayaran QRIS di kasir.",  # exact leak from train
        "Pengiriman lambat sekali kurir tidak ramah dan barang penyok!",  # near leak from train
        "Completely unseen test review about battery life on laptop.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tr_df = pd.DataFrame({"id": np.arange(len(tr_texts)), "text": tr_texts, "target": tr_labels})
        te_df = pd.DataFrame({"id": np.arange(len(te_texts)), "text": te_texts})
        out_json = os.path.join(tmpdir, "dup_leak.json")

        cfg = CFG(text_col="text", target_col="target", sim_threshold=0.90, output_json=out_json)
        report = run_duplicate_and_leak_audit(tr_df, te_df, cfg)

        assert report["exact_audit"]["train_exact_duplicate_rows"] == 2
        assert report["exact_audit"]["train_conflicting_label_groups"] == 1
        assert report["exact_audit"]["train_test_exact_leak_rows"] == 1
        assert report["near_duplicate_audit"]["train_train_near_duplicate_pairs_count"] >= 2
        assert report["near_duplicate_audit"]["train_test_near_duplicate_rows_count"] >= 2
        assert os.path.exists(out_json)
        print("[SMOKE TEST PASSED] duplicate_and_leak_check.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact & char n-gram TF-IDF near-duplicate leak checker.")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--text-col", type=str, default=CFG.text_col)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--sim-threshold", type=float, default=CFG.sim_threshold)
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
            sim_threshold=args.sim_threshold,
            output_json=args.output_json,
        )
        tr = load_dataframe(cfg.train_path)
        te = load_dataframe(cfg.test_path) if cfg.test_path and os.path.exists(cfg.test_path) else None
        run_duplicate_and_leak_audit(tr, te, cfg)
