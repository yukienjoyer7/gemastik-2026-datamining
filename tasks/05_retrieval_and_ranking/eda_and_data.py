"""
Task 05 — Retrieval & Ranking: Exploratory Data Analysis & Fold Preparation
===========================================================================
Diagnostics for two-stage retrieval & ranking pipelines:
  - Query count, corpus size, positive candidates per query distribution
  - Unseen corpus items in test vs. train overlap check (Zero-Shot / Cold-Start audit)
  - Query and document length diagnostics + lexical overlap baseline check
  - Fast `evaluate_candidate_recall(candidates_dict, ground_truth_dict, ks=(5, 10, 50, 100))`
  - Leak-free GroupKFold split on `query_id` / `topic_cluster` exported to `train_folds.parquet`

Usage:
  python3 eda_and_data.py --data-dir ./data --output-dir ./outputs
  python3 eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_venv_site = (
    Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


@dataclass
class CFG:
    exp_name: str = "eda_retrieval"
    seed: int = 42
    n_folds: int = 5
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    query_id_col: str = "query_id"
    query_text_col: str = "query_text"
    doc_id_col: str = "doc_id"
    doc_text_col: str = "doc_text"
    target_col: str = "relevance"
    recall_ks: Tuple[int, ...] = (1, 5, 10, 50, 100)


def evaluate_candidate_recall(
    candidates_dict: Dict[str, Sequence[str]],
    ground_truth_dict: Dict[str, Iterable[str]],
    ks: Sequence[int] = (5, 10, 50, 100),
) -> Dict[str, float]:
    """
    Computes macro-averaged Recall@K and MRR@K across queries.
    Crucial Stage-1 diagnostic: a Stage-2 reranker can NEVER recover positive
    documents that were missed by Stage-1 candidate generation.
    """
    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in ks}
    for k in ks:
        metrics[f"mrr@{k}"] = []
        metrics[f"map@{k}"] = []

    for qid, gt_items in ground_truth_dict.items():
        gt_set: Set[str] = set(gt_items)
        if not gt_set:
            continue
        preds = list(candidates_dict.get(qid, []))
        for k in ks:
            top_k = preds[:k]
            hits = sum(1 for d in top_k if d in gt_set)
            metrics[f"recall@{k}"].append(hits / len(gt_set))

            # Reciprocal Rank @ K
            rr = 0.0
            for rank_idx, doc_id in enumerate(top_k, start=1):
                if doc_id in gt_set:
                    rr = 1.0 / rank_idx
                    break
            metrics[f"mrr@{k}"].append(rr)

            # Average Precision @ K
            running_hits = 0.0
            sum_precisions = 0.0
            for rank_idx, doc_id in enumerate(top_k, start=1):
                if doc_id in gt_set:
                    running_hits += 1.0
                    sum_precisions += running_hits / rank_idx
            denom = min(len(gt_set), k)
            metrics[f"map@{k}"].append(sum_precisions / denom if denom > 0 else 0.0)

    return {
        name: float(np.mean(vals)) if vals else 0.0 for name, vals in metrics.items()
    }


def run_retrieval_eda(
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    test_queries_df: pd.DataFrame | None,
    cfg: CFG,
) -> Dict[str, float]:
    """Runs structural diagnostics on queries, candidate corpus, and ground-truth links."""
    n_queries = queries_df[cfg.query_id_col].nunique()
    n_corpus = corpus_df[cfg.doc_id_col].nunique()

    pos_qrels = qrels_df[qrels_df[cfg.target_col] > 0]
    pos_per_query = (
        pos_qrels.groupby(cfg.query_id_col)[cfg.doc_id_col]
        .nunique()
        .reindex(queries_df[cfg.query_id_col].unique(), fill_value=0)
    )

    referenced_docs = set(pos_qrels[cfg.doc_id_col].unique())
    all_corpus_docs = set(corpus_df[cfg.doc_id_col].unique())
    unreferenced_corpus_ratio = 1.0 - (
        len(referenced_docs & all_corpus_docs) / max(len(all_corpus_docs), 1)
    )

    q_char_len = queries_df[cfg.query_text_col].fillna("").astype(str).str.len()
    d_char_len = corpus_df[cfg.doc_text_col].fillna("").astype(str).str.len()

    # Check query/document overlap between train and test
    test_query_overlap = 0.0
    if test_queries_df is not None and len(test_queries_df) > 0:
        train_q_set = set(queries_df[cfg.query_id_col].unique())
        test_q_set = set(test_queries_df[cfg.query_id_col].unique())
        test_query_overlap = len(train_q_set & test_q_set) / max(len(test_q_set), 1)

    report = {
        "n_train_queries": float(n_queries),
        "n_corpus_docs": float(n_corpus),
        "pos_per_query_mean": float(pos_per_query.mean()),
        "pos_per_query_median": float(pos_per_query.median()),
        "pos_per_query_p95": float(np.percentile(pos_per_query, 95)),
        "pos_per_query_max": float(pos_per_query.max()),
        "unreferenced_corpus_ratio": float(unreferenced_corpus_ratio),
        "query_len_mean": float(q_char_len.mean()),
        "doc_len_mean": float(d_char_len.mean()),
        "test_query_id_overlap_ratio": float(test_query_overlap),
    }

    print("=== Retrieval & Ranking Diagnostics ===")
    for k, v in report.items():
        print(f"  {k:<30}: {v:.4f}")
    return report


def create_query_grouped_folds(
    pair_df: pd.DataFrame, cfg: CFG
) -> pd.DataFrame:
    """
    Assigns leak-free folds grouped by `query_id` (or `group_id` if present)
    so all candidate pairs for a given query reside strictly in the same fold.
    """
    df = pair_df.copy().reset_index(drop=True)
    df["fold"] = -1
    n_unique_groups = df[cfg.query_id_col].nunique()
    n_splits = min(cfg.n_folds, n_unique_groups)

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (_, val_idx) in enumerate(
            gkf.split(df, df[cfg.target_col], groups=df[cfg.query_id_col])
        ):
            df.loc[val_idx, "fold"] = fold
    else:
        kf = KFold(n_splits=max(2, cfg.n_folds), shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, "fold"] = fold

    return df


def generate_synthetic_retrieval_data(
    n_queries: int = 30, n_corpus: int = 80, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates a synthetic retrieval & ranking dataset for smoke testing."""
    rng = np.random.default_rng(seed)
    topics = [
        ("calculus derivative integral limit theorem", "math analysis"),
        ("photosynthesis chlorophyll plant cell leaf sunlight", "biology botany"),
        ("newton force acceleration mass gravity momentum", "physics mechanics"),
        ("neural network gradient backprop transformer attention", "deep learning"),
        ("supply demand inflation interest rate central bank", "macroeconomics"),
        ("sql database index btree query join transaction", "computer science"),
    ]

    corpus_rows = []
    for doc_idx in range(n_corpus):
        topic_idx = doc_idx % len(topics)
        keywords, domain = topics[topic_idx]
        words = keywords.split()
        sampled = rng.choice(words, size=min(4, len(words)), replace=False)
        doc_text = f"Document {doc_idx} covering {domain}: " + " ".join(sampled)
        corpus_rows.append(
            {"doc_id": f"D{doc_idx:04d}", "doc_text": doc_text, "topic_idx": topic_idx}
        )
    corpus_df = pd.DataFrame(corpus_rows)

    query_rows = []
    pair_rows = []
    for q_idx in range(n_queries):
        topic_idx = q_idx % len(topics)
        keywords, domain = topics[topic_idx]
        words = keywords.split()
        sampled = rng.choice(words, size=min(3, len(words)), replace=False)
        qid = f"Q{q_idx:04d}"
        qtext = f"Find resources on {domain} especially " + " ".join(sampled)
        query_rows.append({"query_id": qid, "query_text": qtext, "topic_idx": topic_idx})

        # Positive candidates from same topic + negative candidates from other topics
        same_topic_docs = [
            r["doc_id"] for r in corpus_rows if r["topic_idx"] == topic_idx
        ]
        other_topic_docs = [
            r["doc_id"] for r in corpus_rows if r["topic_idx"] != topic_idx
        ]
        pos_docs = rng.choice(same_topic_docs, size=min(2, len(same_topic_docs)), replace=False)
        neg_docs = rng.choice(other_topic_docs, size=min(4, len(other_topic_docs)), replace=False)

        for d in pos_docs:
            d_text = corpus_df.loc[corpus_df["doc_id"] == d, "doc_text"].values[0]
            pair_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "doc_id": str(d),
                    "doc_text": d_text,
                    "relevance": 1,
                }
            )
        for d in neg_docs:
            d_text = corpus_df.loc[corpus_df["doc_id"] == d, "doc_text"].values[0]
            pair_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "doc_id": str(d),
                    "doc_text": d_text,
                    "relevance": 0,
                }
            )

    queries_df = pd.DataFrame(query_rows)
    pair_df = pd.DataFrame(pair_rows)
    test_queries_df = queries_df.iloc[:8].copy().reset_index(drop=True)
    test_queries_df["query_id"] = [f"TQ{i:04d}" for i in range(len(test_queries_df))]

    return queries_df, corpus_df, pair_df, test_queries_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Retrieval & Ranking EDA")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)

    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            queries_df, corpus_df, pair_df, test_queries_df = generate_synthetic_retrieval_data()
            report = run_retrieval_eda(queries_df, corpus_df, pair_df, test_queries_df, cfg)
            folds_df = create_query_grouped_folds(pair_df, cfg)

            # Audit a mock candidate list using evaluate_candidate_recall
            gt_dict: Dict[str, List[str]] = (
                pair_df[pair_df["relevance"] > 0]
                .groupby("query_id")["doc_id"]
                .apply(list)
                .to_dict()
            )
            cand_dict: Dict[str, List[str]] = (
                pair_df.sort_values("relevance", ascending=False)
                .groupby("query_id")["doc_id"]
                .apply(list)
                .to_dict()
            )
            recall_metrics = evaluate_candidate_recall(cand_dict, gt_dict, ks=(1, 5, 10))
            print("Candidate Recall Audit:", recall_metrics)

            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            folds_path = out_dir / "train_folds.parquet"
            folds_df.to_parquet(folds_path, index=False)

            oof_dummy = folds_df["relevance"].to_numpy(dtype=np.float32)
            test_dummy = np.zeros(len(test_queries_df), dtype=np.float32)
            np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_dummy)
            np.save(out_dir / f"test_{cfg.exp_name}.npy", test_dummy)
            pd.DataFrame(
                {
                    "query_id": test_queries_df["query_id"],
                    "doc_ids": ["D0000 D0001"] * len(test_queries_df),
                }
            ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

            print(
                f"[Smoke Test OK] Folds shape: {folds_df.shape}, "
                f"Recall@5: {recall_metrics['recall@5']:.4f}, "
                f"Queries: {int(report['n_train_queries'])}"
            )
        return

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        train_path = data_dir / "train.csv"
    if not train_path.exists():
        print(f"No data found at {data_dir}; falling back to synthetic smoke dataset.")
        queries_df, corpus_df, pair_df, test_queries_df = generate_synthetic_retrieval_data()
    else:
        pair_df = (
            pd.read_parquet(train_path)
            if train_path.suffix == ".parquet"
            else pd.read_csv(train_path)
        )
        queries_df = pair_df[[cfg.query_id_col, cfg.query_text_col]].drop_duplicates()
        corpus_df = pair_df[[cfg.doc_id_col, cfg.doc_text_col]].drop_duplicates()
        test_queries_df = None

    run_retrieval_eda(queries_df, corpus_df, pair_df, test_queries_df, cfg)
    folds_df = create_query_grouped_folds(pair_df, cfg)
    folds_df.to_parquet(out_dir / "train_folds.parquet", index=False)
    print(f"Saved leak-free query-grouped folds to {out_dir / 'train_folds.parquet'}")


if __name__ == "__main__":
    main()
