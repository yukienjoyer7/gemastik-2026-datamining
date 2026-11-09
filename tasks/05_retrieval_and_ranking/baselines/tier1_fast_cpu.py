"""
Task 05 — Tier 1 Fast CPU Baseline: Two-Stage Sparse Retrieval + GBDT Reranker
==============================================================================
Complete Two-Stage Retrieval & Ranking pipeline running purely on CPU:
  - Stage 1 (Candidate Generation):
      * Built-in vectorized Okapi BM25 (zero external `rank_bm25` dependency)
      * Word N-gram TF-IDF cosine similarity
      * Character N-gram TF-IDF cosine similarity
      * Reciprocal Rank Fusion (RRF) combining all 3 retrievers -> Top-K candidates
      * Reports Stage-1 Recall@K ceiling before Stage-2 training
  - Stage 2 (Query-Candidate Feature Engineering & GBDT Reranking):
      * Lexical & statistical pair features (BM25, Word/Char TF-IDF cosine, Jaccard,
        token coverage, length difference & ratio)
      * Query-contextualized relative features (within-query z-score, min-max,
        reciprocal rank, score gap to top-1 & top-3 candidate)
      * Out-Of-Fold GBDT reranker (`LightGBM` if available, else `HistGradientBoostingClassifier`)
      * Evaluates OOF MAP@K, NDCG@K, MRR@K, and Recall@K
      * Exports `oof_<exp_name>.npy`, `test_<exp_name>.npy`, and `sub_<exp_name>.csv`

Usage:
  python3 tier1_fast_cpu.py --data-dir ./data --output-dir ./outputs
  python3 tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import re
import site
import sys
import tempfile
from collections import Counter
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class CFG:
    exp_name: str = "tier1_bm25_gbdt"
    seed: int = 42
    n_folds: int = 5
    stage1_top_k: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    eval_k: int = 10
    data_dir: str = "./data"
    output_dir: str = "./outputs"


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


class VectorizedOkapiBM25:
    """Self-contained pure-Python/NumPy Okapi BM25 retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.empty(0, dtype=np.float32)
        self.doc_lens: np.ndarray = np.empty(0, dtype=np.float32)
        self.avgdl: float = 1.0
        self.doc_term_freqs: List[Dict[int, int]] = []
        self.n_docs: int = 0

    def fit(self, corpus_texts: Sequence[str]) -> "VectorizedOkapiBM25":
        self.n_docs = len(corpus_texts)
        df_counts: Counter[int] = Counter()
        self.doc_term_freqs = []
        doc_lengths = []

        for text in corpus_texts:
            tokens = tokenize(text)
            doc_lengths.append(len(tokens))
            tf_local: Counter[int] = Counter()
            for tok in tokens:
                tid = self.vocab.setdefault(tok, len(self.vocab))
                tf_local[tid] += 1
            for tid in tf_local:
                df_counts[tid] += 1
            self.doc_term_freqs.append(dict(tf_local))

        self.doc_lens = np.asarray(doc_lengths, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.n_docs > 0 else 1.0

        n_vocab = len(self.vocab)
        self.idf = np.zeros(n_vocab, dtype=np.float32)
        for tid, df in df_counts.items():
            self.idf[tid] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
        return self

    def score_all(self, query_text: str) -> np.ndarray:
        q_tokens = tokenize(query_text)
        q_tids = [self.vocab[t] for t in q_tokens if t in self.vocab]
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not q_tids or self.n_docs == 0:
            return scores

        norm_factors = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / max(self.avgdl, 1e-6)))
        for d_idx, tf_map in enumerate(self.doc_term_freqs):
            dl_norm = norm_factors[d_idx]
            s = 0.0
            for tid in q_tids:
                tf = tf_map.get(tid, 0)
                if tf > 0:
                    s += float(self.idf[tid]) * (tf * (self.k1 + 1.0)) / (tf + dl_norm)
            scores[d_idx] = s
        return scores


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60, top_n: int = 20
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_idx, did in enumerate(ranked, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank_idx)
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ordered[:top_n]


def evaluate_ranking_metrics(
    df: pd.DataFrame, score_col: str, k: int = 10
) -> Dict[str, float]:
    map_list, ndcg_list, recall_list, mrr_list = [], [], [], []

    for _, group in df.groupby("query_id", sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        rels = ordered["relevance"].to_numpy(dtype=np.float32)
        total_pos = float(np.sum(rels > 0))
        if total_pos == 0:
            continue

        top_rels = rels[:k]
        # Recall@K
        recall_list.append(float(np.sum(top_rels > 0)) / total_pos)

        # MRR@K
        pos_positions = np.where(top_rels > 0)[0]
        mrr_list.append(1.0 / (pos_positions[0] + 1.0) if len(pos_positions) > 0 else 0.0)

        # MAP@K
        hits = 0.0
        sum_prec = 0.0
        for idx, r in enumerate(top_rels, start=1):
            if r > 0:
                hits += 1.0
                sum_prec += hits / idx
        map_list.append(sum_prec / min(total_pos, float(k)))

        # NDCG@K
        discounts = np.log2(np.arange(2, len(top_rels) + 2))
        dcg = float(np.sum((2.0**top_rels - 1.0) / discounts))
        ideal_rels = np.sort(rels)[::-1][:k]
        idcg = float(np.sum((2.0**ideal_rels - 1.0) / np.log2(np.arange(2, len(ideal_rels) + 2))))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"MAP@{k}": float(np.mean(map_list)) if map_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        f"MRR@{k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
        f"Recall@{k}": float(np.mean(recall_list)) if recall_list else 0.0,
    }


def build_stage1_and_pair_features(
    queries_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    qrels_map: Dict[str, Set[str]] | None,
    cfg: CFG,
) -> pd.DataFrame:
    """
    Runs Stage-1 multi-retriever candidate generation (BM25 + Word TF-IDF + Char TF-IDF -> RRF)
    and builds rich Stage-2 query-candidate pair features.
    """
    doc_ids = corpus_df["doc_id"].astype(str).tolist()
    doc_texts = corpus_df["doc_text"].fillna("").astype(str).tolist()
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    bm25 = VectorizedOkapiBM25(k1=cfg.bm25_k1, b=cfg.bm25_b).fit(doc_texts)
    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(doc_texts)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit(
        doc_texts
    )

    doc_word_mat = word_vec.transform(doc_texts)
    doc_char_mat = char_vec.transform(doc_texts)
    doc_token_sets = [set(tokenize(t)) for t in doc_texts]

    q_ids = queries_df["query_id"].astype(str).tolist()
    q_texts = queries_df["query_text"].fillna("").astype(str).tolist()
    q_word_mat = word_vec.transform(q_texts)
    q_char_mat = char_vec.transform(q_texts)

    word_sim_all = (q_word_mat @ doc_word_mat.T).toarray()
    char_sim_all = (q_char_mat @ doc_char_mat.T).toarray()

    rows = []
    for q_idx, (qid, qtext) in enumerate(zip(q_ids, q_texts)):
        bm25_scores = bm25.score_all(qtext)
        w_sims = word_sim_all[q_idx]
        c_sims = char_sim_all[q_idx]

        bm25_top = [doc_ids[i] for i in np.argsort(-bm25_scores)[: cfg.stage1_top_k]]
        word_top = [doc_ids[i] for i in np.argsort(-w_sims)[: cfg.stage1_top_k]]
        char_top = [doc_ids[i] for i in np.argsort(-c_sims)[: cfg.stage1_top_k]]

        fused_cands = reciprocal_rank_fusion(
            [bm25_top, word_top, char_top], k=cfg.rrf_k, top_n=cfg.stage1_top_k
        )

        # Ensure positive docs are included in training candidate pool if qrels_map is provided
        cand_map = dict(fused_cands)
        if qrels_map is not None:
            for pos_did in qrels_map.get(qid, set()):
                if pos_did in doc_id_to_idx and pos_did not in cand_map:
                    cand_map[pos_did] = 0.0

        q_tokens = set(tokenize(qtext))
        q_len = max(len(qtext), 1)
        q_tok_len = max(len(q_tokens), 1)

        for did, rrf_score in cand_map.items():
            d_idx = doc_id_to_idx[did]
            d_tokens = doc_token_sets[d_idx]
            d_text = doc_texts[d_idx]
            inter = len(q_tokens & d_tokens)
            union = max(len(q_tokens | d_tokens), 1)

            rel = 0
            if qrels_map is not None and did in qrels_map.get(qid, set()):
                rel = 1

            rows.append(
                {
                    "query_id": qid,
                    "doc_id": did,
                    "relevance": rel,
                    "rrf_score": float(rrf_score),
                    "bm25_score": float(bm25_scores[d_idx]),
                    "word_tfidf_cos": float(w_sims[d_idx]),
                    "char_tfidf_cos": float(c_sims[d_idx]),
                    "jaccard_overlap": float(inter / union),
                    "query_token_coverage": float(inter / q_tok_len),
                    "len_ratio": float(len(d_text) / q_len),
                    "len_diff": float(abs(len(d_text) - q_len)),
                }
            )

    pair_df = pd.DataFrame(rows)

    # Add query-contextualized relative features (z-score, rank, gap to top-1 & top-3)
    score_cols = ["rrf_score", "bm25_score", "word_tfidf_cos", "char_tfidf_cos", "jaccard_overlap"]
    grouped = pair_df.groupby("query_id", sort=False)
    for col in score_cols:
        q_mean = grouped[col].transform("mean")
        q_std = grouped[col].transform("std").fillna(0.0)
        q_max = grouped[col].transform("max")
        ranks = grouped[col].rank(ascending=False, method="min")

        pair_df[f"{col}_q_zscore"] = (pair_df[col] - q_mean) / (q_std + 1e-6)
        pair_df[f"{col}_q_rr"] = 1.0 / ranks
        pair_df[f"{col}_gap_top1"] = pair_df[col] - q_max

        top3_mean = (
            pair_df[["query_id", col]]
            .sort_values(["query_id", col], ascending=[True, False])
            .groupby("query_id", sort=False)
            .head(3)
            .groupby("query_id", sort=False)[col]
            .mean()
        )
        pair_df[f"{col}_gap_top3"] = pair_df[col] - pair_df["query_id"].map(top3_mean)

    return pair_df


def train_oof_gbdt_reranker(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    """Trains an Out-Of-Fold GBDT reranker grouped by `query_id`."""
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)

    if "fold" not in train_df.columns:
        train_df = train_df.copy()
        train_df["fold"] = 0
        gkf = GroupKFold(n_splits=min(cfg.n_folds, train_df["query_id"].nunique()))
        for fold, (_, val_idx) in enumerate(
            gkf.split(train_df, train_df["relevance"], groups=train_df["query_id"])
        ):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    unique_folds = sorted(train_df["fold"].unique())
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    for fold in unique_folds:
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        X_tr = train_df.loc[tr_mask, feature_cols].to_numpy(dtype=np.float32)
        y_tr = train_df.loc[tr_mask, "relevance"].to_numpy(dtype=np.int32)
        X_va = train_df.loc[va_mask, feature_cols].to_numpy(dtype=np.float32)

        if HAS_LGBM:
            model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=15,
                random_state=cfg.seed + fold,
                verbosity=-1,
            )
            model.fit(X_tr, y_tr)
        else:
            model = HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                random_state=cfg.seed + fold,
            )
            model.fit(X_tr, y_tr)

        oof_preds[va_mask] = model.predict_proba(X_va)[:, 1].astype(np.float32)
        test_preds += model.predict_proba(X_test)[:, 1].astype(np.float32) / len(unique_folds)

    return oof_preds, test_preds


def _generate_synthetic_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    domains = [
        ("calculus derivative integral limit series convergence", "mathematics"),
        ("chlorophyll photosynthesis leaf sunlight glucose plant", "botany"),
        ("velocity acceleration force momentum gravity orbit", "physics"),
        ("transformer attention backprop gradient embedding layer", "deep learning"),
        ("inflation interest central bank fiscal monetary bond", "economics"),
    ]
    corpus_rows = []
    for i in range(40):
        kw, dom = domains[i % len(domains)]
        toks = rng.choice(kw.split(), size=4, replace=False)
        corpus_rows.append(
            {"doc_id": f"D{i:03d}", "doc_text": f"Reference material on {dom}: " + " ".join(toks)}
        )
    corpus_df = pd.DataFrame(corpus_rows)

    q_rows = []
    qrels: Dict[str, Set[str]] = {}
    for q_idx in range(20):
        dom_idx = q_idx % len(domains)
        kw, dom = domains[dom_idx]
        toks = rng.choice(kw.split(), size=3, replace=False)
        qid = f"Q{q_idx:03d}"
        q_rows.append({"query_id": qid, "query_text": f"Query about {dom} " + " ".join(toks)})
        pos_docs = {f"D{i:03d}" for i in range(40) if i % len(domains) == dom_idx}
        qrels[qid] = set(list(pos_docs)[:3])

    queries_df = pd.DataFrame(q_rows)
    test_queries_df = queries_df.iloc[:6].copy().reset_index(drop=True)
    test_queries_df["query_id"] = [f"TQ{i:03d}" for i in range(len(test_queries_df))]
    return queries_df, corpus_df, qrels, test_queries_df


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    if smoke_test:
        cfg.stage1_top_k = 12
        cfg.eval_k = 5
        queries_df, corpus_df, qrels, test_queries_df = _generate_synthetic_dataset(cfg.seed)
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            raw = pd.read_parquet(folds_path)
            queries_df = raw[["query_id", "query_text"]].drop_duplicates().reset_index(drop=True)
            corpus_df = raw[["doc_id", "doc_text"]].drop_duplicates().reset_index(drop=True)
            qrels = (
                raw[raw["relevance"] > 0]
                .groupby("query_id")["doc_id"]
                .apply(set)
                .to_dict()
            )
            test_queries_df = queries_df.iloc[:5].copy().reset_index(drop=True)
        else:
            queries_df, corpus_df, qrels, test_queries_df = _generate_synthetic_dataset(cfg.seed)

    train_pairs = build_stage1_and_pair_features(queries_df, corpus_df, qrels, cfg)
    test_pairs = build_stage1_and_pair_features(test_queries_df, corpus_df, None, cfg)

    exclude_cols = {"query_id", "doc_id", "relevance", "fold"}
    feature_cols = [c for c in train_pairs.columns if c not in exclude_cols]

    stage1_metrics = evaluate_ranking_metrics(train_pairs, "rrf_score", k=cfg.eval_k)
    oof_preds, test_preds = train_oof_gbdt_reranker(train_pairs, test_pairs, feature_cols, cfg)
    train_pairs["gbdt_score"] = oof_preds
    test_pairs["gbdt_score"] = test_preds
    stage2_metrics = evaluate_ranking_metrics(train_pairs, "gbdt_score", k=cfg.eval_k)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_preds)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_preds)

    sub_rows = []
    for qid, group in test_pairs.sort_values("gbdt_score", ascending=False).groupby(
        "query_id", sort=False
    ):
        top_docs = " ".join(group["doc_id"].head(cfg.eval_k).tolist())
        sub_rows.append({"query_id": qid, "doc_ids": top_docs})
    pd.DataFrame(sub_rows).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Tier1 CPU Retrieval+GBDT] Features={len(feature_cols)} | "
        f"Stage1 RRF MAP@{cfg.eval_k}={stage1_metrics[f'MAP@{cfg.eval_k}']:.4f} -> "
        f"Stage2 GBDT MAP@{cfg.eval_k}={stage2_metrics[f'MAP@{cfg.eval_k}']:.4f}, "
        f"NDCG@{cfg.eval_k}={stage2_metrics[f'NDCG@{cfg.eval_k}']:.4f}, "
        f"OOF shape={oof_preds.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Tier 1 BM25 + GBDT Reranker")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            run_pipeline(cfg, smoke_test=True)
    else:
        run_pipeline(cfg, smoke_test=False)


if __name__ == "__main__":
    main()
