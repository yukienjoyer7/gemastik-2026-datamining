"""
Task 05 — Retrieval & Ranking Reusable Modules
==============================================
Core components for competition retrieval & reranking pipelines:
  1. `MultipleNegativesRankingLoss`: Symmetric/asymmetric InfoNCE loss with temperature `tau`
     and optional explicit mined hard negatives.
  2. `HardNegativeMiner`: Dense cosine top-K negative miner with False-Negative Margin Filtering
     (drops candidates with `sim > margin * pos_sim` to avoid penalizing unlabeled positives).
  3. `reciprocal_rank_fusion`: Parameter-free multi-retriever fusion (BM25 + TF-IDF + Bi-Encoder).
  4. `alpha_query_expansion`: Query-side neighborhood blending (AQE / alpha-QE).
  5. `build_query_context_features`: Stage-2 GBDT query-level relative features (z-score, min-max,
     within-query rank, gap to top-1 and top-3 candidate).

Usage:
  python3 retrieval_and_reranking.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

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
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    exp_name: str = "retrieval_modules"
    seed: int = 42
    temperature: float = 0.05
    rrf_k: int = 60
    alpha_qe: float = 0.5
    false_neg_margin: float = 0.95
    top_k_negatives: int = 5


class MultipleNegativesRankingLoss(nn.Module):
    """
    InfoNCE / Multiple Negatives Ranking Loss for Bi-Encoder contrastive training.
    Supports both in-batch negatives `(batch_size, dim)` and explicit hard negatives
    `(batch_size, n_hard_negs, dim)`.
    """

    def __init__(self, temperature: float = 0.05, symmetric: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        query_embs: torch.Tensor,
        pos_doc_embs: torch.Tensor,
        hard_neg_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = F.normalize(query_embs, p=2, dim=-1)
        p = F.normalize(pos_doc_embs, p=2, dim=-1)

        # In-batch similarity matrix: (B, B)
        logits = torch.matmul(q, p.transpose(0, 1)) / self.temperature

        if hard_neg_embs is not None:
            n = F.normalize(hard_neg_embs, p=2, dim=-1)
            if n.dim() == 2:
                # Shared pool of hard negatives: (M, D) -> (B, M)
                neg_logits = torch.matmul(q, n.transpose(0, 1)) / self.temperature
            else:
                # Per-query hard negatives: (B, K, D) -> (B, K)
                neg_logits = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([logits, neg_logits], dim=1)

        labels = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        loss_q2d = F.cross_entropy(logits, labels)

        if self.symmetric and hard_neg_embs is None:
            loss_d2q = F.cross_entropy(logits.transpose(0, 1), labels)
            return 0.5 * (loss_q2d + loss_d2q)
        return loss_q2d


class HardNegativeMiner:
    """
    Mines hard negatives from a dense embedding index while mitigating False Negatives.
    In real-world retrieval datasets (Learning Equality, MS MARCO, Shopee), many top-ranked
    unlabeled items are actually true positives. Any candidate whose similarity exceeds
    `margin * max_positive_similarity` is filtered out as a suspected false negative.
    """

    def __init__(self, top_k: int = 10, margin: float = 0.95) -> None:
        self.top_k = top_k
        self.margin = margin

    def mine(
        self,
        query_embs: np.ndarray,
        doc_embs: np.ndarray,
        query_ids: Sequence[str],
        doc_ids: Sequence[str],
        positive_map: Dict[str, Set[str]],
    ) -> Dict[str, List[str]]:
        q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)
        d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
        sim_matrix = q_norm @ d_norm.T

        doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
        mined_negatives: Dict[str, List[str]] = {}

        for q_idx, qid in enumerate(query_ids):
            sims = sim_matrix[q_idx]
            pos_set = positive_map.get(qid, set())
            pos_indices = [doc_id_to_idx[d] for d in pos_set if d in doc_id_to_idx]

            if pos_indices:
                pos_sim_ref = float(np.max(sims[pos_indices]))
                threshold = self.margin * pos_sim_ref
            else:
                threshold = 1.0

            ranked_indices = np.argsort(-sims)
            hard_negs: List[str] = []
            for d_idx in ranked_indices:
                did = doc_ids[int(d_idx)]
                if did in pos_set:
                    continue
                if float(sims[d_idx]) > threshold:
                    # Suspected false negative: skip
                    continue
                hard_negs.append(did)
                if len(hard_negs) >= self.top_k:
                    break
            mined_negatives[qid] = hard_negs

        return mined_negatives


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF): combines multiple ranked candidate ID lists
    via `score(d) = sum_m w_m / (k + rank_m(d))`.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused_scores: Dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank_idx, doc_id in enumerate(ranked, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + float(w) / (k + rank_idx)

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        return ordered[:top_n]
    return ordered


def alpha_query_expansion(
    query_emb: np.ndarray,
    top_doc_embs: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Alpha Query Expansion (AQE): blends a normalized query embedding with the
    similarity-weighted centroid of its top retrieved documents.
    `query_emb`: shape (D,) or (B, D)
    `top_doc_embs`: shape (K, D) or (B, K, D)
    """
    single = query_emb.ndim == 1
    if single:
        query_emb = query_emb[None, :]
        top_doc_embs = top_doc_embs[None, :, :]

    q_norm = query_emb / (np.linalg.norm(query_emb, axis=-1, keepdims=True) + 1e-12)
    d_norm = top_doc_embs / (np.linalg.norm(top_doc_embs, axis=-1, keepdims=True) + 1e-12)

    # Cosine weights: (B, K, 1)
    sims = np.clip(np.sum(q_norm[:, None, :] * d_norm, axis=-1, keepdims=True), 0.0, None)
    weights = np.power(sims, alpha)
    neighbor_mean = np.sum(weights * d_norm, axis=1)
    expanded = q_norm + neighbor_mean
    expanded = expanded / (np.linalg.norm(expanded, axis=-1, keepdims=True) + 1e-12)

    return expanded[0] if single else expanded


def build_query_context_features(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    query_col: str = "query_id",
) -> pd.DataFrame:
    """
    Enriches a query-candidate DataFrame with query-contextualized relative features:
      - `<col>_q_zscore`: within-query standardized score
      - `<col>_q_minmax`: within-query [0, 1] scaled score
      - `<col>_q_rank`: within-query descending rank (1 = highest)
      - `<col>_q_rr`: within-query reciprocal rank `1 / rank`
      - `<col>_diff_top1`: difference from best candidate of the same query
      - `<col>_diff_top3_mean`: difference from mean of top-3 candidates of the same query
    """
    out = df.copy()
    grouped = out.groupby(query_col, sort=False)

    for col in score_cols:
        q_mean = grouped[col].transform("mean")
        q_std = grouped[col].transform("std").fillna(0.0)
        q_min = grouped[col].transform("min")
        q_max = grouped[col].transform("max")

        out[f"{col}_q_zscore"] = (out[col] - q_mean) / (q_std + 1e-6)
        out[f"{col}_q_minmax"] = (out[col] - q_min) / (q_max - q_min + 1e-6)

        ranks = grouped[col].rank(ascending=False, method="min")
        out[f"{col}_q_rank"] = ranks
        out[f"{col}_q_rr"] = 1.0 / ranks
        out[f"{col}_diff_top1"] = out[col] - q_max

        # Top-3 mean per query
        top3_means = (
            out[[query_col, col]]
            .sort_values([query_col, col], ascending=[True, False])
            .groupby(query_col, sort=False)
            .head(3)
            .groupby(query_col, sort=False)[col]
            .mean()
        )
        out[f"{col}_diff_top3_mean"] = out[col] - out[query_col].map(top3_means)

    return out


def _run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # 1. Test MultipleNegativesRankingLoss
    loss_fn = MultipleNegativesRankingLoss(temperature=cfg.temperature, symmetric=True)
    q_t = torch.randn(8, 16, requires_grad=True)
    p_t = q_t.detach() + 0.1 * torch.randn(8, 16)
    n_t = torch.randn(8, 3, 16)
    loss_inbatch = loss_fn(q_t, p_t)
    loss_hard = loss_fn(q_t, p_t, n_t)
    loss_hard.backward()

    # 2. Test HardNegativeMiner
    q_embs = rng.normal(size=(6, 16)).astype(np.float32)
    d_embs = rng.normal(size=(20, 16)).astype(np.float32)
    q_ids = [f"Q{i}" for i in range(6)]
    d_ids = [f"D{i}" for i in range(20)]
    pos_map = {f"Q{i}": {f"D{i}"} for i in range(6)}
    miner = HardNegativeMiner(top_k=cfg.top_k_negatives, margin=cfg.false_neg_margin)
    mined = miner.mine(q_embs, d_embs, q_ids, d_ids, pos_map)

    # 3. Test RRF & Alpha Query Expansion
    fused = reciprocal_rank_fusion([["D1", "D2", "D3"], ["D2", "D4", "D1"]], k=cfg.rrf_k)
    expanded_q = alpha_query_expansion(q_embs[0], d_embs[:4], alpha=cfg.alpha_qe)

    # 4. Test Query Context Features
    sample_df = pd.DataFrame(
        {
            "query_id": ["Q0", "Q0", "Q0", "Q1", "Q1", "Q1"],
            "bm25_score": [3.2, 1.8, 0.4, 5.1, 4.9, 1.0],
            "tfidf_cos": [0.75, 0.42, 0.11, 0.88, 0.80, 0.25],
        }
    )
    feat_df = build_query_context_features(sample_df, ["bm25_score", "tfidf_cos"])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", feat_df["bm25_score_q_zscore"].to_numpy())
        np.save(out_dir / f"test_{cfg.exp_name}.npy", expanded_q)
        pd.DataFrame(fused, columns=["doc_id", "rrf_score"]).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] MNRL loss(in-batch)={loss_inbatch.item():.4f}, "
        f"loss(hard-neg)={loss_hard.item():.4f}, "
        f"mined_Q0={len(mined['Q0'])}, RRF_top1={fused[0][0]}, "
        f"context_features_shape={feat_df.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Retrieval & Reranking Modules")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
