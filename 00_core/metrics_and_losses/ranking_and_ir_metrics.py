#!/usr/bin/env python3
"""
ranking_and_ir_metrics.py
=========================
Fast NumPy implementations of Ranking, Information Retrieval, and Recommendation metrics:
  - `apk(actual, predicted, k=10)`
  - `map_at_k(actual_list, predicted_list, k=10)`
  - `ndcg_at_k(actual_list_or_matrix, predicted_list_or_scores, k=10)`
  - `mrr_at_k(actual_list, predicted_list, k=10)`
  - `recall_at_k(actual_list, predicted_list, k=10)`
  - `hit_rate_at_k(actual_list, predicted_list, k=10)`
  - `fbeta_retrieval_score(actual_list, predicted_list, beta=2.0)`

Usage:
  python3 00_core/metrics_and_losses/ranking_and_ir_metrics.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Union

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np


@dataclass
class CFG:
    k: int = 10
    beta: float = 2.0


def apk(actual: Sequence, predicted: Sequence, k: int = 10) -> float:
    """
    Average Precision at K (AP@K) for a single query.
    Deduplicates `predicted` while preserving rank order.
    """
    if len(actual) == 0:
        return 0.0
    actual_set: Set = set(actual)
    pred_k = list(predicted)[:k]

    score = 0.0
    num_hits = 0.0
    seen: Set = set()
    for i, p in enumerate(pred_k):
        if p in actual_set and p not in seen:
            seen.add(p)
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return float(score / min(len(actual_set), k))


def map_at_k(actual_list: Sequence[Sequence], predicted_list: Sequence[Sequence], k: int = 10) -> float:
    """Mean Average Precision at K (MAP@K) across all queries."""
    if len(actual_list) == 0:
        return 0.0
    return float(np.mean([apk(a, p, k=k) for a, p in zip(actual_list, predicted_list)]))


def ndcg_at_k(
    actual: Union[Sequence[Sequence], np.ndarray],
    predicted: Union[Sequence[Sequence], np.ndarray],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K).
    Supports two input formats:
      1) Set/List retrieval format: `actual` is a list of relevant item IDs per query,
         `predicted` is a ranked list of predicted item IDs per query.
      2) Dense relevance matrix format: `actual` (N, M) ground-truth relevance grades and
         `predicted` (N, M) predicted continuous scores.
    """
    if isinstance(actual, np.ndarray) and isinstance(predicted, np.ndarray) and actual.ndim == 2:
        N, M = actual.shape
        k_eff = min(k, M)
        order = np.argsort(-predicted, axis=1)[:, :k_eff]
        gains = np.take_along_axis(actual, order, axis=1)
        discounts = np.log2(np.arange(2, k_eff + 2, dtype=np.float64))[None, :]
        dcg = np.sum((np.power(2.0, gains) - 1.0) / discounts, axis=1)

        ideal_order = np.argsort(-actual, axis=1)[:, :k_eff]
        ideal_gains = np.take_along_axis(actual, ideal_order, axis=1)
        idcg = np.sum((np.power(2.0, ideal_gains) - 1.0) / discounts, axis=1)
        ndcg = np.where(idcg > 0.0, dcg / idcg, 0.0)
        return float(np.mean(ndcg))

    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    scores: List[float] = []
    for act, pred in zip(actual, predicted):
        act_set = set(act)
        if not act_set:
            scores.append(0.0)
            continue
        pred_k = list(pred)[:k]
        seen = set()
        hits = np.zeros(len(pred_k), dtype=np.float64)
        for i, item in enumerate(pred_k):
            if item in act_set and item not in seen:
                seen.add(item)
                hits[i] = 1.0
        dcg = float(np.sum(hits * discounts[: len(pred_k)]))
        n_ideal = min(len(act_set), k)
        idcg = float(np.sum(discounts[:n_ideal]))
        scores.append(dcg / idcg if idcg > 0.0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mrr_at_k(actual_list: Sequence[Sequence], predicted_list: Sequence[Sequence], k: int = 10) -> float:
    """Mean Reciprocal Rank at K (MRR@K)."""
    rr: List[float] = []
    for act, pred in zip(actual_list, predicted_list):
        act_set = set(act)
        rec = 0.0
        for rank, item in enumerate(list(pred)[:k], start=1):
            if item in act_set:
                rec = 1.0 / float(rank)
                break
        rr.append(rec)
    return float(np.mean(rr)) if rr else 0.0


def recall_at_k(actual_list: Sequence[Sequence], predicted_list: Sequence[Sequence], k: int = 10) -> float:
    """Mean Recall@K across queries."""
    recalls: List[float] = []
    for act, pred in zip(actual_list, predicted_list):
        act_set = set(act)
        if not act_set:
            continue
        pred_set = set(list(pred)[:k])
        recalls.append(len(act_set & pred_set) / float(len(act_set)))
    return float(np.mean(recalls)) if recalls else 0.0


def hit_rate_at_k(actual_list: Sequence[Sequence], predicted_list: Sequence[Sequence], k: int = 10) -> float:
    """HitRate@K (fraction of queries with at least 1 relevant item in top-K)."""
    hits: List[float] = []
    for act, pred in zip(actual_list, predicted_list):
        act_set = set(act)
        pred_set = set(list(pred)[:k])
        hits.append(1.0 if len(act_set & pred_set) > 0 else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def fbeta_retrieval_score(
    actual_list: Sequence[Sequence],
    predicted_list: Sequence[Sequence],
    beta: float = 2.0,
) -> float:
    """
    Mean sample-wise F-beta score over retrieved item sets (e.g. F2 score in Learning Equality / Shop retrieval).
    """
    b2 = beta ** 2
    scores: List[float] = []
    for act, pred in zip(actual_list, predicted_list):
        act_set = set(act)
        pred_set = set(pred)
        if not act_set and not pred_set:
            scores.append(1.0)
            continue
        if not act_set or not pred_set:
            scores.append(0.0)
            continue
        tp = float(len(act_set & pred_set))
        prec = tp / float(len(pred_set))
        rec = tp / float(len(act_set))
        denom = b2 * prec + rec
        scores.append(((1.0 + b2) * prec * rec / denom) if denom > 0.0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def run_smoke_test() -> None:
    actual = [
        [1, 2, 3],
        [10, 20],
        [7],
        [99],
    ]
    predicted = [
        [1, 4, 2, 5, 3],  # hits at rank 1, 3, 5
        [5, 10, 8, 20],   # hits at rank 2, 4
        [7, 1, 2],        # hit at rank 1
        [1, 2, 3, 4],     # miss
    ]

    map5 = map_at_k(actual, predicted, k=5)
    ndcg5_list = ndcg_at_k(actual, predicted, k=5)
    mrr5 = mrr_at_k(actual, predicted, k=5)
    rec5 = recall_at_k(actual, predicted, k=5)
    hr5 = hit_rate_at_k(actual, predicted, k=5)
    f2 = fbeta_retrieval_score(actual, predicted, beta=2.0)

    # Also test dense matrix NDCG@K
    rel_mat = np.array([[3.0, 2.0, 0.0, 1.0], [0.0, 1.0, 2.0, 0.0]])
    score_mat = np.array([[2.9, 2.1, 0.1, 0.8], [0.2, 0.5, 1.9, 0.1]])
    ndcg3_dense = ndcg_at_k(rel_mat, score_mat, k=3)

    assert 0.50 < map5 < 1.0
    assert 0.50 < ndcg5_list < 1.0
    assert np.isclose(mrr5, (1.0 + 0.5 + 1.0 + 0.0) / 4.0)
    assert np.isclose(rec5, 0.75)
    assert np.isclose(hr5, 0.75)
    assert 0.50 < f2 < 1.0
    assert np.isclose(ndcg3_dense, 1.0)

    print(
        f"[RANKING METRICS @5] MAP@5={map5:.5f} | NDCG@5(list)={ndcg5_list:.5f} | "
        f"NDCG@3(dense)={ndcg3_dense:.5f} | MRR@5={mrr5:.5f} | Recall@5={rec5:.5f} | "
        f"HitRate@5={hr5:.5f} | F2={f2:.5f}"
    )
    print("[SMOKE TEST PASSED] ranking_and_ir_metrics.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorized IR and ranking metrics.")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke_test()
