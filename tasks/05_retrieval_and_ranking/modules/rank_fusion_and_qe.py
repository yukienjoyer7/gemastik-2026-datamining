"""
Task 05 — Reciprocal Rank Fusion (RRF) & Alpha Query Expansion (AQE)
====================================================================
Standalone module implementing:
  1. `reciprocal_rank_fusion`: Parameter-free multi-retriever rank fusion
     (`score(d) = sum_m w_m / (k + rank_m(d))`) across BM25, TF-IDF, and Bi-Encoders.
  2. `alpha_query_expansion`: Query-side neighborhood blending (AQE / alpha-QE)
     that refines query embeddings using similarity-weighted top-K retrieved documents.

Usage:
  python3 rank_fusion_and_qe.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


@dataclass
class CFG:
    exp_name: str = "rank_fusion_and_qe"
    seed: int = 42
    rrf_k: int = 60
    alpha_qe: float = 0.5


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


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    q_embs = rng.normal(size=(4, 16)).astype(np.float32)
    d_embs = rng.normal(size=(10, 16)).astype(np.float32)

    fused = reciprocal_rank_fusion(
        [["D1", "D2", "D3"], ["D2", "D4", "D1"], ["D2", "D1", "D5"]],
        k=cfg.rrf_k,
        weights=[1.0, 1.2, 0.8],
    )
    expanded_single = alpha_query_expansion(q_embs[0], d_embs[:4], alpha=cfg.alpha_qe)
    expanded_batch = alpha_query_expansion(
        q_embs, np.broadcast_to(d_embs[:4], (4, 4, 16)), alpha=cfg.alpha_qe
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", expanded_batch)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", expanded_single)
        pd.DataFrame(fused, columns=["doc_id", "rrf_score"]).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] RRF top1={fused[0][0]} (score={fused[0][1]:.4f}) | "
        f"AQE norm={float(np.linalg.norm(expanded_single)):.4f}, batch_shape={expanded_batch.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 RRF & Alpha Query Expansion Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
