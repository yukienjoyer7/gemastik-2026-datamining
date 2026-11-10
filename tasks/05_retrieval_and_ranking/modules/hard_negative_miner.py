"""
Task 05 — Hard Negative Miner with False-Negative Margin Filtering
==================================================================
Standalone module implementing `HardNegativeMiner` for iterative contrastive retrieval:
  - Mines top-K hardest non-positive documents per query from dense/sparse similarity matrices
  - Filters out suspected unlabeled positives (False Negatives) whenever
    `sim(q, d) > margin * max_{d+ in Pos(q)} sim(q, d+)` (keeps `sim <= margin * pos_sim`)

Usage:
  python3 hard_negative_miner.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

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
    exp_name: str = "hard_negative_miner"
    seed: int = 42
    top_k_negatives: int = 5
    false_neg_margin: float = 0.95


class HardNegativeMiner:
    """
    Mines hard negatives from a dense/sparse embedding index while mitigating False Negatives.
    In real-world retrieval datasets (Learning Equality, MS MARCO, Shopee), many top-ranked
    unlabeled items are actually true positives. Any candidate whose similarity exceeds
    `margin * max_positive_similarity` is filtered out (`sim <= margin * pos_sim` retained).
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


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    q_embs = rng.normal(size=(6, 16)).astype(np.float32)
    d_embs = rng.normal(size=(24, 16)).astype(np.float32)
    # Inject a near-duplicate false negative for Q0 at D10
    d_embs[0] = q_embs[0]
    d_embs[10] = q_embs[0] * 0.999

    q_ids = [f"Q{i}" for i in range(6)]
    d_ids = [f"D{i}" for i in range(24)]
    pos_map = {f"Q{i}": {f"D{i}"} for i in range(6)}

    miner = HardNegativeMiner(top_k=cfg.top_k_negatives, margin=cfg.false_neg_margin)
    mined = miner.mine(q_embs, d_embs, q_ids, d_ids, pos_map)
    assert "D10" not in mined["Q0"], "Near-identical false negative D10 should be filtered out"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        counts = np.array([len(v) for v in mined.values()], dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", counts)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", counts)
        pd.DataFrame(
            [{"query_id": k, "hard_negatives": " ".join(v)} for k, v in mined.items()]
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] HardNegativeMiner mined {len(mined['Q0'])} negatives/query | "
        f"False-negative D10 filtered={'D10' not in mined['Q0']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Hard Negative Miner Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
