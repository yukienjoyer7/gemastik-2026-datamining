"""
Task 05 — Post-Processing: Dynamic Candidate Cutoff & Mutual k-NN Score Boosting
================================================================================
Post-processing utilities for variable-length retrieval & ranking competitions
(e.g., Kaggle Learning Equality F2, Shopee Price Match F1, Product Matching):
  1. `mutual_knn_score_boost`: Boosts similarity/probability for pairs `(q, d)` that
     are reciprocal nearest neighbors (if `d` is in top-K of `q` AND `q` is in top-K of `d`).
  2. `dynamic_candidate_selector`: Variable-length cutoff per query accepting candidates
     where `p_i >= tau_abs` AND `p_i >= tau_rel * max_j(p_j)`, bounded by `[min_k, max_k]`.
  3. `optimize_dynamic_cutoffs_fbeta`: Out-Of-Fold grid search over `(tau_abs, tau_rel)`
     directly maximizing competition F_beta (e.g., F2-score).

Usage:
  python3 postprocess.py --smoke-test
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


@dataclass
class CFG:
    exp_name: str = "retrieval_postprocess"
    seed: int = 42
    beta: float = 2.0
    min_k: int = 1
    max_k: int = 10
    mutual_knn_k: int = 5
    mutual_boost: float = 0.10
    output_dir: str = "./outputs"


def mutual_knn_score_boost(
    sim_matrix: np.ndarray, k: int = 5, boost: float = 0.10
) -> np.ndarray:
    """
    Boosts scores in a `(n_queries, n_docs)` similarity matrix when a query and
    document are mutual top-K neighbors of each other.
    """
    n_q, n_d = sim_matrix.shape
    k_q = min(k, n_d)
    k_d = min(k, n_q)

    q_top = np.argsort(-sim_matrix, axis=1)[:, :k_q]
    d_top = np.argsort(-sim_matrix, axis=0)[:k_d, :]

    q_mask = np.zeros_like(sim_matrix, dtype=bool)
    d_mask = np.zeros_like(sim_matrix, dtype=bool)
    for i in range(n_q):
        q_mask[i, q_top[i]] = True
    for j in range(n_d):
        d_mask[d_top[:, j], j] = True

    mutual_mask = q_mask & d_mask
    boosted = sim_matrix.copy()
    boosted[mutual_mask] += boost
    return boosted


def dynamic_candidate_selector(
    df: pd.DataFrame,
    score_col: str,
    tau_abs: float = 0.35,
    tau_rel: float = 0.60,
    min_k: int = 1,
    max_k: int = 10,
    query_col: str = "query_id",
    doc_col: str = "doc_id",
) -> Dict[str, List[str]]:
    """
    Selects a dynamic number of candidates per query:
      - Accepts candidate `i` if `score_i >= tau_abs` and `score_i >= tau_rel * max_score_for_query`
      - Always returns at least `min_k` and at most `max_k` candidates per query.
    """
    selected: Dict[str, List[str]] = {}
    ordered_df = df.sort_values([query_col, score_col], ascending=[True, False])

    for qid, group in ordered_df.groupby(query_col, sort=False):
        scores = group[score_col].to_numpy(dtype=np.float32)
        docs = group[doc_col].astype(str).tolist()
        if len(docs) == 0:
            selected[str(qid)] = []
            continue

        best_score = float(scores[0])
        keep_mask = (scores >= tau_abs) & (scores >= tau_rel * best_score)
        chosen = [d for d, keep in zip(docs, keep_mask) if keep]

        if len(chosen) < min_k:
            chosen = docs[:min_k]
        elif len(chosen) > max_k:
            chosen = chosen[:max_k]

        selected[str(qid)] = chosen
    return selected


def compute_fbeta_retrieval(
    preds_dict: Dict[str, Sequence[str]],
    gt_dict: Dict[str, Set[str]],
    beta: float = 2.0,
) -> float:
    """Computes macro-averaged retrieval F_beta across queries."""
    beta2 = beta * beta
    f_scores: List[float] = []

    for qid, gt_set in gt_dict.items():
        pred_set = set(preds_dict.get(qid, []))
        if not gt_set and not pred_set:
            f_scores.append(1.0)
            continue
        if not gt_set or not pred_set:
            f_scores.append(0.0)
            continue

        tp = float(len(gt_set & pred_set))
        prec = tp / len(pred_set)
        rec = tp / len(gt_set)
        if prec + rec == 0.0:
            f_scores.append(0.0)
        else:
            f_scores.append((1.0 + beta2) * prec * rec / (beta2 * prec + rec))

    return float(np.mean(f_scores)) if f_scores else 0.0


def optimize_dynamic_cutoffs_fbeta(
    oof_df: pd.DataFrame,
    score_col: str,
    gt_dict: Dict[str, Set[str]],
    cfg: CFG,
) -> Tuple[float, float, float]:
    """Grid searches `(tau_abs, tau_rel)` on OOF predictions to maximize F_beta."""
    best_score = -1.0
    best_abs = 0.30
    best_rel = 0.50

    s_min = float(oof_df[score_col].quantile(0.10))
    s_max = float(oof_df[score_col].quantile(0.90))
    abs_grid = np.linspace(s_min, s_max, 9)
    rel_grid = np.linspace(0.30, 0.90, 7)

    for t_abs in abs_grid:
        for t_rel in rel_grid:
            preds = dynamic_candidate_selector(
                oof_df,
                score_col=score_col,
                tau_abs=float(t_abs),
                tau_rel=float(t_rel),
                min_k=cfg.min_k,
                max_k=cfg.max_k,
            )
            score = compute_fbeta_retrieval(preds, gt_dict, beta=cfg.beta)
            if score > best_score:
                best_score = score
                best_abs = float(t_abs)
                best_rel = float(t_rel)

    return best_abs, best_rel, best_score


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)

    # 1. Mutual k-NN boost test
    sim = rng.uniform(0.1, 0.8, size=(12, 20)).astype(np.float32)
    boosted_sim = mutual_knn_score_boost(sim, k=cfg.mutual_knn_k, boost=cfg.mutual_boost)

    # 2. Synthetic OOF candidate table for F2 dynamic threshold optimization
    rows = []
    gt_dict: Dict[str, Set[str]] = {}
    for q_idx in range(25):
        qid = f"Q{q_idx:03d}"
        n_pos = int(rng.integers(1, 4))
        gt_dict[qid] = {f"D{d}" for d in range(n_pos)}
        for d in range(8):
            is_pos = d < n_pos
            score = float(rng.uniform(0.55, 0.95) if is_pos else rng.uniform(0.05, 0.60))
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": f"D{d}",
                    "relevance": int(is_pos),
                    "prob": score,
                }
            )
    oof_df = pd.DataFrame(rows)

    fixed_top1 = {
        qid: [grp.sort_values("prob", ascending=False)["doc_id"].iloc[0]]
        for qid, grp in oof_df.groupby("query_id")
    }
    base_f2 = compute_fbeta_retrieval(fixed_top1, gt_dict, beta=cfg.beta)
    best_abs, best_rel, tuned_f2 = optimize_dynamic_cutoffs_fbeta(oof_df, "prob", gt_dict, cfg)
    selected_test = dynamic_candidate_selector(
        oof_df, "prob", tau_abs=best_abs, tau_rel=best_rel, min_k=cfg.min_k, max_k=cfg.max_k
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_df["prob"].to_numpy(dtype=np.float32))
        np.save(out_dir / f"test_{cfg.exp_name}.npy", boosted_sim.ravel())
        pd.DataFrame(
            [{"query_id": k, "doc_ids": " ".join(v)} for k, v in selected_test.items()]
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Fixed Top-1 F2={base_f2:.4f} -> Dynamic Cutoff F2={tuned_f2:.4f} "
        f"(tau_abs={best_abs:.3f}, tau_rel={best_rel:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Retrieval Post-Processing")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
