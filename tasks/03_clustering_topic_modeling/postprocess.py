#!/usr/bin/env python3
r"""
Task 03 — Clustering & Topic Modeling: Balanced Size-Constrained Post-Processing
================================================================================
Standalone post-processing pipeline providing:
  1. **1-Step Self-Training Centroid Refinement (DEC / Sinkhorn sharpening)**:
     Sharpens soft cluster assignment probabilities $p_{ic} \propto p_{ic}^\gamma / (\sum_j p_{jc})^\alpha$
     to pull high-confidence points toward cluster cores before final assignment.
  2. **Exact Capacity-Constrained / Balanced Cluster Assignment**:
     When a competition specifies uniform cluster sizes ($N / K$) or a known prior
     distribution $\pi = (\pi_1, \dots, \pi_K)$, standard `argmax` often produces
     lopsided clusters. We solve the minimum-cost bipartite matching problem via
     `scipy.optimize.linear_sum_assignment` (for exact assignment up to $N \le 1500$)
     or a fast Sinkhorn + greedy margin-priority capacity assigner for large $N$.
  3. Standardized output contract: saves `oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     and `sub_<exp_name>.csv`.

Usage:
    python postprocess.py --oof-path oof_tier2_bertopic.npy --test-probs-path test_tier2_bertopic.npy
    python postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class CFG:
    """Configuration for Task 03 Size-Constrained Post-Processing."""

    exp_name: str = "tier2_clustering_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_clustering.npy"
    test_probs_path: str = "test_tier1_clustering.npy"
    output_dir: str = "."
    id_col: str = "id"
    target_col: str = "cluster"

    sharpen_power: float = 1.5
    sinkhorn_iters: int = 15
    exact_hungarian_max_n: int = 1200
    smoke_test: bool = False


def self_training_sharpen(
    probs: np.ndarray,
    target_prior: Optional[np.ndarray] = None,
    power: float = 1.5,
    n_iters: int = 15,
) -> np.ndarray:
    """
    1-step Deep Embedded Clustering (DEC) + Sinkhorn-Knopp iterative scaling so that:
      - rows sum to 1 (valid sample probability distribution)
      - columns match the desired cluster size prior `target_prior` (uniform by default).
    """
    n_samples, k_clusters = probs.shape
    if target_prior is None:
        col_targets = np.full(k_clusters, n_samples / float(k_clusters), dtype=np.float64)
    else:
        prior = np.asarray(target_prior, dtype=np.float64)
        col_targets = (prior / prior.sum()) * n_samples

    Q = np.power(np.clip(probs.astype(np.float64), 1e-12, 1.0), power)
    for _ in range(n_iters):
        # Row normalization
        Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
        # Column scaling toward target prior counts
        col_sums = np.maximum(Q.sum(axis=0, keepdims=True), 1e-12)
        Q *= (col_targets.reshape(1, -1) / col_sums)

    Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
    return Q.astype(np.float32)


def balanced_capacity_assignment(
    probs: np.ndarray,
    target_prior: Optional[np.ndarray] = None,
    exact_max_n: int = 1200,
) -> np.ndarray:
    r"""
    Assigns $N$ samples to $K$ clusters respecting exact integer capacities $c_1, \dots, c_K$
    ($\sum_k c_k = N$) to maximize total log-probability / similarity:
      - For $N \le \text{exact\_max\_n}$, expands columns into $c_k$ slots and solves the
        exact global optimum via `scipy.optimize.linear_sum_assignment`.
      - For larger $N$, sorts samples by confidence margin ($p_{(1)} - p_{(2)}$) and greedily
        assigns each sample to its highest-probability cluster with remaining capacity.
    """
    n_samples, k_clusters = probs.shape
    if target_prior is None:
        prior = np.full(k_clusters, 1.0 / k_clusters, dtype=np.float64)
    else:
        prior = np.asarray(target_prior, dtype=np.float64)
        prior = prior / prior.sum()

    raw_caps = prior * n_samples
    capacities = np.floor(raw_caps).astype(np.int64)
    remainder = int(n_samples - capacities.sum())
    if remainder > 0:
        frac_order = np.argsort(raw_caps - capacities)[::-1]
        for idx in frac_order[:remainder]:
            capacities[idx] += 1

    if n_samples <= exact_max_n:
        # Replicate each cluster column k by capacities[k] times -> (N, N) assignment matrix
        slot_to_cluster = np.repeat(np.arange(k_clusters, dtype=np.int64), capacities)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        cost_matrix = -log_probs[:, slot_to_cluster]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        labels = np.zeros(n_samples, dtype=np.int64)
        labels[row_ind] = slot_to_cluster[col_ind]
        return labels

    # Fast greedy margin-priority assignment for large N
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    sample_order = np.argsort(margins)[::-1]
    rem_caps = capacities.copy()
    labels = np.full(n_samples, -1, dtype=np.int64)

    for i in sample_order:
        pref_clusters = np.argsort(probs[i])[::-1]
        for c in pref_clusters:
            if rem_caps[c] > 0:
                labels[i] = int(c)
                rem_caps[c] -= 1
                break

    return labels


def run_postprocess(
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    test_ids: np.ndarray,
    cfg: CFG,
    true_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_oof_labels = np.argmax(oof_probs, axis=1)
    raw_counts = np.bincount(raw_oof_labels, minlength=oof_probs.shape[1])

    oof_sharp = self_training_sharpen(
        oof_probs, power=cfg.sharpen_power, n_iters=cfg.sinkhorn_iters
    )
    test_sharp = self_training_sharpen(
        test_probs, power=cfg.sharpen_power, n_iters=cfg.sinkhorn_iters
    )

    bal_oof_labels = balanced_capacity_assignment(oof_sharp, exact_max_n=cfg.exact_hungarian_max_n)
    bal_test_labels = balanced_capacity_assignment(test_sharp, exact_max_n=cfg.exact_hungarian_max_n)
    bal_counts = np.bincount(bal_oof_labels, minlength=oof_probs.shape[1])

    print("=" * 72)
    print("TASK 03 BALANCED CLUSTER SIZE & SELF-TRAINING POST-PROCESSING")
    print("=" * 72)
    print(f"  Raw Argmax Cluster Counts      : {raw_counts.tolist()}")
    print(f"  Balanced Capacity Counts       : {bal_counts.tolist()}")

    if true_labels is not None:
        raw_ari = float(adjusted_rand_score(true_labels, raw_oof_labels))
        bal_ari = float(adjusted_rand_score(true_labels, bal_oof_labels))
        raw_nmi = float(normalized_mutual_info_score(true_labels, raw_oof_labels))
        bal_nmi = float(normalized_mutual_info_score(true_labels, bal_oof_labels))
        print(f"  Raw Argmax NMI / ARI           : NMI={raw_nmi:.4f} | ARI={raw_ari:.4f}")
        print(f"  Balanced Post-Proc NMI / ARI   : NMI={bal_nmi:.4f} | ARI={bal_ari:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_sharp)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_sharp)
    pd.DataFrame({cfg.id_col: test_ids, cfg.target_col: bal_test_labels}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )
    return oof_sharp, test_sharp, bal_oof_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 03 Balanced Cluster Post-Processing")
    parser.add_argument("--oof-path", type=str, default=CFG.oof_path)
    parser.add_argument("--test-probs-path", type=str, default=CFG.test_probs_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        oof_path=args.oof_path,
        test_probs_path=args.test_probs_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(42)
        n_oof, n_test, k = 120, 40, 4
        true_labels = np.arange(n_oof) % k
        logits = rng.normal(0, 0.9, size=(n_oof, k))
        for i, c in enumerate(true_labels):
            logits[i, c] += 1.6
        # Inject strong cluster-0 bias so naive argmax creates imbalanced clusters
        logits[:, 0] += 1.1
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        oof_probs = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

        te_logits = rng.normal(0, 1.0, size=(n_test, k))
        te_exp = np.exp(te_logits - te_logits.max(axis=1, keepdims=True))
        test_probs = (te_exp / te_exp.sum(axis=1, keepdims=True)).astype(np.float32)
        test_ids = np.array([f"te_{i}" for i in range(n_test)])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            oof_sharp, test_sharp, bal_labels = run_postprocess(
                oof_probs, test_probs, test_ids, cfg, true_labels=true_labels
            )
            counts = np.bincount(bal_labels, minlength=k)
            assert np.all(counts == n_oof // k), f"Expected exact uniform counts {n_oof // k}, got {counts}"
            print("[SUCCESS] Task 03 balanced cluster post-processing smoke test passed.")
    else:
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        p_te = pathlib.Path(cfg.test_path)
        if p_te.exists():
            te_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
            test_ids = te_df[cfg.id_col].values
        else:
            test_ids = np.arange(len(test_probs))
        run_postprocess(oof_probs, test_probs, test_ids, cfg)


if __name__ == "__main__":
    main()
