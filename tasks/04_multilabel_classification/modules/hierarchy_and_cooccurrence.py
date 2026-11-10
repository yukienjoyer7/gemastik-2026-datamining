#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Hierarchy Constraint Enforcer & Label Co-occurrence Features
=================================================================================================
Atomic standalone post-processing and meta-feature builders for correlated multilabel tasks:
  1. `HierarchicalConstraintEnforcer`:
     Enforces parent-child implication rules ($\text{child} \implies \text{parent}$, i.e.,
     $P(\text{parent}) \ge P(\text{child})$) via `bottom_up`, `top_down`, or `average` updates,
     plus automatic implication rule discovery from training labels ($P(y_j=1 \mid y_i=1) \ge \tau$).
  2. `LabelCooccurrenceFeatureBuilder`:
     Computes Laplace-smoothed conditional co-occurrence transition matrices
     $C_{ij} = P(y_j = 1 \mid y_i = 1)$ on out-of-fold training folds and projects first-stage
     predicted probabilities $P \in [0, 1]^{N \times L}$ into second-stage co-occurrence
     meta-features ($P C$, max conditional support, and label-cardinality expectations).

Usage:
    python3 tasks/04_multilabel_classification/modules/hierarchy_and_cooccurrence.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


@dataclass
class CFG:
    """Configuration for Hierarchy & Label Co-occurrence Smoke Test."""

    n_samples: int = 64
    num_labels: int = 6
    min_support: int = 5
    implication_threshold: float = 0.98
    laplace_alpha: float = 1.0
    smoke_test: bool = False


class HierarchicalConstraintEnforcer:
    r"""
    Enforces parent-child probability consistency for a list of `(child_idx, parent_idx)`
    implication pairs ($\text{child} \implies \text{parent}$, so $P(\text{parent}) \ge P(\text{child})$).
    """

    def __init__(
        self, edges: Sequence[Tuple[int, int]] = (), mode: str = "bottom_up"
    ) -> None:
        self.edges: List[Tuple[int, int]] = list(edges)
        self.mode = mode

    @classmethod
    def discover_from_labels(
        cls,
        y_true: np.ndarray,
        min_support: int = 10,
        threshold: float = 0.98,
        mode: str = "bottom_up",
    ) -> "HierarchicalConstraintEnforcer":
        """
        Automatically discovers `(child_idx, parent_idx)` edges where `count(child) >= min_support`
        and `P(parent = 1 | child = 1) >= threshold`.
        """
        Y = (np.asarray(y_true, dtype=np.float64) > 0.5).astype(np.float64)
        n_labels = Y.shape[1]
        cooc = Y.T @ Y
        support = Y.sum(axis=0)
        edges: List[Tuple[int, int]] = []
        for child in range(n_labels):
            if support[child] < min_support:
                continue
            for parent in range(n_labels):
                if child == parent or support[parent] < support[child]:
                    continue
                cond_prob = cooc[child, parent] / max(support[child], 1.0)
                if cond_prob >= threshold:
                    edges.append((child, parent))
        return cls(edges=edges, mode=mode)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float64).copy()
        if not self.edges:
            return arr

        def _bottom_up(p: np.ndarray) -> np.ndarray:
            out = p.copy()
            for _ in range(max(1, len(self.edges))):
                for child_idx, parent_idx in self.edges:
                    out[:, parent_idx] = np.maximum(out[:, parent_idx], out[:, child_idx])
            return out

        def _top_down(p: np.ndarray) -> np.ndarray:
            out = p.copy()
            for _ in range(max(1, len(self.edges))):
                for child_idx, parent_idx in self.edges:
                    out[:, child_idx] = np.minimum(out[:, child_idx], out[:, parent_idx])
            return out

        if self.mode == "bottom_up":
            return _bottom_up(arr)
        if self.mode == "top_down":
            return _top_down(arr)
        return 0.5 * _bottom_up(arr) + 0.5 * _top_down(arr)


class LabelCooccurrenceFeatureBuilder:
    r"""
    Computes the Laplace-smoothed conditional label co-occurrence matrix
    $C_{ij} = P(y_j = 1 \mid y_i = 1)$ (with zeroed diagonal $C_{ii} = 0$) and builds
    2nd-stage stacking meta-features from Stage-1 probability predictions $P \in [0, 1]^{N \times L}$:
      - `P @ C`: expected co-occurrence support from other active labels `(N, L)`
      - `max_i (P_{n, i} * C_{i, j})`: strongest single implicating neighbor per label `(N, L)`
      - `row_sum(P)` and `row_entropy(P)`: expected label cardinality & uncertainty `(N, 2)`.
    """

    def __init__(self, laplace_alpha: float = 1.0) -> None:
        self.laplace_alpha = laplace_alpha
        self.cond_matrix_: np.ndarray = np.empty((0, 0), dtype=np.float64)

    def fit(self, y_true: np.ndarray) -> "LabelCooccurrenceFeatureBuilder":
        Y = (np.asarray(y_true, dtype=np.float64) > 0.5).astype(np.float64)
        cooc = Y.T @ Y  # (L, L)
        support = Y.sum(axis=0, keepdims=True).T  # (L, 1)
        cond = (cooc + self.laplace_alpha) / (support + 2.0 * self.laplace_alpha)
        np.fill_diagonal(cond, 0.0)
        self.cond_matrix_ = cond
        return self

    def transform(self, stage1_probs: np.ndarray) -> np.ndarray:
        P = np.clip(np.asarray(stage1_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        C = self.cond_matrix_
        expected_cooc = P @ C  # (N, L)
        # Strongest implicating neighbor per target label j: max_i (P[:, i] * C[i, j])
        max_imply = np.max(P[:, :, None] * C[None, :, :], axis=1)  # (N, L)
        expected_cardinality = P.sum(axis=1, keepdims=True)  # (N, 1)
        binary_entropy = -(P * np.log(P) + (1.0 - P) * np.log(1.0 - P)).mean(
            axis=1, keepdims=True
        )  # (N, 1)
        return np.hstack(
            [expected_cooc, max_imply, expected_cardinality, binary_entropy]
        ).astype(np.float32)


def run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(42)
    Y = rng.integers(0, 2, size=(cfg.n_samples, cfg.num_labels)).astype(np.float64)
    # Enforce ground-truth hierarchy rule: label 1 => label 0
    Y[:, 0] = np.maximum(Y[:, 0], Y[:, 1])

    enforcer = HierarchicalConstraintEnforcer.discover_from_labels(
        Y, min_support=cfg.min_support, threshold=cfg.implication_threshold, mode="bottom_up"
    )
    assert (1, 0) in enforcer.edges

    raw_probs = rng.uniform(0.1, 0.9, size=(cfg.n_samples, cfg.num_labels))
    raw_probs[:, 1] = 0.92
    raw_probs[:, 0] = 0.25
    fixed_probs = enforcer.transform(raw_probs)
    assert np.all(fixed_probs[:, 0] >= fixed_probs[:, 1] - 1e-7)

    cooc_builder = LabelCooccurrenceFeatureBuilder(laplace_alpha=cfg.laplace_alpha).fit(Y)
    meta_feats = cooc_builder.transform(raw_probs)
    assert meta_feats.shape == (cfg.n_samples, 2 * cfg.num_labels + 2)

    print("[SMOKE TEST] Task 04 Hierarchy & Co-occurrence (`hierarchy_and_cooccurrence.py`)")
    print(f"  Discovered hierarchy edges  : {enforcer.edges}")
    print(f"  Enforced P(parent)>=P(child): {fixed_probs[0, 0]:.2f} >= {fixed_probs[0, 1]:.2f}")
    print(f"  Co-occurrence meta-features : {meta_feats.shape}")
    print("[SUCCESS] Hierarchy constraint and label co-occurrence checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Hierarchy & Label Co-occurrence")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
