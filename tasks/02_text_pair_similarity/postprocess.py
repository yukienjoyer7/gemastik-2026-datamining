#!/usr/bin/env python3
r"""
Task 02 — Text Pair Similarity: Transitive Graph Post-Processing & Calibration
==============================================================================
Standalone post-processing script providing:
  1. **Transitive Triangle Consistency Adjustment**:
     In pair similarity / duplicate detection graphs, if $P(A, B)$ and $P(B, C)$
     are high within a connected component, $P(A, C)$ should be consistent with
     triangle transitivity:
       $P^{(t+1)}(A, C) = (1 - \alpha) P^{(t)}(A, C) + \alpha \max_{w \in N(A) \cap N(C)} (P^{(t)}(A, w) \cdot P^{(t)}(w, C))$
     or average spectral neighbor propagation across shared wedges $(A, w, C)$.
  2. **MinMax & Rank-Average Scaling**:
     Aligns arbitrary model score ranges before blending or thresholding.
  3. **OOF Threshold Optimization**:
     Finds the optimal binary decision threshold $\tau^*$ maximizing F1 or Accuracy.
  4. Standardized OOF & prediction contract (`oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     `sub_<exp_name>.csv`).

Usage:
    python postprocess.py --oof-path oof_tier2_cross_encoder.npy --test-probs-path test_tier2_cross_encoder.npy
    python postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import f1_score


@dataclass
class CFG:
    """Configuration for Task 02 Transitive Post-Processing."""

    exp_name: str = "tier2_pair_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_pair.npy"
    test_probs_path: str = "test_tier1_pair.npy"
    output_dir: str = "."
    id_col: str = "id"
    text_a_col: str = "text_a"
    text_b_col: str = "text_b"
    target_col: str = "label"

    alpha_transitive: float = 0.25
    n_propagation_steps: int = 2
    high_edge_thr: float = 0.60
    scaling_mode: str = "minmax"  # "minmax" | "rank" | "none"
    binarize_submission: bool = False
    smoke_test: bool = False


def scale_scores(scores: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Applies MinMax `[0, 1]` scaling or uniform rank scaling."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if mode == "rank":
        return (rankdata(arr, method="average") - 0.5) / max(len(arr), 1)
    if mode == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            return np.clip(arr, 0.0, 1.0)
        return (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def transitive_triangle_adjustment(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    probs: np.ndarray,
    alpha: float = 0.25,
    n_steps: int = 2,
    high_edge_thr: float = 0.60,
) -> np.ndarray:
    """
    Propagates transitive consistency across triangles `(u, w, v)` in the pair graph:
      - Builds a weighted undirected graph where edge weight `W(u, v) = P(u, v)`.
      - For each target pair `(u, v)`, inspects shared neighbors `w in N(u) ∩ N(v)`.
      - Computes the transitive wedge lower bound `T(u, v) = max_{w} (W(u, w) * W(w, v))`
        and soft-updates `W(u, v) <- (1 - alpha) * W(u, v) + alpha * max(W(u, v), T(u, v))`
        whenever `T(u, v) >= high_edge_thr ** 2`.
    """
    n_pairs = len(probs)
    curr_probs = np.clip(np.asarray(probs, dtype=np.float64).copy(), 0.0, 1.0)

    node_to_id: Dict[str, int] = {}
    u_ids = np.zeros(n_pairs, dtype=np.int64)
    v_ids = np.zeros(n_pairs, dtype=np.int64)

    for i, (ra, rb) in enumerate(zip(texts_a, texts_b)):
        sa, sb = str(ra), str(rb)
        if sa not in node_to_id:
            node_to_id[sa] = len(node_to_id)
        if sb not in node_to_id:
            node_to_id[sb] = len(node_to_id)
        u_ids[i] = node_to_id[sa]
        v_ids[i] = node_to_id[sb]

    for _ in range(n_steps):
        adj_weights: Dict[int, Dict[int, float]] = defaultdict(dict)
        for i in range(n_pairs):
            u, v = int(u_ids[i]), int(v_ids[i])
            if u != v:
                w = float(curr_probs[i])
                if w > adj_weights[u].get(v, -1.0):
                    adj_weights[u][v] = w
                    adj_weights[v][u] = w

        next_probs = curr_probs.copy()
        min_wedge_prod = high_edge_thr * high_edge_thr

        for i in range(n_pairs):
            u, v = int(u_ids[i]), int(v_ids[i])
            if u == v:
                next_probs[i] = 1.0
                continue
            nb_u = adj_weights[u]
            nb_v = adj_weights[v]
            if len(nb_u) > len(nb_v):
                nb_u, nb_v = nb_v, nb_u

            best_wedge = 0.0
            for w, p_uw in nb_u.items():
                if w in nb_v:
                    prod = p_uw * nb_v[w]
                    if prod > best_wedge:
                        best_wedge = prod

            if best_wedge >= min_wedge_prod and best_wedge > curr_probs[i]:
                next_probs[i] = (1.0 - alpha) * curr_probs[i] + alpha * best_wedge

        curr_probs = np.clip(next_probs, 0.0, 1.0)

    return curr_probs


def optimize_binary_threshold(oof_probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Grid-searches the optimal binary classification threshold maximizing F1."""
    y_bin = (np.asarray(y_true) >= 0.5).astype(int)
    best_thr = 0.50
    best_f1 = -1.0
    for thr in np.linspace(0.15, 0.85, 141):
        score = float(f1_score(y_bin, (oof_probs >= thr).astype(int), zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def run_postprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray]:
    n_tr = len(train_df)
    all_a = train_df[cfg.text_a_col].fillna("").astype(str).tolist() + test_df[cfg.text_a_col].fillna("").astype(str).tolist()
    all_b = train_df[cfg.text_b_col].fillna("").astype(str).tolist() + test_df[cfg.text_b_col].fillna("").astype(str).tolist()

    oof_scaled = scale_scores(oof_probs, mode=cfg.scaling_mode)
    test_scaled = scale_scores(test_probs, mode=cfg.scaling_mode)
    combined_scores = np.concatenate([oof_scaled, test_scaled], axis=0)

    adjusted_all = transitive_triangle_adjustment(
        all_a,
        all_b,
        combined_scores,
        alpha=cfg.alpha_transitive,
        n_steps=cfg.n_propagation_steps,
        high_edge_thr=cfg.high_edge_thr,
    )
    adj_oof = adjusted_all[:n_tr]
    adj_test = adjusted_all[n_tr:]

    y_true = train_df[cfg.target_col].values.astype(np.float64)
    raw_r = float(pearsonr(y_true, oof_scaled)[0]) if np.std(oof_scaled) > 1e-9 else 0.0
    adj_r = float(pearsonr(y_true, adj_oof)[0]) if np.std(adj_oof) > 1e-9 else 0.0
    best_thr, best_f1 = optimize_binary_threshold(adj_oof, y_true)

    print("=" * 72)
    print("TASK 02 TRANSITIVE GRAPH POST-PROCESSING SUMMARY")
    print("=" * 72)
    print(f"  Scaled OOF Pearson r      : {raw_r:.4f}")
    print(f"  Transitive Adjusted OOF r : {adj_r:.4f}")
    print(f"  Optimal F1 Threshold      : thr={best_thr:.3f} | F1={best_f1:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", adj_oof.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", adj_test.astype(np.float32))

    sub_vals = (adj_test >= best_thr).astype(int) if cfg.binarize_submission else adj_test
    pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values, cfg.target_col: sub_vals}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )
    return adj_oof, adj_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Transitive Triangle Post-Processing")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--oof-path", type=str, default=CFG.oof_path)
    parser.add_argument("--test-probs-path", type=str, default=CFG.test_probs_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        oof_path=args.oof_path,
        test_probs_path=args.test_probs_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        # Build a synthetic triangle (A-B high, B-C high, A-C underestimated)
        train_df = pd.DataFrame(
            [
                {"id": "tr_0", "text_a": "Node_A", "text_b": "Node_B", "label": 1.0},
                {"id": "tr_1", "text_a": "Node_B", "text_b": "Node_C", "label": 1.0},
                {"id": "tr_2", "text_a": "Node_A", "text_b": "Node_C", "label": 1.0},
                {"id": "tr_3", "text_a": "Node_D", "text_b": "Node_E", "label": 0.0},
            ]
        )
        test_df = pd.DataFrame(
            [
                {"id": "te_0", "text_a": "Node_A", "text_b": "Node_C"},
                {"id": "te_1", "text_a": "Node_D", "text_b": "Node_A"},
            ]
        )
        oof_probs = np.array([0.92, 0.90, 0.55, 0.10], dtype=np.float32)
        test_probs = np.array([0.52, 0.08], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            adj_oof, adj_test = run_postprocess(train_df, test_df, oof_probs, test_probs, cfg)
            assert adj_oof[2] > oof_probs[2], "Transitive triangle adjustment did not boost (Node_A, Node_C)!"
            print(
                f"  Triangle edge (Node_A, Node_C) boosted: {oof_probs[2]:.3f} -> {adj_oof[2]:.3f}"
            )
            print("[SUCCESS] Task 02 post-processing smoke test passed.")
    else:
        p_tr = pathlib.Path(cfg.train_path)
        train_df = pd.read_parquet(p_tr) if p_tr.suffix.lower() == ".parquet" else pd.read_csv(p_tr)
        p_te = pathlib.Path(cfg.test_path)
        test_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        run_postprocess(train_df, test_df, oof_probs, test_probs, cfg)


if __name__ == "__main__":
    main()
