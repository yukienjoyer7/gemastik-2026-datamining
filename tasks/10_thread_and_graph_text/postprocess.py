"""
Task 10 — Post-Processing: Graph Laplacian / Markov Random Field (MRF) Smoothing
================================================================================
Post-processing for Conversational Threads, Reply Trees & User-Post Graphs:
  - Given node probability predictions `P` `(N, C)`, row-normalized reply-tree adjacency
    `A_tree`, and row-normalized same-author adjacency `A_user`:
      `P_smooth = (1 - alpha_tree - alpha_user) * P + alpha_tree * (A_tree @ P) + alpha_user * (A_user @ P)`
  - Grid-searches `(alpha_tree, alpha_user)` on Out-Of-Fold validation to directly maximize
    Macro-F1 without any target leakage (uses only neighbor predicted probabilities!).

Usage:
  python3 postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score


@dataclass
class CFG:
    exp_name: str = "thread_graph_postprocess"
    seed: int = 42
    n_Steps: int = 1


def build_normalized_adjacencies(
    df: pd.DataFrame,
    id_col: str = "id",
    parent_col: str = "parent_id",
    user_col: str = "user_id",
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Constructs row-normalized sparse adjacency matrices `(A_tree, A_user)` of shape `(N, N)`
    with zero diagonal (strictly excluding self-loops).
    """
    n = len(df)
    ids = df[id_col].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    parents = (
        df[parent_col].fillna("").astype(str).tolist()
        if parent_col in df.columns
        else [""] * n
    )
    users = (
        df[user_col].fillna("u0").astype(str).tolist()
        if user_col in df.columns
        else ["u0"] * n
    )

    t_rows, t_cols = [], []
    for i, pid in enumerate(parents):
        if pid in id_to_idx and id_to_idx[pid] != i:
            j = id_to_idx[pid]
            t_rows.extend([i, j])
            t_cols.extend([j, i])

    u_groups: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(users):
        u_groups[u].append(i)

    u_rows, u_cols = [], []
    for idxs in u_groups.values():
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    if i != j:
                        u_rows.append(i)
                        u_cols.append(j)

    def _row_norm(rows: List[int], cols: List[int]) -> csr_matrix:
        if not rows:
            return csr_matrix((n, n), dtype=np.float32)
        mat = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
        )
        deg = np.asarray(mat.sum(axis=1)).ravel()
        inv_deg = np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)
        return csr_matrix(mat.multiply(inv_deg[:, None]))

    return _row_norm(t_rows, t_cols), _row_norm(u_rows, u_cols)


def apply_graph_laplacian_smoothing(
    probs: np.ndarray,
    a_tree: csr_matrix,
    a_user: csr_matrix,
    alpha_tree: float = 0.20,
    alpha_user: float = 0.10,
    n_steps: int = 1,
) -> np.ndarray:
    """
    Applies `n_steps` of Markov Random Field / Graph Laplacian probability smoothing:
      `P^{(t+1)} = (1 - a_t - a_u) * P^{(t)} + a_t * (A_tree @ P^{(t)}) + a_u * (A_user @ P^{(t)})`
    For isolated nodes with zero neighbors in `A_tree` or `A_user`, retains `P^{(t)}` weight.
    """
    p_curr = probs.astype(np.float32).copy()
    tree_has_nb = (np.asarray(a_tree.sum(axis=1)).ravel() > 0)[:, None]
    user_has_nb = (np.asarray(a_user.sum(axis=1)).ravel() > 0)[:, None]

    for _ in range(n_steps):
        p_tree = np.where(tree_has_nb, a_tree @ p_curr, p_curr)
        p_user = np.where(user_has_nb, a_user @ p_curr, p_curr)
        w_self = max(0.0, 1.0 - alpha_tree - alpha_user)
        p_curr = w_self * p_curr + alpha_tree * p_tree + alpha_user * p_user
        p_curr = p_curr / np.clip(p_curr.sum(axis=1, keepdims=True), 1e-8, None)

    return p_curr.astype(np.float32)


def optimize_graph_smoothing_weights(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    a_tree: csr_matrix,
    a_user: csr_matrix,
) -> Tuple[float, float, float, float]:
    """Grid-searches `(alpha_tree, alpha_user)` on OOF predictions to maximize Macro-F1."""
    base_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro"))
    best_f1 = base_f1
    best_at = 0.0
    best_au = 0.0

    for at in np.linspace(0.0, 0.45, 10):
        for au in np.linspace(0.0, 0.35, 8):
            if at + au > 0.70:
                continue
            smoothed = apply_graph_laplacian_smoothing(
                oof_probs, a_tree, a_user, alpha_tree=float(at), alpha_user=float(au)
            )
            f1 = float(f1_score(y_true, np.argmax(smoothed, axis=1), average="macro"))
            if f1 > best_f1:
                best_f1 = f1
                best_at = float(at)
                best_au = float(au)

    return best_at, best_au, base_f1, best_f1


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for t in range(15):
        lbl = t % 3
        for n_i in range(5):
            nid = f"T{t}_N{n_i}"
            pid = "" if n_i == 0 else f"T{t}_N{n_i - 1}"
            uid = f"U_{lbl}_{n_i % 2}"
            rows.append({"id": nid, "parent_id": pid, "user_id": uid, "label": lbl})
    df = pd.DataFrame(rows)
    y_true = df["label"].to_numpy(dtype=np.int32)

    # Noisy node-level OOF probabilities where 30% of nodes have high noise
    logits = rng.normal(0.0, 1.1, size=(len(df), 3))
    for i, y in enumerate(y_true):
        logits[i, y] += 1.1
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    oof_probs = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

    a_tree, a_user = build_normalized_adjacencies(df)
    best_at, best_au, base_f1, tuned_f1 = optimize_graph_smoothing_weights(
        oof_probs, y_true, a_tree, a_user
    )
    smoothed = apply_graph_laplacian_smoothing(
        oof_probs, a_tree, a_user, alpha_tree=best_at, alpha_user=best_au
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", smoothed)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", smoothed)
        pd.DataFrame({"id": df["id"], "label": np.argmax(smoothed, axis=1)}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] Unsmoothed Macro-F1={base_f1:.4f} -> "
        f"Graph-Smoothed Macro-F1={tuned_f1:.4f} "
        f"(alpha_tree={best_at:.2f}, alpha_user={best_au:.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10 Graph Laplacian Post-Processing")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
