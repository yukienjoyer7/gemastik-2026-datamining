#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Exploratory Data Analysis & Hierarchy Detector
===================================================================================
Standalone multilabel diagnostics script providing:
  1. **Label Frequency Distribution (Head vs. Tail Labels)**:
     Per-label positive count, positive rate, imbalance ratio vs. most frequent label,
     and classification into Head / Medium / Rare Tail (<20 positives) buckets.
  2. **Label Cardinality & Zero-Label Check**:
     Distribution of active labels per sample ($\sum_l y_{il}$), label density, and
     percentage of unlabeled / all-zero rows ($\sum_l y_{il} == 0$).
  3. **Pairwise Label Co-Occurrence Matrix & Implication Rule Discovery**:
     Computes empirical joint counts $C_{ij} = \sum_n y_{ni} y_{nj}$ and conditional
     probabilities $P(y_j = 1 \mid y_i = 1) = \frac{C_{ij}}{C_{ii}}$. Automatically
     detects strict or near-strict hierarchical implication rules ($y_i \implies y_j$
     when $P(y_j = 1 \mid y_i = 1) \ge 0.98$).

Usage:
    python eda_and_data.py --train-path train_folds.csv --label-cols label_0,label_1,label_2
    python eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class CFG:
    """Configuration for Multilabel Classification EDA & Hierarchy Discovery."""

    train_path: str = "train_folds.parquet"
    id_col: str = "id"
    text_col: str = "text"
    label_prefix: str = "label_"
    label_cols: Optional[Tuple[str, ...]] = None
    fold_col: str = "fold"

    implication_min_support: int = 5
    implication_conf_thr: float = 0.98
    tail_support_thr: int = 20
    seed: int = 42
    smoke_test: bool = False


def resolve_label_columns(df: pd.DataFrame, cfg: CFG) -> List[str]:
    if cfg.label_cols is not None and len(cfg.label_cols) > 0:
        return list(cfg.label_cols)
    cols = [c for c in df.columns if c.startswith(cfg.label_prefix)]
    if not cols:
        excluded = {cfg.id_col, cfg.text_col, cfg.fold_col}
        cols = [c for c in df.columns if c not in excluded]
    return cols


def analyze_label_frequencies(
    Y: np.ndarray, label_names: Sequence[str], tail_thr: int = 20
) -> pd.DataFrame:
    """Computes per-label support, positive rate, imbalance ratio, and Head/Tail bucket."""
    n_samples = max(Y.shape[0], 1)
    pos_counts = Y.sum(axis=0).astype(np.int64)
    max_pos = max(int(pos_counts.max()), 1)

    rows = []
    for name, cnt in zip(label_names, pos_counts):
        rate = (cnt / n_samples) * 100.0
        imb = max_pos / max(int(cnt), 1)
        if cnt < tail_thr:
            bucket = "TAIL (<20)"
        elif rate >= 25.0:
            bucket = "HEAD"
        else:
            bucket = "MEDIUM"
        rows.append(
            {
                "label": name,
                "pos_count": int(cnt),
                "pos_pct": round(float(rate), 2),
                "imbalance_vs_head": round(float(imb), 2),
                "bucket": bucket,
            }
        )
    return pd.DataFrame(rows).sort_values("pos_count", ascending=False).reset_index(drop=True)


def analyze_cardinality_and_zero_labels(Y: np.ndarray) -> Dict[str, Any]:
    """Analyzes active labels per sample, label density, and zero-label sample rate."""
    n_samples, n_labels = Y.shape
    row_sums = Y.sum(axis=1)
    zero_count = int((row_sums == 0).sum())
    val_counts = pd.Series(row_sums).value_counts().sort_index().to_dict()

    return {
        "num_samples": n_samples,
        "num_labels": n_labels,
        "label_cardinality_mean": round(float(row_sums.mean()), 4),
        "label_cardinality_std": round(float(row_sums.std()), 4),
        "label_density": round(float(row_sums.mean() / max(n_labels, 1)), 4),
        "zero_label_samples": zero_count,
        "zero_label_pct": round(float(zero_count * 100.0 / max(n_samples, 1)), 2),
        "max_labels_per_sample": int(row_sums.max()),
        "cardinality_histogram": {int(k): int(v) for k, v in val_counts.items()},
    }


def compute_cooccurrence_and_implications(
    Y: np.ndarray,
    label_names: Sequence[str],
    min_support: int = 5,
    conf_thr: float = 0.98,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    r"""
    Computes:
      1. Co-occurrence matrix $C = Y^T Y$ where $C_{ij} = \sum_n y_{ni} y_{nj}$.
      2. Conditional probability matrix $P_{ij} = P(y_j = 1 \mid y_i = 1) = \frac{C_{ij}}{C_{ii}}$.
      3. Strict / near-strict implication rules $(y_i \implies y_j)$ where
         $C_{ii} \ge \text{min\_support}$, $P(y_j = 1 \mid y_i = 1) \ge \text{conf\_thr}$,
         and $C_{jj} > C_{ii}$ (so $y_j$ is the parent/broader label and $y_i$ is the child).
    """
    Y_bin = (Y > 0.5).astype(np.float64)
    cooc = Y_bin.T @ Y_bin  # (L, L)
    diag = np.diag(cooc).copy()
    cond_prob = cooc / np.maximum(diag[:, None], 1.0)

    rules: List[Dict[str, Any]] = []
    n_labels = len(label_names)
    for i in range(n_labels):
        if diag[i] < min_support:
            continue
        for j in range(n_labels):
            if i == j:
                continue
            p_j_given_i = float(cond_prob[i, j])
            if p_j_given_i >= conf_thr and diag[j] >= diag[i]:
                rules.append(
                    {
                        "child_label": label_names[i],
                        "parent_label": label_names[j],
                        "child_support": int(diag[i]),
                        "parent_support": int(diag[j]),
                        "P(parent=1|child=1)": round(p_j_given_i, 4),
                    }
                )

    rules.sort(key=lambda r: (r["P(parent=1|child=1)"], r["child_support"]), reverse=True)
    return cooc, cond_prob, rules


def generate_synthetic_multilabel_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic multilabel dataset with:
      - Head labels, medium labels, and rare tail labels
      - Strict hierarchical implication rules (`label_deep_learning => label_tech`,
        `label_monetary_policy => label_economy`)
      - ~10% zero-label neutral samples
    """
    rng = np.random.default_rng(seed)
    label_cols = [
        "label_economy",
        "label_monetary_policy",
        "label_tech",
        "label_deep_learning",
        "label_sports",
        "label_rare_tail",
    ]
    Y = np.zeros((n_samples, len(label_cols)), dtype=np.int64)
    Y[:, 0] = rng.binomial(1, 0.40, size=n_samples)
    Y[:, 1] = rng.binomial(1, 0.18, size=n_samples)
    Y[:, 2] = rng.binomial(1, 0.35, size=n_samples)
    Y[:, 3] = rng.binomial(1, 0.15, size=n_samples)
    Y[:, 4] = rng.binomial(1, 0.25, size=n_samples)
    Y[:, 5] = rng.binomial(1, 0.05, size=n_samples)

    # Enforce strict hierarchy: child=1 => parent=1
    Y[Y[:, 1] == 1, 0] = 1  # monetary_policy => economy
    Y[Y[:, 3] == 1, 2] = 1  # deep_learning => tech

    rows = []
    for i in range(n_samples):
        row: Dict[str, Any] = {
            "id": f"doc_{i}",
            "text": f"synthetic document {i} discussing topics",
            "fold": i % 5,
        }
        for c_idx, col in enumerate(label_cols):
            row[col] = int(Y[i, c_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Multilabel EDA & Hierarchy Detector")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--label-cols", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    label_cols_tuple = tuple(args.label_cols.split(",")) if args.label_cols else None
    cfg = CFG(train_path=args.train_path, label_cols=label_cols_tuple, smoke_test=args.smoke_test)

    if cfg.smoke_test:
        print("[SMOKE TEST] Generating synthetic hierarchical multilabel dataset...")
        df = generate_synthetic_multilabel_data(seed=cfg.seed)
    else:
        df = load_table(cfg.train_path)

    label_cols = resolve_label_columns(df, cfg)
    Y = df[label_cols].values.astype(np.float64)

    print("=" * 72)
    print("1. LABEL FREQUENCY DISTRIBUTION (HEAD VS. TAIL)")
    print("=" * 72)
    freq_df = analyze_label_frequencies(Y, label_cols, tail_thr=cfg.tail_support_thr)
    print(freq_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("2. LABEL CARDINALITY & ZERO-LABEL SAMPLES CHECK")
    print("=" * 72)
    card_stats = analyze_cardinality_and_zero_labels(Y)
    for k, v in card_stats.items():
        print(f"  {k:<28}: {v}")

    print("\n" + "=" * 72)
    print("3. CONDITIONAL PROBABILITY P(y_j=1 | y_i=1) & HIERARCHY RULES")
    print("=" * 72)
    _, cond_prob, rules = compute_cooccurrence_and_implications(
        Y,
        label_cols,
        min_support=cfg.implication_min_support,
        conf_thr=cfg.implication_conf_thr,
    )
    cond_df = pd.DataFrame(np.round(cond_prob, 2), index=label_cols, columns=label_cols)
    print(cond_df.to_string())
    print("\nDetected Implication Rules (y_child => y_parent):")
    for r in rules:
        print(
            f"  {r['child_label']} => {r['parent_label']} "
            f"(support={r['child_support']}, P={r['P(parent=1|child=1)']:.4f})"
        )

    if cfg.smoke_test:
        assert len(rules) >= 2, "Expected at least 2 hierarchical implication rules in smoke test!"
        print("[SUCCESS] Task 04 Multilabel EDA & Hierarchy check passed.")


if __name__ == "__main__":
    main()
