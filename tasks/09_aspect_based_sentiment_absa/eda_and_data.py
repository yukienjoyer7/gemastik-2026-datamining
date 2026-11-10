"""
Task 09 — Aspect-Based Sentiment Analysis (ABSA): EDA & Multi-Aspect Fold Builder
=================================================================================
Diagnostics for Aspect-Category Sentiment Analysis (IndoNLU HoASA / CASA style):
  - Audits aspect presence rate per aspect (`y != 0` where `0=None, 1=Neg, 2=Neu, 3=Pos`)
  - Audits polarity balance within each aspect
  - Computes Mixed-Polarity Contrast Rate (reviews that simultaneously express Positive
    sentiment toward one aspect and Negative sentiment toward another aspect — the exact
    regime where document-level classifiers fail!)
  - Builds leak-free `MultilabelStratifiedKFold` (iterative stratification) across all
    `(aspect, polarity)` combinations and exports `train_folds.parquet`.

Usage:
  python3 eda_and_data.py --data-dir ./data --output-dir ./outputs
  python3 eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass, field
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
from sklearn.model_selection import StratifiedKFold


DEFAULT_ASPECTS: Tuple[str, ...] = (
    "kebersihan",
    "pelayanan",
    "harga",
    "lokasi",
)

POLARITY_MAP: Dict[int, str] = {
    0: "none",
    1: "negative",
    2: "neutral",
    3: "positive",
}


@dataclass
class CFG:
    exp_name: str = "eda_absa"
    seed: int = 42
    n_folds: int = 5
    id_col: str = "id"
    text_col: str = "text"
    aspect_cols: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_ASPECTS)
    num_polarities: int = 4
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def audit_absa_distribution(
    df: pd.DataFrame, cfg: CFG
) -> Dict[str, float]:
    """
    Audits aspect presence, polarity distribution, and mixed-polarity co-occurrence
    within single reviews.
    """
    Y = df[list(cfg.aspect_cols)].to_numpy(dtype=np.int32)
    n_samples, n_aspects = Y.shape

    active_mask = Y > 0
    aspects_per_review = active_mask.sum(axis=1)
    multi_aspect_rate = float(np.mean(aspects_per_review >= 2))

    # Reviews containing BOTH at least one Positive (3) AND at least one Negative (1) aspect
    has_pos = np.any(Y == 3, axis=1)
    has_neg = np.any(Y == 1, axis=1)
    mixed_polarity_rate = float(np.mean(has_pos & has_neg))

    report: Dict[str, float] = {
        "n_reviews": float(n_samples),
        "n_aspects": float(n_aspects),
        "mean_active_aspects_per_review": float(np.mean(aspects_per_review)),
        "multi_aspect_review_rate": multi_aspect_rate,
        "mixed_polarity_contrast_rate": mixed_polarity_rate,
    }

    for a_idx, aspect in enumerate(cfg.aspect_cols):
        col_y = Y[:, a_idx]
        report[f"aspect_{aspect}_presence_rate"] = float(np.mean(col_y > 0))
        for p_id, p_name in POLARITY_MAP.items():
            report[f"aspect_{aspect}_{p_name}_rate"] = float(np.mean(col_y == p_id))

    print("=== ABSA Multi-Aspect & Polarity Contrast Audit ===")
    for k, v in report.items():
        print(f"  {k:<38}: {v:.4f}")
    return report


def create_absa_multilabel_folds(
    df: pd.DataFrame, cfg: CFG
) -> pd.DataFrame:
    """
    Builds leak-free folds stratified jointly across all `(aspect, polarity)` labels
    using greedy iterative multilabel stratification (with Composite-Key StratifiedKFold fallback).
    """
    out = df.copy().reset_index(drop=True)
    Y = out[list(cfg.aspect_cols)].to_numpy(dtype=np.int32)
    n_samples = len(out)
    k = min(cfg.n_folds, n_samples)

    # Build binary indicator matrix (N, A * C) for iterative stratification
    indicators = []
    for a_idx in range(Y.shape[1]):
        for c_id in range(cfg.num_polarities):
            indicators.append((Y[:, a_idx] == c_id).astype(np.int32))
    bin_mat = np.stack(indicators, axis=1)

    rng = np.random.default_rng(cfg.seed)
    folds = np.full(n_samples, -1, dtype=np.int32)
    fold_sizes = np.zeros(k, dtype=np.int32)
    max_fold_size = int(np.ceil(n_samples / k))
    desired_label_counts = np.outer(
        np.ones(k, dtype=np.float64) / k, bin_mat.sum(axis=0)
    )

    unassigned = set(range(n_samples))
    while unassigned:
        # Pick the rarest remaining label column
        rem_idx = np.fromiter(unassigned, dtype=np.int32)
        col_counts = bin_mat[rem_idx].sum(axis=0)
        pos_cols = np.where(col_counts > 0)[0]
        if len(pos_cols) == 0:
            for idx in rem_idx:
                avail = np.where(fold_sizes < max_fold_size)[0]
                f_choice = int(avail[np.argmin(fold_sizes[avail])])
                folds[idx] = f_choice
                fold_sizes[f_choice] += 1
            break

        rarest_col = int(pos_cols[np.argmin(col_counts[pos_cols])])
        candidates = [i for i in rem_idx if bin_mat[i, rarest_col] == 1]
        rng.shuffle(candidates)

        for idx in candidates:
            if idx not in unassigned:
                continue
            avail = np.where(fold_sizes < max_fold_size)[0]
            # Choose fold with highest remaining need for this sample's active labels
            active_cols = np.where(bin_mat[idx] == 1)[0]
            scores = desired_label_counts[avail][:, active_cols].sum(axis=1) - 0.01 * fold_sizes[avail]
            best_f = int(avail[int(np.argmax(scores))])
            folds[idx] = best_f
            fold_sizes[best_f] += 1
            desired_label_counts[best_f, active_cols] -= 1.0
            unassigned.remove(idx)

    out["fold"] = folds
    return out


def generate_synthetic_absa_data(
    n_train: int = 80, n_test: int = 24, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates realistic Indonesian HoASA/CASA-style multi-aspect review data."""
    rng = np.random.default_rng(seed)

    aspect_lexicon = {
        "kebersihan": {
            1: ["kamar kotor berdebu bau apek sprei kusam", "lantai lengket kamar mandi jorok"],
            2: ["kebersihan kamar standar biasa saja", "cukup bersih namun handuk agak lama"],
            3: ["kamar sangat bersih wangi rapi higienis", "sprei bersih kamar mandi kinclong wangi"],
        },
        "pelayanan": {
            1: ["staf jutek pelayanan lambat resepsionis tidak ramah", "check in lama petugas cuek"],
            2: ["pelayanan staf biasa saja standar hotel", "resepsionis cukup membantu"],
            3: ["pelayanan sangat ramah staf sigap membantu cepat", "resepsionis sopan senyum ramah"],
        },
        "harga": {
            1: ["harga terlalu mahal biaya tambahan tidak sepadan", "tarif kemahalan fasilitas kurang"],
            2: ["harga standar sesuai bintang hotel", "tarif wajar normal"],
            3: ["harga sangat murah terjangkau hemat worth it", "diskon promo murah meriah sepadan"],
        },
        "lokasi": {
            1: ["lokasi jauh macet akses jalan sempit terpencil", "susah cari makan jauh dari pusat"],
            2: ["lokasi cukup dekat jalan raya", "akses lokasi standar"],
            3: ["lokasi strategis dekat stasiun pusat kota mall", "akses mudah dekat bandara strategis"],
        },
    }

    def _build_rows(n_rows: int, prefix: str) -> pd.DataFrame:
        rows = []
        aspects = list(DEFAULT_ASPECTS)
        for i in range(n_train if prefix == "TR" else n_rows):
            if i >= n_rows:
                break
            labels = {a: 0 for a in aspects}
            clauses: List[str] = []
            # Activate 1 to 3 aspects per review with contrasting polarities
            n_active = int(rng.integers(1, 4))
            chosen_aspects = rng.choice(aspects, size=n_active, replace=False)
            for a_idx, asp in enumerate(chosen_aspects):
                pol = int(rng.choice([1, 2, 3], p=[0.40, 0.15, 0.45]))
                labels[str(asp)] = pol
                phrase = str(rng.choice(aspect_lexicon[str(asp)][pol]))
                connector = " tetapi " if a_idx > 0 else ""
                clauses.append(f"{connector}untuk aspek {asp} {phrase}")
            row = {"id": f"{prefix}_{i:04d}", "text": "".join(clauses).strip()}
            row.update(labels)
            rows.append(row)
        return pd.DataFrame(rows)

    return _build_rows(n_train, "TR"), _build_rows(n_test, "TE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 ABSA EDA & Multi-Aspect Folds")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)

    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            train_df, test_df = generate_synthetic_absa_data(n_train=60, n_test=16, seed=cfg.seed)
            report = audit_absa_distribution(train_df, cfg)
            folds_df = create_absa_multilabel_folds(train_df, cfg)

            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            folds_df.to_parquet(out_dir / "train_folds.parquet", index=False)

            oof_dummy = folds_df[list(cfg.aspect_cols)].to_numpy(dtype=np.float32)
            test_dummy = np.zeros((len(test_df), len(cfg.aspect_cols)), dtype=np.float32)
            np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_dummy)
            np.save(out_dir / f"test_{cfg.exp_name}.npy", test_dummy)
            sub_df = pd.DataFrame({"id": test_df["id"]})
            for a in cfg.aspect_cols:
                sub_df[a] = 0
            sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

            print(
                f"[Smoke Test OK] Folds shape={folds_df.shape} | "
                f"Mixed-Polarity Rate={report['mixed_polarity_contrast_rate']:.3f}"
            )
        return

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        train_path = data_dir / "train.csv"
    if train_path.exists():
        train_df = (
            pd.read_parquet(train_path)
            if train_path.suffix == ".parquet"
            else pd.read_csv(train_path)
        )
    else:
        print(f"No data found at {data_dir}; generating synthetic ABSA dataset.")
        train_df, _ = generate_synthetic_absa_data(seed=cfg.seed)

    audit_absa_distribution(train_df, cfg)
    folds_df = create_absa_multilabel_folds(train_df, cfg)
    folds_df.to_parquet(out_dir / "train_folds.parquet", index=False)
    print(f"Saved multi-aspect stratified folds to {out_dir / 'train_folds.parquet'}")


if __name__ == "__main__":
    main()
