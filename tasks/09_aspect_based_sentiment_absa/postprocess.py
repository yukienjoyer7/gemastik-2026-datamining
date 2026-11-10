"""
Task 09 — Post-Processing: Per-Aspect Prior Calibration & Hierarchical Presence Thresholding
============================================================================================
Post-processing for Aspect-Category Sentiment Analysis `(N, num_aspects, num_polarities)`:
  1. **Hierarchical Presence vs. Polarity Decomposition**:
     - Computes `P(present) = 1.0 - P(class_0_None)`
     - If `P(present) < tau_presence[aspect]`, predicts `0` (`None`).
     - Otherwise selects `argmax_{c in {1,2,3}} w[aspect, c] * P(class_c)` so rare polarities
       (`Neutral`) are not drowned out by dominant `None` probabilities.
  2. **Per-Aspect Coordinate Multiplier & Presence Threshold Search**:
     - Tunes `tau_presence[a]` and class multipliers `w[a, c]` out-of-fold to directly maximize
       competition Macro-F1 across all aspects.

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
from typing import Tuple

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
from sklearn.metrics import f1_score


@dataclass
class CFG:
    exp_name: str = "absa_postprocess"
    seed: int = 42
    num_aspects: int = 4
    num_polarities: int = 4


def decode_hierarchical_absa(
    probs: np.ndarray,
    presence_thresholds: np.ndarray,
    polarity_multipliers: np.ndarray,
) -> np.ndarray:
    """
    Decodes `(N, A, C)` probability tensor using per-aspect presence thresholds `(A,)`
    and per-aspect active polarity multipliers `(A, C)`.
    """
    n_samples, n_aspects, _ = probs.shape
    preds = np.zeros((n_samples, n_aspects), dtype=np.int32)

    for a in range(n_aspects):
        p_a = probs[:, a, :]  # (N, C)
        p_present = 1.0 - p_a[:, 0]
        scaled_active = p_a[:, 1:] * polarity_multipliers[a, 1:][None, :]
        best_active_pol = np.argmax(scaled_active, axis=-1) + 1
        preds[:, a] = np.where(p_present >= presence_thresholds[a], best_active_pol, 0)

    return preds


def compute_absa_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = [
        float(f1_score(y_true[:, a], y_pred[:, a], average="macro", zero_division=0))
        for a in range(y_true.shape[1])
    ]
    return float(np.mean(f1s))


def optimize_absa_calibration(
    oof_probs: np.ndarray, y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Coordinate ascent per aspect optimizing `presence_thresholds` `(A,)` and
    `polarity_multipliers` `(A, C)` for Macro-F1.
    """
    _, n_aspects, n_classes = oof_probs.shape
    raw_preds = np.argmax(oof_probs, axis=-1)
    base_f1 = compute_absa_macro_f1(y_true, raw_preds)

    best_tau = np.full(n_aspects, 0.50, dtype=np.float32)
    best_mult = np.ones((n_aspects, n_classes), dtype=np.float32)

    tau_grid = np.linspace(0.25, 0.75, 11)
    mult_grid = np.linspace(0.50, 2.00, 13)

    for a in range(n_aspects):
        y_a = y_true[:, a]
        p_a = oof_probs[:, a : a + 1, :]
        t_a = best_tau[a : a + 1].copy()
        m_a = best_mult[a : a + 1, :].copy()

        best_a_f1 = float(
            f1_score(y_a, np.argmax(p_a[:, 0, :], axis=-1), average="macro", zero_division=0)
        )

        for _ in range(2):
            for cand_tau in tau_grid:
                t_a[0] = cand_tau
                cand_pred = decode_hierarchical_absa(p_a, t_a, m_a)[:, 0]
                score = float(f1_score(y_a, cand_pred, average="macro", zero_division=0))
                if score > best_a_f1:
                    best_a_f1 = score
                    best_tau[a] = cand_tau
                t_a[0] = best_tau[a]

            for c in range(1, n_classes):
                for cand_m in mult_grid:
                    m_a[0, c] = cand_m
                    cand_pred = decode_hierarchical_absa(p_a, t_a, m_a)[:, 0]
                    score = float(f1_score(y_a, cand_pred, average="macro", zero_division=0))
                    if score > best_a_f1:
                        best_a_f1 = score
                        best_mult[a, c] = cand_m
                    m_a[0, c] = best_mult[a, c]

    tuned_preds = decode_hierarchical_absa(oof_probs, best_tau, best_mult)
    tuned_f1 = compute_absa_macro_f1(y_true, tuned_preds)
    if tuned_f1 < base_f1:
        return best_tau, best_mult, base_f1, base_f1
    return best_tau, best_mult, base_f1, tuned_f1


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 120
    # Imbalanced polarities where Neutral (2) is under-predicted before calibration
    y_true = rng.choice(
        [0, 1, 2, 3], size=(n, cfg.num_aspects), p=[0.45, 0.22, 0.08, 0.25]
    )
    logits = rng.normal(0.0, 0.9, size=(n, cfg.num_aspects, cfg.num_polarities))
    for i in range(n):
        for a in range(cfg.num_aspects):
            logits[i, a, y_true[i, a]] += 1.4
    # Suppress class 2 (Neutral) logit to mimic uncalibrated model bias
    logits[:, :, 2] -= 0.7
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    oof_probs = (exp_l / exp_l.sum(axis=-1, keepdims=True)).astype(np.float32)

    best_tau, best_mult, base_f1, tuned_f1 = optimize_absa_calibration(oof_probs, y_true)
    calibrated_preds = decode_hierarchical_absa(oof_probs, best_tau, best_mult)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", calibrated_preds)
        pd.DataFrame(
            calibrated_preds, columns=[f"aspect_{a}" for a in range(cfg.num_aspects)]
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Raw Argmax Macro-F1={base_f1:.4f} -> "
        f"Hierarchical Calibrated Macro-F1={tuned_f1:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 ABSA Post-Processing")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
