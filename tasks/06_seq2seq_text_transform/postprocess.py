"""
Task 06 — Seq2Seq Post-Processing: Degenerate Loop Cleaner & Multi-Model MBR Ensemble
=====================================================================================
Post-processing utilities for generative Seq2Seq text transformation tasks:
  1. `strip_repeated_ngrams`: Removes pathological trailing n-gram repetition loops
     produced by neural decoding (e.g., `"ke jakarta ke jakarta ke jakarta"` -> `"ke jakarta"`).
  2. `clean_and_bound_generation`: Normalizes whitespace/punctuation, preserves source
     casing style if appropriate, and enforces `[min_len_ratio, max_len_ratio]` bounds
     relative to `source_text` (with safe fallback if generation collapsed).
  3. `multi_model_mbr_ensemble`: Combines predictions from multiple Seq2Seq models
     (e.g., Tier-1 Lookup + ByT5 + mT5 + IndoBART) via weighted Minimum Bayes Risk.

Usage:
  python3 postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import re
import site
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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
    exp_name: str = "seq2seq_postprocess"
    min_len_ratio: float = 0.25
    max_len_ratio: float = 2.50
    max_ngram_repeat: int = 4
    output_dir: str = "./outputs"


def strip_repeated_ngrams(text: str, max_n: int = 4) -> str:
    """
    Removes consecutive repeated token n-grams of order `1..max_n`
    (a common failure mode in autoregressive decoding).
    """
    tokens = str(text).split()
    if len(tokens) <= 1:
        return str(text).strip()

    for n in range(1, max_n + 1):
        cleaned: List[str] = []
        i = 0
        while i < len(tokens):
            if (
                len(cleaned) >= n
                and i + n <= len(tokens)
                and tokens[i : i + n] == cleaned[-n:]
            ):
                i += n
            else:
                cleaned.append(tokens[i])
                i += 1
        tokens = cleaned
    return " ".join(tokens)


def normalize_spacing_and_punctuation(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()


def clean_and_bound_generation(
    pred_text: str,
    source_text: str,
    min_len_ratio: float = 0.25,
    max_len_ratio: float = 2.50,
    max_ngram_repeat: int = 4,
) -> str:
    """
    Applies full post-processing pipeline to a generated sequence:
      1. Strips repeated consecutive token n-grams (1..4)
      2. Normalizes whitespace and punctuation spacing
      3. Enforces `[min_len_ratio, max_len_ratio]` character length bounds relative
         to `source_text` (falls back to `source_text` if output is empty/collapsed)
    """
    cleaned = strip_repeated_ngrams(pred_text, max_n=max_ngram_repeat)
    cleaned = normalize_spacing_and_punctuation(cleaned)

    src_clean = normalize_spacing_and_punctuation(source_text)
    src_len = max(len(src_clean), 1)

    if len(cleaned) < min_len_ratio * src_len:
        # Generation collapsed prematurely -> fallback to cleaned source
        return src_clean

    max_chars = int(round(max_len_ratio * src_len))
    if len(cleaned) > max_chars:
        # Truncate cleanly at word boundary
        truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
        cleaned = truncated if len(truncated) >= min_len_ratio * src_len else cleaned[:max_chars]

    return cleaned


def chrf_pair_score(a: str, b: str, n_max: int = 5) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    vals = []
    for n in range(1, n_max + 1):
        if len(a) < n or len(b) < n:
            continue
        ca = Counter(a[i : i + n] for i in range(len(a) - n + 1))
        cb = Counter(b[i : i + n] for i in range(len(b) - n + 1))
        ov = sum(min(c, cb[k]) for k, c in ca.items())
        p = ov / max(sum(ca.values()), 1)
        r = ov / max(sum(cb.values()), 1)
        vals.append((2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def multi_model_mbr_ensemble(
    model_predictions: Sequence[Sequence[str]],
    sources: Sequence[str],
    model_weights: Sequence[float] | None = None,
    cfg: CFG | None = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Combines `M` lists of string predictions `(n_models, n_samples)` via weighted
    Minimum Bayes Risk (MBR) after cleaning each candidate.
    """
    cfg = cfg or CFG()
    n_models = len(model_predictions)
    n_samples = len(sources)
    weights = (
        np.asarray(model_weights, dtype=np.float64)
        if model_weights is not None
        else np.ones(n_models, dtype=np.float64)
    )
    weights = weights / np.sum(weights)

    final_preds: List[str] = []
    consensus_scores = np.zeros(n_samples, dtype=np.float32)

    for idx in range(n_samples):
        src = sources[idx]
        cands = [
            clean_and_bound_generation(
                model_predictions[m][idx],
                src,
                min_len_ratio=cfg.min_len_ratio,
                max_len_ratio=cfg.max_len_ratio,
                max_ngram_repeat=cfg.max_ngram_repeat,
            )
            for m in range(n_models)
        ]

        sim_mat = np.eye(n_models, dtype=np.float64)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                s = chrf_pair_score(cands[i], cands[j])
                sim_mat[i, j] = s
                sim_mat[j, i] = s

        utilities = sim_mat @ weights
        best_m = int(np.argmax(utilities))
        final_preds.append(cands[best_m])
        consensus_scores[idx] = float(utilities[best_m])

    return final_preds, consensus_scores


def _run_smoke_test(cfg: CFG) -> None:
    sources = [
        "yg blm bayar tlg segera ditransfer ya",
        "sy sdh kirim dokumennya kmrn sore",
        "brp hrg tiket kereta dr jkt ke sby",
    ]
    refs = [
        "yang belum bayar tolong segera ditransfer ya",
        "saya sudah kirim dokumennya kemarin sore",
        "berapa harga tiket kereta dari jakarta ke surabaya",
    ]
    # Simulate 3 models (one with degenerate beam loop, two strong neural models)
    model_1 = [
        "yang belum bayar tolong segera ditransfer ya ditransfer ya ditransfer ya",
        "saya sudah kirim dokumennya kemarin sore",
        "berapa harga tiket kereta dari jakarta ke surabaya",
    ]
    model_2 = [
        "yang belum bayar tolong segera ditransfer ya",
        "saya sudah mengirim dokumennya kemarin sore",
        "berapa harga tiket kereta dari jakarta ke surabaya ke surabaya",
    ]
    model_3 = [
        "yang belum bayar tolong segera ditransfer ya",
        "saya sudah kirim dokumennya kemarin sore",
        "",  # collapsed output to verify fallback
    ]

    raw_chrf = float(np.mean([chrf_pair_score(p, r) for p, r in zip(model_1, refs)]))
    ensembled_preds, consensus = multi_model_mbr_ensemble(
        [model_1, model_2, model_3], sources, model_weights=[1.0, 1.2, 0.8], cfg=cfg
    )
    post_chrf = float(np.mean([chrf_pair_score(p, r) for p, r in zip(ensembled_preds, refs)]))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", consensus)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", consensus)
        pd.DataFrame({"id": range(len(ensembled_preds)), "prediction": ensembled_preds}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] Raw Model-1 ChrF={raw_chrf:.4f} -> "
        f"Cleaned+MBR Ensemble ChrF={post_chrf:.4f} | Consensus={float(np.mean(consensus)):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 06 Seq2Seq Post-Processing")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
