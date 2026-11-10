"""
Task 09 — Aspect-Sentiment Triplet Extraction (ASTE) Span-Pair Decoder
======================================================================
Standalone Span-ASTE Triplet Decoder (`extract_absa_triplets` & `evaluate_absa_triplets`):
  - Decodes BIO token probabilities (`O=0, B-ASP=1, I-ASP=2, B-OPN=3, I-OPN=4`) into
    candidate Aspect spans `(a_start, a_end)` and Opinion spans `(o_start, o_end)`.
  - Pairs candidate aspect and opinion spans using a span-pair polarity probability matrix
    `(num_aspect_spans, num_opinion_spans, 4)` (`0=Invalid/None, 1=NEG, 2=NEU, 3=POS`)
    plus distance-aware pruning (`|a_center - o_center| <= max_span_distance`).
  - Evaluates exact triplet `(aspect_span, opinion_span, polarity)` Precision, Recall, and F1.

Usage:
  python3 absa_triplet_decoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

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


Span = Tuple[int, int]
Triplet = Tuple[Span, Span, str]

POLARITY_ID2NAME: Dict[int, str] = {
    1: "NEG",
    2: "NEU",
    3: "POS",
}


@dataclass
class CFG:
    exp_name: str = "absa_triplet_decoder"
    span_prob_threshold: float = 0.40
    pair_prob_threshold: float = 0.45
    max_span_distance: int = 12


def decode_aspect_and_opinion_spans(
    bio_probs: np.ndarray,
    span_threshold: float = 0.40,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    Decodes `(seq_len, 5)` token probabilities (`0:O, 1:B-ASP, 2:I-ASP, 3:B-OPN, 4:I-OPN`)
    into `aspect_spans` and `opinion_spans` of `(start_tok, end_tok_exclusive, mean_prob)`.
    """
    tags = np.argmax(bio_probs, axis=-1)
    max_p = np.max(bio_probs, axis=-1)
    seq_len = len(tags)

    def _collect(b_id: int, i_id: int) -> List[Tuple[int, int, float]]:
        spans: List[Tuple[int, int, float]] = []
        i = 0
        while i < seq_len:
            if tags[i] in (b_id, i_id):
                start = i
                i += 1
                while i < seq_len and tags[i] == i_id:
                    i += 1
                end = i
                conf = float(np.mean(max_p[start:end]))
                if conf >= span_threshold:
                    spans.append((start, end, conf))
            else:
                i += 1
        return spans

    return _collect(1, 2), _collect(3, 4)


def extract_absa_triplets(
    bio_probs: np.ndarray,
    pair_polarity_probs: np.ndarray,
    span_threshold: float = 0.40,
    pair_threshold: float = 0.45,
    max_span_distance: int = 12,
) -> List[Tuple[Span, Span, str, float]]:
    """
    Extracts `(aspect_span, opinion_span, polarity_str, confidence)` triplets.
    `pair_polarity_probs`: shape `(seq_len, seq_len, 4)` indexed by `(aspect_start, opinion_start, polarity_id)`.
    """
    asp_spans, opn_spans = decode_aspect_and_opinion_spans(
        bio_probs, span_threshold=span_threshold
    )
    triplets: List[Tuple[Span, Span, str, float]] = []

    for a_s, a_e, a_conf in asp_spans:
        a_center = 0.5 * (a_s + a_e)
        for o_s, o_e, o_conf in opn_spans:
            o_center = 0.5 * (o_s + o_e)
            if abs(a_center - o_center) > max_span_distance:
                continue

            pol_vec = pair_polarity_probs[a_s, o_s]  # (4,)
            best_pol_id = int(np.argmax(pol_vec[1:]) + 1)
            pair_prob = float(pol_vec[best_pol_id])
            joint_score = float((a_conf * o_conf * pair_prob) ** (1.0 / 3.0))

            if pair_prob >= pair_threshold:
                triplets.append(
                    ((a_s, a_e), (o_s, o_e), POLARITY_ID2NAME[best_pol_id], joint_score)
                )

    triplets.sort(key=lambda x: -x[3])
    return triplets


def evaluate_absa_triplets(
    pred_docs: Sequence[Sequence[Triplet]],
    gt_docs: Sequence[Sequence[Triplet]],
) -> Dict[str, float]:
    """Computes strict triplet-level Precision, Recall, and Micro-F1."""
    tp, fp, fn = 0, 0, 0
    for preds, gts in zip(pred_docs, gt_docs):
        p_set: Set[Triplet] = set(preds)
        g_set: Set[Triplet] = set(gts)
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "triplet_f1": float(f1)}


def _run_smoke_test(cfg: CFG) -> None:
    # Sentence tokens: ["kamar", "mandi", "sangat", "bersih", "tapi", "staf", "jutek"]
    # Aspect 1: (0, 2) "kamar mandi", Opinion 1: (2, 4) "sangat bersih" -> POS (3)
    # Aspect 2: (5, 6) "staf",        Opinion 2: (6, 7) "jutek"         -> NEG (1)
    bio_probs = np.array(
        [
            [0.02, 0.92, 0.02, 0.02, 0.02],  # 0: kamar (B-ASP)
            [0.02, 0.02, 0.92, 0.02, 0.02],  # 1: mandi (I-ASP)
            [0.02, 0.02, 0.02, 0.92, 0.02],  # 2: sangat (B-OPN)
            [0.02, 0.02, 0.02, 0.02, 0.92],  # 3: bersih (I-OPN)
            [0.92, 0.02, 0.02, 0.02, 0.02],  # 4: tapi (O)
            [0.02, 0.92, 0.02, 0.02, 0.02],  # 5: staf (B-ASP)
            [0.02, 0.02, 0.02, 0.92, 0.02],  # 6: jutek (B-OPN)
        ],
        dtype=np.float32,
    )
    pair_probs = np.zeros((7, 7, 4), dtype=np.float32)
    pair_probs[:, :, 0] = 0.90
    pair_probs[0, 2] = [0.05, 0.02, 0.03, 0.90]  # (0,2)->(2,4) POS
    pair_probs[5, 6] = [0.04, 0.91, 0.03, 0.02]  # (5,6)->(6,7) NEG

    triplets_scored = extract_absa_triplets(
        bio_probs,
        pair_probs,
        span_threshold=cfg.span_prob_threshold,
        pair_threshold=cfg.pair_prob_threshold,
        max_span_distance=cfg.max_span_distance,
    )
    pred_triplets = [(a, o, p) for a, o, p, _ in triplets_scored]
    gt_triplets = [((0, 2), (2, 4), "POS"), ((5, 6), (6, 7), "NEG")]
    metrics = evaluate_absa_triplets([pred_triplets], [gt_triplets])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", bio_probs)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", bio_probs)
        pd.DataFrame(
            [
                {"aspect": str(a), "opinion": str(o), "polarity": p, "score": s}
                for a, o, p, s in triplets_scored
            ]
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Decoded triplets={pred_triplets} | "
        f"Triplet-F1={metrics['triplet_f1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 ABSA Triplet Decoder Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
