"""
Task 07 — Character-Level Span Probability Aggregator & Span Decoder
====================================================================
Standalone module for multi-tokenizer character-level NER ensembling:
  1. `token_probs_to_char_probs`: Projects sliding-window subword probabilities
     `(n_tokens, num_classes)` + `offset_mapping` onto a unified character-level
     probability array `(doc_char_len, num_classes)` — allowing direct blending of
     DeBERTa SentencePiece, RoBERTa BPE, and IndoBERT WordPiece predictions.
  2. `char_probs_to_spans`: Decodes character-level class probability matrices into
     clean `(start_char, end_char, label, mean_prob)` entity spans with whitespace trimming.

Usage:
  python3 char_span_aggregator.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
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


@dataclass
class CFG:
    exp_name: str = "char_span_aggregator"
    min_span_chars: int = 2


def token_probs_to_char_probs(
    doc_char_len: int,
    window_token_probs: Sequence[np.ndarray],
    window_offsets: Sequence[Sequence[Tuple[int, int]]],
) -> np.ndarray:
    """
    Projects overlapping sliding-window token probabilities onto a unified
    character-level probability matrix `(doc_char_len, num_classes)`.
    Characters not covered by any token default to class 0 (`O` = Outside).
    """
    if not window_token_probs:
        return np.zeros((doc_char_len, 1), dtype=np.float32)

    num_classes = window_token_probs[0].shape[-1]
    char_prob_sum = np.zeros((doc_char_len, num_classes), dtype=np.float32)
    char_counts = np.zeros((doc_char_len, 1), dtype=np.float32)

    for probs, offsets in zip(window_token_probs, window_offsets):
        for tok_idx, (s, e) in enumerate(offsets):
            s_c = max(0, min(doc_char_len, int(s)))
            e_c = max(0, min(doc_char_len, int(e)))
            if e_c > s_c and tok_idx < len(probs):
                char_prob_sum[s_c:e_c] += probs[tok_idx]
                char_counts[s_c:e_c] += 1.0

    unvisited = char_counts[:, 0] == 0.0
    char_counts[unvisited, 0] = 1.0
    char_prob_sum[unvisited, 0] = 1.0
    return char_prob_sum / char_counts


def char_probs_to_spans(
    text: str,
    char_probs: np.ndarray,
    id_to_label: Dict[int, str],
    class_thresholds: Dict[str, float] | None = None,
    min_span_chars: int = 2,
) -> List[Tuple[int, int, str, float]]:
    """
    Decodes a `(doc_char_len, num_classes)` probability array into entity spans
    `(start_char, end_char, label, mean_prob)`, stripping leading/trailing whitespace.
    """
    class_thresholds = class_thresholds or {}
    pred_ids = np.argmax(char_probs, axis=-1)

    for cid, cname in id_to_label.items():
        if cid == 0 or cname == "O":
            continue
        base_lbl = cname.split("-", 1)[-1]
        thr = class_thresholds.get(base_lbl, class_thresholds.get(cname, 0.0))
        if thr > 0.0:
            low_mask = (pred_ids == cid) & (char_probs[:, cid] < thr)
            pred_ids[low_mask] = 0

    spans: List[Tuple[int, int, str, float]] = []
    n_chars = len(text)
    for k in range(1, n_chars - 1):
        if text[k].isspace() and pred_ids[k - 1] > 0 and pred_ids[k - 1] == pred_ids[k + 1]:
            pred_ids[k] = pred_ids[k - 1]

    i = 0
    while i < n_chars:
        cid = int(pred_ids[i])
        cname = id_to_label.get(cid, "O")
        if cid == 0 or cname == "O":
            i += 1
            continue
        base_lbl = cname.split("-", 1)[-1]
        start = i
        while i < n_chars:
            next_cid = int(pred_ids[i])
            next_name = id_to_label.get(next_cid, "O")
            if next_cid == 0 or next_name == "O" or next_name.split("-", 1)[-1] != base_lbl:
                break
            i += 1
        end = i

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if (end - start) >= min_span_chars:
            mean_p = float(np.mean(np.max(char_probs[start:end], axis=-1)))
            spans.append((start, end, base_lbl, mean_p))

    return spans


def _run_smoke_test(cfg: CFG) -> None:
    text = "Budi Santoso works at Bank Mandiri in Jakarta."
    id2label = {0: "O", 1: "PER", 2: "ORG", 3: "LOC"}
    win_offsets = [[(0, 12), (13, 21), (22, 34), (35, 37), (38, 45)]]
    win_probs = [
        np.array(
            [
                [0.02, 0.94, 0.02, 0.02],  # Budi Santoso -> PER
                [0.95, 0.02, 0.02, 0.01],  # works at -> O
                [0.03, 0.02, 0.92, 0.03],  # Bank Mandiri -> ORG
                [0.96, 0.01, 0.02, 0.01],  # in -> O
                [0.02, 0.02, 0.03, 0.93],  # Jakarta -> LOC
            ],
            dtype=np.float32,
        )
    ]

    char_probs = token_probs_to_char_probs(len(text), win_probs, win_offsets)
    spans = char_probs_to_spans(
        text, char_probs, id2label, min_span_chars=cfg.min_span_chars
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", char_probs)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", char_probs)
        pd.DataFrame(spans, columns=["start", "end", "label", "prob"]).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] char_probs shape={char_probs.shape} | "
        f"decoded spans={[(text[s:e], l) for s, e, l, _ in spans]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 Character Span Aggregator Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
