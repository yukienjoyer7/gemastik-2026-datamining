"""
Task 07 — NER Post-Processing: BIO Sequence Repair, Class Threshold Optimizer & Span NMS
========================================================================================
High-leverage post-processing utilities for Token Classification & Span Extraction:
  1. `repair_bio_sequence`: Fixes illegal `O -> I-X` and `B-X -> I-Y` transitions
     (either promoting orphan `I-X` to `B-X` or relabeling to `I-X`).
  2. `merge_adjacent_same_label_spans`: Merges adjacent spans `(s1, e1, lbl)` and
     `(s2, e2, lbl)` separated only by whitespace (e.g., `"Budi"` + `" "` + `"Santoso"`).
  3. `span_nms`: Non-Maximum Suppression resolving overlapping entity spans by
     retaining the higher-confidence (or longer) span when IoU / overlap exceeds a threshold.
  4. `optimize_per_class_thresholds_and_min_len`: Out-Of-Fold coordinate grid search
     over per-class probability thresholds and minimum span character length to
     directly maximize $F_\beta$ (e.g., $F_5$ in Kaggle PII Data Detection).

Usage:
  python3 postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
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


ScoredSpan = Tuple[int, int, str, float]  # (start, end, label, score)
SpanTuple = Tuple[int, int, str]


@dataclass
class CFG:
    exp_name: str = "ner_postprocess"
    seed: int = 42
    beta: float = 5.0  # F5-score as in Kaggle PII Data Detection
    nms_iou_threshold: float = 0.50
    max_merge_gap: int = 1
    output_dir: str = "./outputs"


def repair_bio_sequence(tags: Sequence[str], mode: str = "promote_to_b") -> List[str]:
    """
    Repairs illegal BIO tag transitions:
      - If `O -> I-X`: converts `I-X` to `B-X` (if `mode == 'promote_to_b'`) or `'O'`.
      - If `B-X / I-X -> I-Y` (where `X != Y`): converts `I-Y` to `B-Y` or `I-X`.
    """
    repaired: List[str] = []
    prev_type: str | None = None

    for tag in tags:
        if tag == "O" or not tag:
            repaired.append("O")
            prev_type = None
            continue

        prefix, _, ent_type = tag.partition("-")
        if prefix == "B":
            repaired.append(f"B-{ent_type}")
            prev_type = ent_type
        elif prefix == "I":
            if prev_type == ent_type:
                repaired.append(f"I-{ent_type}")
            else:
                if mode == "promote_to_b":
                    repaired.append(f"B-{ent_type}")
                    prev_type = ent_type
                elif mode == "continue_prev" and prev_type is not None:
                    repaired.append(f"I-{prev_type}")
                else:
                    repaired.append("O")
                    prev_type = None
        else:
            repaired.append(f"B-{tag}")
            prev_type = tag

    return repaired


def merge_adjacent_same_label_spans(
    text: str, spans: Sequence[ScoredSpan], max_gap: int = 1
) -> List[ScoredSpan]:
    """
    Merges adjacent spans of the same entity class when separated only by up to
    `max_gap` whitespace characters.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[ScoredSpan] = [ordered[0]]

    for s, e, lbl, prob in ordered[1:]:
        ps, pe, plbl, pprob = merged[-1]
        gap_str = text[pe:s] if s >= pe else ""
        if lbl == plbl and 0 <= (s - pe) <= max_gap and (gap_str == "" or gap_str.isspace()):
            new_prob = 0.5 * (pprob + prob)
            merged[-1] = (ps, max(pe, e), plbl, new_prob)
        else:
            merged.append((s, e, lbl, prob))

    return merged


def span_nms(
    spans: Sequence[ScoredSpan], iou_threshold: float = 0.50
) -> List[ScoredSpan]:
    """
    1D Span Non-Maximum Suppression: sorts candidate spans by `(score, length)`
    descending and suppresses overlapping spans whose 1D IoU or containment exceeds
    `iou_threshold`.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda x: (x[3], x[1] - x[0]), reverse=True)
    kept: List[ScoredSpan] = []

    for cand in ordered:
        cs, ce, _, _ = cand
        clen = max(ce - cs, 1)
        suppress = False
        for ks, ke, _, _ in kept:
            klen = max(ke - ks, 1)
            inter = max(0, min(ce, ke) - max(cs, ks))
            if inter > 0:
                union = clen + klen - inter
                iou = inter / max(union, 1)
                overlap_min = inter / min(clen, klen)
                if iou >= iou_threshold or overlap_min >= 0.80:
                    suppress = True
                    break
        if not suppress:
            kept.append(cand)

    return sorted(kept, key=lambda x: (x[0], x[1]))


def compute_fbeta_spans(
    pred_docs: Sequence[Sequence[SpanTuple]],
    gt_docs: Sequence[Sequence[SpanTuple]],
    beta: float = 5.0,
) -> float:
    tp, fp, fn = 0, 0, 0
    for p, g in zip(pred_docs, gt_docs):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    b2 = beta * beta
    return float((1.0 + b2) * prec * rec / (b2 * prec + rec))


def filter_spans_by_class_params(
    docs_scored_spans: Sequence[Sequence[ScoredSpan]],
    class_thresholds: Dict[str, float],
    class_min_lens: Dict[str, int],
) -> List[List[SpanTuple]]:
    filtered_docs: List[List[SpanTuple]] = []
    for doc_spans in docs_scored_spans:
        kept: List[SpanTuple] = []
        for s, e, lbl, prob in doc_spans:
            thr = class_thresholds.get(lbl, 0.50)
            min_len = class_min_lens.get(lbl, 2)
            if prob >= thr and (e - s) >= min_len:
                kept.append((s, e, lbl))
        filtered_docs.append(kept)
    return filtered_docs


def optimize_per_class_thresholds_and_min_len(
    docs_scored_spans: Sequence[Sequence[ScoredSpan]],
    gt_docs: Sequence[Sequence[SpanTuple]],
    entity_classes: Sequence[str],
    beta: float = 5.0,
) -> Tuple[Dict[str, float], Dict[str, int], float]:
    """
    Coordinate-ascent OOF optimizer tuning per-class probability thresholds
    (critical for $F_5$ where false negatives cost $25\times$ more than false positives!)
    and minimum character span lengths.
    """
    best_thrs = {c: 0.50 for c in entity_classes}
    best_lens = {c: 2 for c in entity_classes}
    best_score = compute_fbeta_spans(
        filter_spans_by_class_params(docs_scored_spans, best_thrs, best_lens),
        gt_docs,
        beta=beta,
    )

    thr_grid = np.linspace(0.15, 0.85, 15)
    len_grid = [2, 3, 4, 5]

    for _ in range(2):
        for cls_name in entity_classes:
            for cand_thr in thr_grid:
                trial_thrs = dict(best_thrs)
                trial_thrs[cls_name] = float(cand_thr)
                preds = filter_spans_by_class_params(docs_scored_spans, trial_thrs, best_lens)
                score = compute_fbeta_spans(preds, gt_docs, beta=beta)
                if score > best_score:
                    best_score = score
                    best_thrs[cls_name] = float(cand_thr)

            for cand_len in len_grid:
                trial_lens = dict(best_lens)
                trial_lens[cls_name] = int(cand_len)
                preds = filter_spans_by_class_params(docs_scored_spans, best_thrs, trial_lens)
                score = compute_fbeta_spans(preds, gt_docs, beta=beta)
                if score > best_score:
                    best_score = score
                    best_lens[cls_name] = int(cand_len)

    return best_thrs, best_lens, best_score


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)

    # 1. Test BIO sequence repair
    noisy_bio = ["O", "I-PER", "I-PER", "O", "B-ORG", "I-LOC"]
    repaired_bio = repair_bio_sequence(noisy_bio, mode="promote_to_b")
    assert repaired_bio == ["O", "B-PER", "I-PER", "O", "B-ORG", "B-LOC"]

    # 2. Test adjacent span merging & NMS
    sample_text = "Dr. Budi Santoso works at Bank Mandiri in Jakarta."
    fragmented: List[ScoredSpan] = [
        (4, 8, "PER", 0.82),    # "Budi"
        (9, 16, "PER", 0.88),   # "Santoso" -> should merge into (4, 16, "PER")
        (4, 12, "PER", 0.45),   # lower-score overlapping fragment -> suppressed by NMS
        (26, 38, "ORG", 0.91),  # "Bank Mandiri"
    ]
    merged = merge_adjacent_same_label_spans(sample_text, fragmented, max_gap=1)
    nms_cleaned = span_nms(merged, iou_threshold=cfg.nms_iou_threshold)

    # 3. Test Per-Class Threshold & Min-Len Optimizer for F5
    classes = ["PER", "ORG", "LOC"]
    gt_docs: List[List[SpanTuple]] = []
    pred_docs_scored: List[List[ScoredSpan]] = []

    for i in range(30):
        gt_docs.append([(4, 16, "PER"), (26, 38, "ORG"), (42, 49, "LOC")])
        doc_cands: List[ScoredSpan] = [
            # True positives with moderate probabilities (e.g., 0.25..0.55) that a naive 0.50 cutoff misses!
            (4, 16, "PER", float(rng.uniform(0.24, 0.75))),
            (26, 38, "ORG", float(rng.uniform(0.28, 0.80))),
            (42, 49, "LOC", float(rng.uniform(0.22, 0.70))),
            # Short noisy false positive (length 1)
            (0, 1, "PER", float(rng.uniform(0.30, 0.60))),
            # Low-confidence false positive
            (17, 22, "ORG", float(rng.uniform(0.05, 0.18))),
        ]
        pred_docs_scored.append(doc_cands)

    default_preds = filter_spans_by_class_params(
        pred_docs_scored,
        {c: 0.50 for c in classes},
        {c: 1 for c in classes},
    )
    base_f5 = compute_fbeta_spans(default_preds, gt_docs, beta=cfg.beta)
    best_thrs, best_lens, tuned_f5 = optimize_per_class_thresholds_and_min_len(
        pred_docs_scored, gt_docs, classes, beta=cfg.beta
    )
    tuned_preds = filter_spans_by_class_params(pred_docs_scored, best_thrs, best_lens)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        oof_arr = np.array([len(p) for p in tuned_preds], dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", oof_arr[:6])
        pd.DataFrame(
            {
                "doc_id": [f"D{i:03d}" for i in range(6)],
                "spans": [json.dumps(p) for p in tuned_preds[:6]],
            }
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] BIO Repair={repaired_bio} | Merged+NMS spans={len(nms_cleaned)} | "
        f"Naive(0.50) F5={base_f5:.4f} -> Tuned F5={tuned_f5:.4f} "
        f"(thrs={best_thrs}, min_lens={best_lens})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 NER Post-Processing")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
