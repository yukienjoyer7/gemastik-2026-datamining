"""
Task 07 — Strict Entity Span Evaluator (Micro-F1, Macro-F1, F_beta)
===================================================================
Standalone evaluation module `evaluate_entity_spans`:
  - Computes strict span-level Precision, Recall, Micro-F1, Macro-F1, and Micro-F_beta
    (e.g. F5 for PII Data Detection) over `(start, end, label)` or `(doc_id, start, end, label)` tuples.

Usage:
  python3 span_metrics.py --smoke-test
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


SpanTuple = Tuple[int, int, str]


@dataclass
class CFG:
    exp_name: str = "span_metrics"
    beta: float = 5.0


def evaluate_entity_spans(
    pred_spans_docs: Sequence[Sequence[SpanTuple]],
    gt_spans_docs: Sequence[Sequence[SpanTuple]],
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Computes strict entity-level Precision, Recall, Micro-F1, Macro-F1, and Micro-F_beta
    across a sequence of documents where each span is `(start, end, label)`.
    """
    tp_all, fp_all, fn_all = 0, 0, 0
    labels: Set[str] = set()
    for gts in gt_spans_docs:
        for _, _, lbl in gts:
            labels.add(lbl)
    for preds in pred_spans_docs:
        for _, _, lbl in preds:
            labels.add(lbl)

    per_label_f1: List[float] = []
    for lbl in sorted(labels):
        tp, fp, fn = 0, 0, 0
        for preds, gts in zip(pred_spans_docs, gt_spans_docs):
            p_set = {(s, e, l) for s, e, l in preds if l == lbl}
            g_set = {(s, e, l) for s, e, l in gts if l == lbl}
            tp += len(p_set & g_set)
            fp += len(p_set - g_set)
            fn += len(g_set - p_set)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        denom = 2 * tp + fp + fn
        per_label_f1.append((2.0 * tp / denom) if denom > 0 else 0.0)

    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    beta2 = beta * beta
    micro_fbeta = (
        (1.0 + beta2) * prec * rec / (beta2 * prec + rec)
        if (prec + rec) > 0
        else 0.0
    )

    return {
        "precision": float(prec),
        "recall": float(rec),
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        f"micro_f{beta:g}": float(micro_fbeta),
    }


def _run_smoke_test(cfg: CFG) -> None:
    gt_docs = [
        [(0, 12, "PER"), (22, 34, "ORG"), (38, 45, "LOC")],
        [(5, 15, "PER"), (20, 30, "LOC")],
    ]
    pred_docs = [
        [(0, 12, "PER"), (22, 34, "ORG"), (38, 45, "LOC")],
        [(5, 15, "PER"), (20, 31, "LOC")],  # 1 boundary mismatch
    ]
    metrics = evaluate_entity_spans(pred_docs, gt_docs, beta=cfg.beta)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = np.array(list(metrics.values()), dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr)
        pd.DataFrame([metrics]).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Micro-F1={metrics['micro_f1']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}, Micro-F5={metrics['micro_f5']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 Span Metrics Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
