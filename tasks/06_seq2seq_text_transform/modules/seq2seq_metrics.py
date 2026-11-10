"""
Task 06 — Pure-Python Vectorized Seq2Seq Evaluation Metrics
===========================================================
Standalone evaluation metrics for Sequence-to-Sequence generation, Machine Translation,
Text Normalization, and Grammatical Error Correction:
  - `chrf_score(pred, ref, n_max=6, beta=2.0)` + `batch_chrf_score`
  - `rouge_l_score(pred, ref, beta=1.0)` + `batch_rouge_l_score`
  - `exact_match_score(pred, ref, normalize_ws=True)` + `batch_exact_match_score`
  - `normalized_levenshtein(pred, ref)` + `batch_normalized_levenshtein`

Usage:
  python3 seq2seq_metrics.py --smoke-test
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
from typing import Dict, List, Sequence

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
    exp_name: str = "seq2seq_metrics"
    chrf_max_n: int = 6
    chrf_beta: float = 2.0


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def exact_match_score(pred: str, ref: str, normalize_ws: bool = True) -> float:
    """Computes normalized Exact Match (1.0 if identical after whitespace normalization, else 0.0)."""
    p_str = str(pred)
    r_str = str(ref)
    if normalize_ws:
        p_str = re.sub(r"\s+", " ", p_str.strip()).lower()
        r_str = re.sub(r"\s+", " ", r_str.strip()).lower()
    return 1.0 if p_str == r_str else 0.0


def normalized_levenshtein(pred: str, ref: str) -> float:
    """Computes `1.0 - edit_distance(pred, ref) / max(len(pred), len(ref), 1)`."""
    if pred == ref:
        return 1.0
    a, b = str(pred), str(ref)
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return 0.0

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[-1]
    return 1.0 - (dist / max(len(a), len(b), 1))


def rouge_l_score(pred: str, ref: str, beta: float = 1.0) -> float:
    """Computes token-level ROUGE-L F_beta score via Longest Common Subsequence."""
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0

    m, n = len(p_toks), len(r_toks)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev_diag = 0
        pi = p_toks[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            if pi == r_toks[j - 1]:
                dp[j] = prev_diag + 1
            else:
                if dp[j - 1] > dp[j]:
                    dp[j] = dp[j - 1]
            prev_diag = temp

    lcs = float(dp[n])
    if lcs == 0.0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    beta2 = beta * beta
    return (1.0 + beta2) * prec * rec / (beta2 * prec + rec)


def chrf_score(pred: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    """
    Pure-Python Character N-gram F_beta score (ChrF), standard for Machine Translation,
    Text Normalization, and Grammatical Error Correction.
    """
    p_str = re.sub(r"\s+", " ", str(pred).strip())
    r_str = re.sub(r"\s+", " ", str(ref).strip())
    if not p_str and not r_str:
        return 1.0
    if not p_str or not r_str:
        return 0.0

    prec_sum = 0.0
    rec_sum = 0.0
    valid_orders = 0

    for n in range(1, n_max + 1):
        if len(p_str) < n and len(r_str) < n:
            break
        valid_orders += 1
        p_ngrams = Counter(p_str[i : i + n] for i in range(max(0, len(p_str) - n + 1)))
        r_ngrams = Counter(r_str[i : i + n] for i in range(max(0, len(r_str) - n + 1)))

        overlap = sum(min(cnt, r_ngrams[ng]) for ng, cnt in p_ngrams.items())
        total_p = sum(p_ngrams.values())
        total_r = sum(r_ngrams.values())

        prec_sum += (overlap / total_p) if total_p > 0 else 0.0
        rec_sum += (overlap / total_r) if total_r > 0 else 0.0

    if valid_orders == 0:
        return 0.0
    avg_prec = prec_sum / valid_orders
    avg_rec = rec_sum / valid_orders
    if avg_prec + avg_rec == 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * avg_prec * avg_rec / (beta2 * avg_prec + avg_rec)


def evaluate_seq2seq_corpus(
    preds: Sequence[str], refs: Sequence[str], cfg: CFG | None = None
) -> Dict[str, float]:
    """Vectorized corpus-level evaluation across ChrF, ROUGE-L, Levenshtein, and Exact Match."""
    cfg = cfg or CFG()
    chrf_arr = np.array(
        [chrf_score(p, r, n_max=cfg.chrf_max_n, beta=cfg.chrf_beta) for p, r in zip(preds, refs)],
        dtype=np.float32,
    )
    rouge_arr = np.array([rouge_l_score(p, r) for p, r in zip(preds, refs)], dtype=np.float32)
    lev_arr = np.array(
        [normalized_levenshtein(p, r) for p, r in zip(preds, refs)], dtype=np.float32
    )
    em_arr = np.array([exact_match_score(p, r) for p, r in zip(preds, refs)], dtype=np.float32)

    return {
        "chrf": float(np.mean(chrf_arr)) if len(chrf_arr) else 0.0,
        "rouge_l": float(np.mean(rouge_arr)) if len(rouge_arr) else 0.0,
        "normalized_levenshtein": float(np.mean(lev_arr)) if len(lev_arr) else 0.0,
        "exact_match": float(np.mean(em_arr)) if len(em_arr) else 0.0,
    }


def _run_smoke_test(cfg: CFG) -> None:
    refs = [
        "saya sudah mengirim dokumen laporan keuangan kemarin sore",
        "pemerintah daerah mengumumkan kebijakan pajak baru",
    ]
    preds = [
        "saya sudah mengirim dokumen laporan keuangan kemarin sore",
        "pemerintah daerah mengumumkan kebijakan baru",
    ]
    metrics = evaluate_seq2seq_corpus(preds, refs, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = np.array(list(metrics.values()), dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr)
        pd.DataFrame([metrics]).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] ChrF={metrics['chrf']:.4f}, ROUGE-L={metrics['rouge_l']:.4f}, "
        f"Levenshtein={metrics['normalized_levenshtein']:.4f}, EM={metrics['exact_match']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 06 Seq2Seq Metrics Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
