"""
Task 06 — Minimum Bayes Risk (MBR) Decoding Over N-Best Hypotheses
==================================================================
Standalone Minimum Bayes Risk (MBR) hypothesis selector for N-best beam outputs
or diverse multi-model Seq2Seq ensembles:
  - Given N candidate strings `h_1..h_N` (from beam search, diverse beam search, or multiple models),
    selects `h* = argmax_i sum_j w_j * U(h_i, h_j)` maximizing expected competition utility
    (`chrf`, `rouge_l`, `levenshtein`, or `exact_match`).

Usage:
  python3 mbr_decoding.py --smoke-test
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
from typing import Callable, List, Sequence, Tuple

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
    exp_name: str = "mbr_decoding"
    default_metric: str = "chrf"
    chrf_max_n: int = 6
    chrf_beta: float = 2.0


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def _normalized_levenshtein(pred: str, ref: str) -> float:
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
    return 1.0 - (prev[-1] / max(len(a), len(b), 1))


def _rouge_l_score(pred: str, ref: str, beta: float = 1.0) -> float:
    p_toks = _tokenize(pred)
    r_toks = _tokenize(ref)
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
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev_diag = temp
    lcs = float(dp[n])
    if lcs == 0.0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    beta2 = beta * beta
    return (1.0 + beta2) * prec * rec / (beta2 * prec + rec)


def _chrf_score(pred: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    p_str = re.sub(r"\s+", " ", str(pred).strip())
    r_str = re.sub(r"\s+", " ", str(ref).strip())
    if not p_str and not r_str:
        return 1.0
    if not p_str or not r_str:
        return 0.0
    prec_sum, rec_sum, valid_orders = 0.0, 0.0, 0
    for n in range(1, n_max + 1):
        if len(p_str) < n and len(r_str) < n:
            break
        valid_orders += 1
        p_ng = Counter(p_str[i : i + n] for i in range(max(0, len(p_str) - n + 1)))
        r_ng = Counter(r_str[i : i + n] for i in range(max(0, len(r_str) - n + 1)))
        overlap = sum(min(cnt, r_ng[ng]) for ng, cnt in p_ng.items())
        tp, tr = sum(p_ng.values()), sum(r_ng.values())
        prec_sum += (overlap / tp) if tp > 0 else 0.0
        rec_sum += (overlap / tr) if tr > 0 else 0.0
    if valid_orders == 0:
        return 0.0
    avg_p, avg_r = prec_sum / valid_orders, rec_sum / valid_orders
    if avg_p + avg_r == 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)


def _get_similarity_fn(metric: str) -> Callable[[str, str], float]:
    metric_lower = metric.lower()
    if metric_lower == "chrf":
        return _chrf_score
    if metric_lower == "rouge_l":
        return _rouge_l_score
    if metric_lower in ("levenshtein", "edit"):
        return _normalized_levenshtein
    raise ValueError(f"Unsupported MBR metric: {metric}")


def mbr_select_best_hypothesis(
    candidates_list: Sequence[str],
    metric: str = "chrf",
    weights: Sequence[float] | None = None,
    return_scores: bool = False,
) -> str | Tuple[str, np.ndarray]:
    """
    Minimum Bayes Risk (MBR) Decoding over a list of candidate strings.
    Selects the candidate `h_i` that maximizes expected similarity `sum_j w_j * sim(h_i, h_j)`
    against all hypotheses in `candidates_list`.
    """
    cands = [str(c) for c in candidates_list]
    n = len(cands)
    if n == 0:
        return ("", np.zeros(0, dtype=np.float32)) if return_scores else ""
    if n == 1:
        return (cands[0], np.ones(1, dtype=np.float32)) if return_scores else cands[0]

    sim_fn = _get_similarity_fn(metric)
    w = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones(n, dtype=np.float64)
    )
    w = w / np.sum(w)

    sim_matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_fn(cands[i], cands[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    expected_utility = (sim_matrix @ w).astype(np.float32)
    best_idx = int(np.argmax(expected_utility))
    if return_scores:
        return cands[best_idx], expected_utility
    return cands[best_idx]


def _run_smoke_test(cfg: CFG) -> None:
    beams = [
        "saya sudah mengirim dokumen laporan keuangan kemarin",
        "saya sudah mengirim dokumen laporan keuangan kemarin sore",
        "saya sudah kirim dokumen laporan keuangan kemarin sore",
        "dokumen keuangan sudah dikirim oleh saya kemarin malam",
        "saya sdh mengirim dokumen laporan kemarin sore",
    ]
    best_chrf, utilities = mbr_select_best_hypothesis(
        beams, metric="chrf", return_scores=True
    )
    best_rouge = mbr_select_best_hypothesis(beams, metric="rouge_l")
    best_lev = mbr_select_best_hypothesis(beams, metric="levenshtein")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", utilities)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", utilities)
        pd.DataFrame(
            {"metric": ["chrf", "rouge_l", "levenshtein"], "selected": [best_chrf, best_rouge, best_lev]}
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] MBR selected(ChrF)='{best_chrf}' | "
        f"max_expected_utility={float(np.max(utilities)):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 06 MBR Decoding Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
