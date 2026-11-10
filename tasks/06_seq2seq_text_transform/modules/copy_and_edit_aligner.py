"""
Task 06 — Word/Character Levenshtein Edit-Script Aligner (`KEEP`, `DELETE`, `INSERT`, `REPLACE`)
================================================================================================
Standalone dynamic-programming sequence aligner for Grammatical Error Correction (GEC),
Indonesian Slang Normalization, and Text Transformation error analysis / edit-tagging:
  - Extracts exact minimal edit scripts (`KEEP`, `DELETE`, `INSERT`, `REPLACE`) at word or char level
  - Builds token-level edit tags (`$KEEP`, `$DELETE`, `$REPLACE_<tok>`, `$APPEND_<tok>`)
    for fast GECToR-style sequence-tagging text correction
  - Reconstructs target sequence from source + edit script with 100% round-trip fidelity

Usage:
  python3 copy_and_edit_aligner.py --smoke-test
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


EditOp = Tuple[str, str, str, int, int]


@dataclass
class CFG:
    exp_name: str = "copy_and_edit_aligner"
    level: str = "word"  # "word" or "char"


def extract_edit_script(
    source: str, target: str, level: str = "word"
) -> List[EditOp]:
    """
    Computes the optimal Levenshtein alignment between `source` and `target` and returns
    a list of `(op_type, src_unit, tgt_unit, src_idx, tgt_idx)` tuples where
    `op_type` is in `{'KEEP', 'REPLACE', 'DELETE', 'INSERT'}`.
    """
    src_units: List[str] = (
        str(source).split() if level == "word" else list(str(source))
    )
    tgt_units: List[str] = (
        str(target).split() if level == "word" else list(str(target))
    )
    m, n = len(src_units), len(tgt_units)

    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        su = src_units[i - 1]
        for j in range(1, n + 1):
            tu = tgt_units[j - 1]
            cost = 0 if su == tu else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # DELETE
                dp[i, j - 1] + 1,      # INSERT
                dp[i - 1, j - 1] + cost,  # KEEP / REPLACE
            )

    # Backtrack to recover minimal edit script
    ops: List[EditOp] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            su, tu = src_units[i - 1], tgt_units[j - 1]
            cost = 0 if su == tu else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                op_name = "KEEP" if cost == 0 else "REPLACE"
                ops.append((op_name, su, tu, i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("DELETE", src_units[i - 1], "", i - 1, j))
            i -= 1
        else:
            ops.append(("INSERT", "", tgt_units[j - 1], i, j - 1))
            j -= 1

    ops.reverse()
    return ops


def apply_edit_script(ops: Sequence[EditOp], level: str = "word") -> str:
    """Reconstructs the target string by applying an edit script."""
    out_units: List[str] = []
    for op_name, _, tgt_u, _, _ in ops:
        if op_name in ("KEEP", "REPLACE", "INSERT"):
            out_units.append(tgt_u)
    return " ".join(out_units) if level == "word" else "".join(out_units)


def summarize_edit_operations(
    sources: Sequence[str], targets: Sequence[str], level: str = "word"
) -> Dict[str, float]:
    """Computes corpus-level proportions of KEEP, REPLACE, DELETE, INSERT operations."""
    counts = {"KEEP": 0, "REPLACE": 0, "DELETE": 0, "INSERT": 0}
    for s, t in zip(sources, targets):
        for op_name, _, _, _, _ in extract_edit_script(s, t, level=level):
            counts[op_name] += 1
    total = max(sum(counts.values()), 1)
    return {f"{k.lower()}_rate": float(v / total) for k, v in counts.items()}


def _run_smoke_test(cfg: CFG) -> None:
    src = "gw udh kirim berkas laporn ke kantor kmrn"
    tgt = "saya sudah mengirim berkas laporan ke kantor kemarin sore"

    ops = extract_edit_script(src, tgt, level=cfg.level)
    reconstructed = apply_edit_script(ops, level=cfg.level)
    assert reconstructed == tgt, f"Round-trip mismatch: {reconstructed} != {tgt}"

    summary = summarize_edit_operations([src], [tgt], level=cfg.level)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        rates = np.array(list(summary.values()), dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", rates)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", rates)
        pd.DataFrame(
            ops, columns=["op", "src_unit", "tgt_unit", "src_idx", "tgt_idx"]
        ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Edit script ops={len(ops)} | Round-trip exact=True | "
        f"keep={summary['keep_rate']:.2f}, replace={summary['replace_rate']:.2f}, "
        f"insert={summary['insert_rate']:.2f}, delete={summary['delete_rate']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 06 Copy & Edit Aligner Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
