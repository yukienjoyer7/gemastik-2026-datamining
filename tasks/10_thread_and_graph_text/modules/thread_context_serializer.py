"""
Task 10 — Token-Budget-Aware Thread Context Serializer
======================================================
Standalone module `ThreadContextSerializer` that builds structured conversational context
strings from `(id, parent_id, text)` reply trees:
  - Extracts `[ROOT]`, `[PARENT]`, `[PREV_SIBLING]`, `[TARGET]`, and `[CHILD_SUMMARY]` zones
  - Enforces strict per-zone word/subword token budgets so a verbose `[ROOT]` OP post
    or `[CHILD_SUMMARY]` never truncates the `[TARGET]` post being classified!

Usage:
  python3 thread_context_serializer.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    exp_name: str = "thread_context_serializer"
    root_budget: int = 32
    parent_budget: int = 40
    sibling_budget: int = 20
    target_budget: int = 64
    child_budget: int = 32


class ThreadContextSerializer:
    """
    Serializes each node in a reply tree into a structured multi-zone string:
      `[ROOT] ... [SEP] [PARENT] ... [SEP] [PREV_SIBLING] ... [SEP] [TARGET] ... [SEP] [CHILD_SUMMARY] ...`
    while guaranteeing the `[TARGET]` post is never truncated by ancestor/descendant context.
    """

    def __init__(
        self,
        root_budget: int = 32,
        parent_budget: int = 40,
        sibling_budget: int = 20,
        target_budget: int = 64,
        child_budget: int = 32,
        id_col: str = "id",
        parent_col: str = "parent_id",
        text_col: str = "text",
    ) -> None:
        self.root_budget = root_budget
        self.parent_budget = parent_budget
        self.sibling_budget = sibling_budget
        self.target_budget = target_budget
        self.child_budget = child_budget
        self.id_col = id_col
        self.parent_col = parent_col
        self.text_col = text_col

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = str(text).split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words])

    def serialize_dataframe(self, df: pd.DataFrame) -> List[str]:
        ids = df[self.id_col].astype(str).tolist()
        texts = df[self.text_col].fillna("").astype(str).tolist()
        parents = (
            df[self.parent_col].fillna("").astype(str).tolist()
            if self.parent_col in df.columns
            else [""] * len(df)
        )
        id_to_idx = {nid: i for i, nid in enumerate(ids)}

        parent_idx: Dict[int, int] = {}
        children_idx: Dict[int, List[int]] = defaultdict(list)
        for i, pid in enumerate(parents):
            if pid in id_to_idx and id_to_idx[pid] != i:
                p_i = id_to_idx[pid]
                parent_idx[i] = p_i
                children_idx[p_i].append(i)

        root_idx = np.arange(len(df), dtype=np.int32)
        for i in range(len(df)):
            curr = i
            seen = {curr}
            while curr in parent_idx:
                nxt = parent_idx[curr]
                if nxt in seen:
                    break
                seen.add(nxt)
                curr = nxt
            root_idx[i] = curr

        serialized: List[str] = []
        for i in range(len(df)):
            r_i = int(root_indices_val := root_idx[i])
            r_str = (
                self._truncate_words(texts[r_i], self.root_budget)
                if r_i != i
                else "SELF_ROOT"
            )

            if i in parent_idx:
                p_i = parent_idx[i]
                p_str = self._truncate_words(texts[p_i], self.parent_budget)
                sibs = [s for s in children_idx[p_i] if s < i]
                sib_str = (
                    self._truncate_words(texts[sibs[-1]], self.sibling_budget)
                    if sibs
                    else "NONE"
                )
            else:
                p_str = "NONE"
                sib_str = "NONE"

            t_str = self._truncate_words(texts[i], self.target_budget)
            ch = children_idx.get(i, [])
            ch_str = (
                self._truncate_words(
                    " | ".join(texts[c] for c in ch[:3]), self.child_budget
                )
                if ch
                else "NONE"
            )

            prompt = (
                f"[ROOT] {r_str} [SEP] "
                f"[PARENT] {p_str} [SEP] "
                f"[PREV_SIBLING] {sib_str} [SEP] "
                f"[TARGET] {t_str} [SEP] "
                f"[CHILD_SUMMARY] {ch_str}"
            )
            serialized.append(prompt)

        return serialized


def _run_smoke_test(cfg: CFG) -> None:
    df = pd.DataFrame(
        {
            "id": ["N0", "N1", "N2", "N3"],
            "parent_id": ["", "N0", "N0", "N1"],
            "text": [
                "Breaking news pengumuman kebijakan subsidi energi tahun depan",
                "Apakah kebijakan subsidi ini sudah disahkan oleh parlemen?",
                "Setuju banget semoga tepat sasaran ke masyarakat",
                "Belum disahkan masih tahap pembahasan komisi terkait",
            ],
        }
    )
    serializer = ThreadContextSerializer(
        root_budget=cfg.root_budget,
        parent_budget=cfg.parent_budget,
        sibling_budget=cfg.sibling_budget,
        target_budget=cfg.target_budget,
        child_budget=cfg.child_budget,
    )
    prompts = serializer.serialize_dataframe(df)
    assert "[ROOT]" in prompts[3] and "[PARENT]" in prompts[3] and "[TARGET]" in prompts[3]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        lens = np.array([len(p.split()) for p in prompts], dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", lens)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", lens)
        pd.DataFrame({"id": df["id"], "serialized_prompt": prompts}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] Serialized {len(prompts)} nodes | "
        f"Sample N3 prompt='{prompts[3][:95]}...'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10 Thread Context Serializer Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
