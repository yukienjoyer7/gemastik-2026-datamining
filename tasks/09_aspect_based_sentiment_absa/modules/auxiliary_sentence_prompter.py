"""
Task 09 — Auxiliary-Sentence Pair Prompter & Tensor Reshaper
============================================================
Standalone module implementing `AuxiliarySentencePrompter` (Sun et al., NAACL / SemEval 1st Place):
  - Converts wide multi-aspect DataFrames (`id, text, aspect_1..aspect_A`) into long
    auxiliary-sentence pair DataFrames (`id, aspect_idx, aspect_name, prompt_text, text_pair, label`)
    supporting QA-style (`"Bagaimana sentimen terhadap aspek {aspect}?"`) and NLI-style
    (`"Aspek: {aspect} ({keywords})"`) auxiliary sentences.
  - Reshapes flat `(N * A, C)` out-of-fold / test predictions back into structured `(N, A, C)` tensors.

Usage:
  python3 auxiliary_sentence_prompter.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass, field
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


DEFAULT_ASPECT_DESCRIPTIONS: Dict[str, str] = {
    "kebersihan": "kebersihan kamar mandi sprei handuk bau rapi higienis",
    "pelayanan": "pelayanan staf resepsionis petugas ramah cepat sopan",
    "harga": "harga tarif biaya mahal murah terjangkau diskon sepadan",
    "lokasi": "lokasi akses strategis dekat stasiun bandara pusat kota",
}


@dataclass
class CFG:
    exp_name: str = "auxiliary_sentence_prompter"
    seed: int = 42
    style: str = "nli"  # "nli" or "qa"
    num_polarities: int = 4
    aspect_descriptions: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ASPECT_DESCRIPTIONS)
    )


class AuxiliarySentencePrompter:
    """
    Converts wide multi-aspect tables `(N, A)` into long `(N * A)` auxiliary-sentence
    pair tables for Cross-Encoder training and reshapes predictions back to `(N, A, C)`.
    """

    def __init__(
        self,
        aspect_descriptions: Dict[str, str],
        style: str = "nli",
        id_col: str = "id",
        text_col: str = "text",
    ) -> None:
        self.aspect_descriptions = dict(aspect_descriptions)
        self.aspect_names: List[str] = list(aspect_descriptions.keys())
        self.style = style
        self.id_col = id_col
        self.text_col = text_col

    def format_auxiliary_sentence(self, aspect_name: str) -> str:
        keywords = self.aspect_descriptions.get(aspect_name, aspect_name)
        if self.style == "qa":
            return f"Bagaimana sentimen ulasan ini terhadap aspek {aspect_name} ({keywords})?"
        return f"Aspek: {aspect_name} ({keywords})"

    def wide_to_long(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        rows = []
        has_fold = "fold" in df.columns
        for row_idx, row in df.reset_index(drop=True).iterrows():
            rid = row[self.id_col]
            rtext = str(row[self.text_col])
            for a_idx, aspect in enumerate(self.aspect_names):
                aux = self.format_auxiliary_sentence(aspect)
                item = {
                    "sample_idx": int(row_idx),
                    self.id_col: rid,
                    "aspect_idx": int(a_idx),
                    "aspect_name": aspect,
                    "aux_sentence": aux,
                    self.text_col: rtext,
                    "prompt_text": f"[CLS] {aux} [SEP] {rtext} [SEP]",
                }
                if has_fold:
                    item["fold"] = int(row["fold"])
                if is_train and aspect in df.columns:
                    item["label"] = int(row[aspect])
                rows.append(item)
        return pd.DataFrame(rows)

    def long_probs_to_wide_tensor(
        self,
        long_df: pd.DataFrame,
        long_probs: np.ndarray,
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Reshapes `(N * A, C)` probability predictions back to `(N, A, C)`."""
        if n_samples is None:
            n_samples = int(long_df["sample_idx"].max()) + 1
        n_aspects = len(self.aspect_names)
        n_classes = long_probs.shape[-1]
        out = np.zeros((n_samples, n_aspects, n_classes), dtype=np.float32)

        s_indices = long_df["sample_idx"].to_numpy(dtype=np.int64)
        a_indices = long_df["aspect_idx"].to_numpy(dtype=np.int64)
        out[s_indices, a_indices] = long_probs.astype(np.float32)
        return out


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    wide_df = pd.DataFrame(
        {
            "id": [f"R_{i}" for i in range(10)],
            "text": [f"Ulasan hotel nomor {i} kamar bersih tapi harga mahal" for i in range(10)],
            "kebersihan": rng.integers(0, 4, size=10),
            "pelayanan": rng.integers(0, 4, size=10),
            "harga": rng.integers(0, 4, size=10),
            "lokasi": rng.integers(0, 4, size=10),
            "fold": np.arange(10) % 5,
        }
    )

    prompter = AuxiliarySentencePrompter(
        aspect_descriptions=cfg.aspect_descriptions, style=cfg.style
    )
    long_df = prompter.wide_to_long(wide_df, is_train=True)
    dummy_long_probs = rng.dirichlet(
        np.ones(cfg.num_polarities), size=len(long_df)
    ).astype(np.float32)
    wide_tensor = prompter.long_probs_to_wide_tensor(
        long_df, dummy_long_probs, n_samples=len(wide_df)
    )
    assert wide_tensor.shape == (10, 4, 4)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", wide_tensor)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", wide_tensor)
        long_df.head(12).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] wide_df={wide_df.shape} -> long_df={long_df.shape} -> "
        f"reshaped_tensor={wide_tensor.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 Auxiliary-Sentence Prompter Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
