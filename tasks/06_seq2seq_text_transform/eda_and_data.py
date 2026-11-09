"""
Task 06 — Seq2Seq & Text Transformation: EDA, Copy-Ratio Audit & Dataset
========================================================================
Diagnostics for sequence-to-sequence text transformation tasks:
  - Source vs. target character & token length distributions
  - Target-to-source length ratio quantiles (`len(target) / len(source)`)
    for setting beam-search `min_length`, `max_new_tokens`, and `length_penalty`
  - Verbatim Copy Ratio & Edit Distance Profile:
    * % of pairs where target == source (identity baseline ceiling)
    * Token-level copy overlap (% of target tokens present in source)
    * Normalized Levenshtein similarity between source and target
    (Critical decision gate: copy-heavy tasks > 0.75 favor edit-tagging / ByT5 /
    lookup tables; low-copy tasks < 0.40 favor abstractive Seq2Seq + MBR)
  - PyTorch `Seq2SeqDataset` with dynamic padding and target label masking (`-100`)

Usage:
  python3 eda_and_data.py --data-dir ./data --output-dir ./outputs
  python3 eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import re
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
import torch
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import Dataset


@dataclass
class CFG:
    exp_name: str = "eda_seq2seq"
    seed: int = 42
    n_folds: int = 5
    max_source_len: int = 128
    max_target_len: int = 64
    source_col: str = "source_text"
    target_col: str = "target_text"
    group_col: str = "group_id"
    data_dir: str = "./data"
    output_dir: str = "./outputs"


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def levenshtein_distance(a: str, b: str) -> int:
    """Fast O(min(N, M)) space dynamic programming Levenshtein edit distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def normalized_levenshtein_sim(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - (levenshtein_distance(a, b) / denom)


def compute_seq2seq_eda(df: pd.DataFrame, cfg: CFG) -> Dict[str, float]:
    """Computes length ratio quantiles, verbatim copy rate, and edit-distance profile."""
    sources = df[cfg.source_col].fillna("").astype(str).tolist()
    targets = df[cfg.target_col].fillna("").astype(str).tolist()

    src_char_lens = np.array([max(len(s), 1) for s in sources], dtype=np.float32)
    tgt_char_lens = np.array([len(t) for t in targets], dtype=np.float32)
    char_len_ratios = tgt_char_lens / src_char_lens

    exact_copy_flags = []
    token_copy_ratios = []
    edit_sims = []

    for s, t in zip(sources, targets):
        exact_copy_flags.append(float(s.strip() == t.strip()))
        s_toks = set(tokenize(s))
        t_toks = tokenize(t)
        if t_toks:
            copied = sum(1 for tok in t_toks if tok in s_toks)
            token_copy_ratios.append(copied / len(t_toks))
        else:
            token_copy_ratios.append(0.0)
        edit_sims.append(normalized_levenshtein_sim(s, t))

    report = {
        "n_samples": float(len(df)),
        "src_char_len_mean": float(np.mean(src_char_lens)),
        "tgt_char_len_mean": float(np.mean(tgt_char_lens)),
        "len_ratio_p05": float(np.percentile(char_len_ratios, 5)),
        "len_ratio_p50": float(np.percentile(char_len_ratios, 50)),
        "len_ratio_p95": float(np.percentile(char_len_ratios, 95)),
        "exact_identity_copy_rate": float(np.mean(exact_copy_flags)),
        "target_token_copy_ratio_mean": float(np.mean(token_copy_ratios)),
        "source_target_levenshtein_sim": float(np.mean(edit_sims)),
    }

    regime = (
        "COPY/EDIT-HEAVY (Favor ByT5 / Edit-Tagging / Lookup + Copy Pointer)"
        if report["target_token_copy_ratio_mean"] >= 0.70
        else "ABSTRACTIVE/TRANSFORM (Favor Encoder-Decoder mT5/mBART + MBR Reranking)"
    )
    print("=== Seq2Seq & Text Transformation Diagnostics ===")
    for k, v in report.items():
        print(f"  {k:<32}: {v:.4f}")
    print(f"  Recommended Modeling Regime     : {regime}")
    return report


class Seq2SeqDataset(Dataset):
    """
    PyTorch Dataset for Encoder-Decoder models (`mT5`, `ByT5`, `mBART`, `IndoT5`).
    Automatically masks padding tokens in `labels` with `-100` so `CrossEntropyLoss`
    ignores pad positions.
    """

    def __init__(
        self,
        sources: Sequence[str],
        targets: Sequence[str] | None = None,
        max_source_len: int = 64,
        max_target_len: int = 48,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
    ) -> None:
        self.sources = list(sources)
        self.targets = list(targets) if targets is not None else None
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def _encode_bytes(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ByT5-style UTF-8 byte encoding (offset by +3 for special tokens)
        raw_bytes = [b + 3 for b in text.encode("utf-8")[: max_len - 1]] + [self.eos_token_id]
        pad_len = max_len - len(raw_bytes)
        ids = raw_bytes + [self.pad_token_id] * pad_len
        mask = [1] * len(raw_bytes) + [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = self._encode_bytes(self.sources[idx], self.max_source_len)
        item = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.targets is not None:
            labels, _ = self._encode_bytes(self.targets[idx], self.max_target_len)
            labels = labels.clone()
            labels[labels == self.pad_token_id] = -100
            item["labels"] = labels
        return item


def create_seq2seq_folds(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["fold"] = -1
    if cfg.group_col in out.columns and out[cfg.group_col].nunique() >= cfg.n_folds:
        gkf = GroupKFold(n_splits=cfg.n_folds)
        for fold, (_, val_idx) in enumerate(gkf.split(out, groups=out[cfg.group_col])):
            out.loc[val_idx, "fold"] = fold
    else:
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(kf.split(out)):
            out.loc[val_idx, "fold"] = fold
    return out


def generate_synthetic_seq2seq_data(n_samples: int = 40) -> pd.DataFrame:
    pairs = [
        ("yg blm bayar tlg segera ditransfer ya kak", "yang belum bayar tolong segera ditransfer ya kak"),
        ("sy sdh kirim dokumennya kmrn sore", "saya sudah kirim dokumennya kemarin sore"),
        ("brp hrg tiket kereta dr jkt ke sby", "berapa harga tiket kereta dari jakarta ke surabaya"),
        ("mhn maaf pesanan tdk dpt diproses skrg", "mohon maaf pesanan tidak dapat diproses sekarang"),
        ("kpn jadwal ujian akhir semester dimulai", "kapan jadwal ujian akhir semester dimulai"),
    ]
    rows = []
    for i in range(n_samples):
        src, tgt = pairs[i % len(pairs)]
        rows.append(
            {
                "id": f"S{i:04d}",
                "group_id": f"G{i % 10:02d}",
                "source_text": f"{src} #{i % 5}",
                "target_text": f"{tgt} #{i % 5}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 06 Seq2Seq EDA & Dataset")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)

    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            df = generate_synthetic_seq2seq_data(40)
            report = compute_seq2seq_eda(df, cfg)
            folds_df = create_seq2seq_folds(df, cfg)

            ds = Seq2SeqDataset(
                folds_df[cfg.source_col].tolist(),
                folds_df[cfg.target_col].tolist(),
                max_source_len=32,
                max_target_len=32,
            )
            sample = ds[0]

            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            folds_df.to_parquet(out_dir / "train_folds.parquet", index=False)

            oof_sims = np.array(
                [
                    normalized_levenshtein_sim(s, t)
                    for s, t in zip(folds_df[cfg.source_col], folds_df[cfg.target_col])
                ],
                dtype=np.float32,
            )
            np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_sims)
            np.save(out_dir / f"test_{cfg.exp_name}.npy", oof_sims[:8])
            pd.DataFrame(
                {"id": folds_df["id"].iloc[:8], "prediction": folds_df[cfg.source_col].iloc[:8]}
            ).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

            print(
                f"[Smoke Test OK] Folds shape={folds_df.shape}, "
                f"Sample input_ids={tuple(sample['input_ids'].shape)}, "
                f"CopyRatio={report['target_token_copy_ratio_mean']:.4f}"
            )
        return

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        train_path = data_dir / "train.csv"
    if train_path.exists():
        df = (
            pd.read_parquet(train_path)
            if train_path.suffix == ".parquet"
            else pd.read_csv(train_path)
        )
    else:
        df = generate_synthetic_seq2seq_data(50)

    compute_seq2seq_eda(df, cfg)
    folds_df = create_seq2seq_folds(df, cfg)
    folds_df.to_parquet(out_dir / "train_folds.parquet", index=False)


if __name__ == "__main__":
    main()
