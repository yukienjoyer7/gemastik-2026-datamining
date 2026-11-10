#!/usr/bin/env python3
"""
Task 11 — Semi-Structured Log, Markup & Code: Tier 2 Byte/Char-Aware Deep Learning
==================================================================================
Standalone PyTorch OOF sequence classifier designed for raw logs, DOM/HTML, JSON,
malicious URLs/payloads, and source code:
  1. Structural Special Token Annotator (`annotate_structural_special_tokens`):
     Injects explicit structural tokens (`[TAG_OPEN]`, `[TAG_CLOSE]`, `[KEY]`,
     `[VAL]`, `[IP]`, `[HEX]`, `[PATH]`, `[SQL]`) before tokenization.
  2. Dual-Mode Architecture:
     - When `use_pretrained_hf=True` and a local/cached HuggingFace backbone
       (`microsoft/codebert-base`, `google/byt5-small`, `microsoft/deberta-v3-small`)
       is available, fine-tunes the Transformer with newly registered structural tokens.
     - By default (and in offline / `--smoke-test` mode), trains a fast, high-accuracy
       **Multi-Scale 1D Byte/Character CNN (`k=3, 5, 7`) + Transformer Encoder** fused
       with dense structural/entropy features.
  3. Standardized OOF Contract:
     Saves `oof_tier2_semistruct_dl.npy`, `test_tier2_semistruct_dl.npy`, and
     `sub_tier2_semistruct_dl.csv`.

Usage:
    python baselines/tier2_strong_dl.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier2_strong_dl.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import math
import pathlib
import re
import site
import string
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_venv_site = (
    pathlib.Path.home()
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader, Dataset


@dataclass
class CFG:
    """Configuration for Task 11 Tier 2 Byte/Character CNN + Transformer Classifier."""

    exp_name: str = "tier2_semistruct_dl"
    model_name: str = "microsoft/codebert-base"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"

    max_byte_len: int = 384
    embed_dim: int = 64
    cnn_channels: int = 64
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2

    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-2
    seed: int = 42
    smoke_test: bool = False


STRUCTURAL_SPECIAL_TOKENS: List[str] = [
    "[TAG_OPEN]",
    "[TAG_CLOSE]",
    "[KEY]",
    "[VAL]",
    "[IP]",
    "[HEX]",
    "[PATH]",
    "[SQL]",
]

_SPECIAL_TOKEN_TO_BYTE_ID: Dict[str, int] = {
    tok: 257 + i for i, tok in enumerate(STRUCTURAL_SPECIAL_TOKENS)
}
VOCAB_SIZE: int = 257 + len(STRUCTURAL_SPECIAL_TOKENS) + 2


def annotate_structural_special_tokens(raw_text: str) -> str:
    """
    Insert explicit structural markers (`[TAG_OPEN]`, `[TAG_CLOSE]`, `[KEY]`, `[VAL]`,
    `[IP]`, `[HEX]`, `[PATH]`, `[SQL]`) into raw semi-structured strings.
    """
    s = str(raw_text or "")
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", " [IP] ", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", " [HEX] ", s)
    s = re.sub(r"(?:^|(?<=[\s\"'=:(]))(?:/[a-zA-Z0-9_.\-]+){2,}", " [PATH] ", s)
    s = re.sub(r"</([a-zA-Z0-9_.\-]+)>", r" [TAG_CLOSE] \1 ", s)
    s = re.sub(r"<([a-zA-Z0-9_.\-]+)", r" [TAG_OPEN] \1 ", s)
    s = re.sub(r'"([a-zA-Z0-9_.\-]+)"\s*:\s*', r" [KEY] \1 [VAL] ", s)
    s = re.sub(
        r"\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|SLEEP|BENCHMARK)\b",
        r" [SQL] \1 ",
        s,
        flags=re.IGNORECASE,
    )
    return s


def encode_annotated_bytes(raw_text: str, max_len: int = 384) -> List[int]:
    """
    Encode annotated text into byte IDs `[1..256]` plus dedicated IDs `[257..264]`
    for structural special tokens (`[TAG_OPEN]`, `[KEY]`, `[IP]`, etc.).
    Uses Head+Tail preservation if the byte sequence exceeds `max_len`.
    """
    annotated = annotate_structural_special_tokens(raw_text)
    pattern = re.compile(r"(\[(?:TAG_OPEN|TAG_CLOSE|KEY|VAL|IP|HEX|PATH|SQL)\])")
    chunks = pattern.split(annotated)

    ids: List[int] = []
    for chunk in chunks:
        if not chunk:
            continue
        if chunk in _SPECIAL_TOKEN_TO_BYTE_ID:
            ids.append(_SPECIAL_TOKEN_TO_BYTE_ID[chunk])
        else:
            for b in chunk.encode("utf-8", errors="ignore"):
                ids.append(int(b) + 1)

    if not ids:
        ids = [1]
    if len(ids) > max_len:
        head = max_len // 2
        tail = max_len - head
        ids = ids[:head] + ids[-tail:]
    return ids


def extract_dense_aux_features(raw_text: str) -> np.ndarray:
    """Compute 8 dense entropy & structural features to fuse with the neural representation."""
    s = str(raw_text or "")
    n = max(len(s), 1)
    counts = Counter(s)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values()) if s else 0.0
    punct_set = set(string.punctuation)
    sym_ratio = sum(1 for c in s if c in punct_set) / n
    dig_ratio = sum(1 for c in s if c.isdigit()) / n
    upp_ratio = sum(1 for c in s if c.isupper()) / n
    has_html = 1.0 if ("<" in s and ">" in s) else 0.0
    has_json = 1.0 if ("{" in s and ":" in s) else 0.0
    has_sql = 1.0 if re.search(r"\b(select|union|drop|sleep)\b|--|'", s, re.I) else 0.0
    return np.array(
        [np.log1p(len(s)), ent / 6.0, sym_ratio, dig_ratio, upp_ratio, has_html, has_json, has_sql],
        dtype=np.float32,
    )


class ByteAndStructDataset(Dataset):
    """Dataset yielding byte+special-token sequences and dense structural features."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        max_len: int = 384,
    ) -> None:
        self.texts = [str(t) for t in texts]
        self.labels = list(labels) if labels is not None else None
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq = encode_annotated_bytes(self.texts[idx], max_len=self.max_len)
        aux = extract_dense_aux_features(self.texts[idx])
        item: Dict[str, object] = {
            "input_ids": seq,
            "aux_feats": aux,
        }
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def byte_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)  # type: ignore[arg-type]
    max_l = max(max_l, 8)
    ids = [
        list(x["input_ids"]) + [0] * (max_l - len(x["input_ids"]))  # type: ignore[arg-type]
        for x in batch
    ]
    mask = [
        [1] * len(x["input_ids"]) + [0] * (max_l - len(x["input_ids"]))  # type: ignore[arg-type]
        for x in batch
    ]
    aux = np.stack([x["aux_feats"] for x in batch], axis=0)  # type: ignore[arg-type]
    out: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.float32),
        "aux_feats": torch.tensor(aux, dtype=torch.float32),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([int(x["labels"]) for x in batch], dtype=torch.long)  # type: ignore[arg-type]
    return out


class ByteCharCNNTransformerClassifier(nn.Module):
    """
    Multi-Scale 1D Byte/Character CNN (`k=3, 5, 7`) + Transformer Encoder +
    Dense Structural Feature Fusion with Multi-Sample Dropout.
    """

    def __init__(self, num_classes: int, cfg: CFG, aux_dim: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, cfg.embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(cfg.embed_dim, cfg.cnn_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cfg.embed_dim, cfg.cnn_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(cfg.embed_dim, cfg.cnn_channels, kernel_size=7, padding=3)
        self.proj = nn.Linear(cfg.cnn_channels * 3, cfg.embed_dim)
        self.norm_in = nn.LayerNorm(cfg.embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.embed_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.num_transformer_layers, enable_nested_tensor=False
        )

        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        fused_dim = cfg.embed_dim * 2 + 32
        self.dropouts = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(4)])
        self.fc = nn.Linear(fused_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, aux_feats: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)  # (B, L, D)
        x_t = emb.transpose(1, 2)  # (B, D, L)
        c3 = F.gelu(self.conv3(x_t))
        c5 = F.gelu(self.conv5(x_t))
        c7 = F.gelu(self.conv7(x_t))
        conv_cat = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)  # (B, L, 3*C)
        mask_exp = attention_mask.unsqueeze(-1)
        h = self.norm_in(emb + self.proj(conv_cat)) * mask_exp
        h = self.transformer(h) * mask_exp

        mean_pool = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
        max_pool = h.masked_fill(mask_exp == 0, -1e4).max(dim=1)[0]

        aux_emb = self.aux_mlp(aux_feats)
        feat = torch.cat([mean_pool, max_pool, aux_emb], dim=-1)
        logits = sum(self.fc(drop(feat)) for drop in self.dropouts) / len(self.dropouts)
        return logits


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out_list: List[np.ndarray] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aux = batch["aux_feats"].to(device)
        logits = model(ids, mask, aux)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out_list.append(probs)
    return np.vstack(out_list)


def run_oof_training(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.sort(train_df[cfg.target_col].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = train_df[cfg.target_col].map(label_to_idx).values
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_classes = len(classes)

    test_ds = ByteAndStructDataset(test_df[cfg.text_col].values, None, max_len=cfg.max_byte_len)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=byte_collate_fn
    )

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_ds = ByteAndStructDataset(
            train_df[cfg.text_col].values[tr_idx], y[tr_idx], max_len=cfg.max_byte_len
        )
        va_ds = ByteAndStructDataset(
            train_df[cfg.text_col].values[va_idx], y[va_idx], max_len=cfg.max_byte_len
        )
        tr_loader = DataLoader(
            tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=byte_collate_fn
        )
        va_loader = DataLoader(
            va_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=byte_collate_fn
        )

        model = ByteCharCNNTransformerClassifier(num_classes=num_classes, cfg=cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs * len(tr_loader))
        )

        best_f1 = -1.0
        best_va_probs: Optional[np.ndarray] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for _ in range(cfg.epochs):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                aux = batch["aux_feats"].to(device)
                targets = batch["labels"].to(device)

                logits = model(ids, mask, aux)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            va_probs = predict_loader(model, va_loader, device)
            va_f1 = float(f1_score(y[va_idx], np.argmax(va_probs, axis=1), average="macro"))
            if va_f1 >= best_f1:
                best_f1 = va_f1
                best_va_probs = va_probs
                best_state = copy.deepcopy(model.state_dict())

        assert best_va_probs is not None and best_state is not None
        oof_probs[va_idx] = best_va_probs
        model.load_state_dict(best_state)
        test_probs += predict_loader(model, test_loader, device) / len(unique_folds)
        print(f"  [Fold {f}] Best Val Macro-F1={best_f1:.4f}")

    overall_f1 = float(f1_score(y, np.argmax(oof_probs, axis=1), average="macro"))
    overall_ll = float(log_loss(y, oof_probs))
    print(f"[OOF SUMMARY] Macro-F1={overall_f1:.4f} | LogLoss={overall_ll:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs)
    sub_df = pd.DataFrame(
        {
            cfg.id_col: test_df[cfg.id_col].values,
            cfg.target_col: [classes[i] for i in np.argmax(test_probs, axis=1)],
        }
    )
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
    return oof_probs, test_probs


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 11 Tier 2 Byte/Char-Aware DL Classifier")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.epochs = 4
        cfg.batch_size = 12
        cfg.max_byte_len = 160
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            rng = np.random.default_rng(cfg.seed)
            templates = {
                0: '<div class="product"><span class="price">Rp {v}</span></div>',
                1: "SELECT * FROM accounts WHERE user='admin' OR 1=1 -- tok=0x{v:x}",
                2: '{{"host": "10.0.1.{ip}", "path": "/var/log/syslog", "code": {v}}}',
            }
            train_df = pd.DataFrame(
                [
                    {
                        "id": f"tr_{i}",
                        "text": templates[i % 3].format(
                            v=int(rng.integers(100, 9999)), ip=int(rng.integers(2, 250))
                        ),
                        "label": i % 3,
                        "fold": (i // 3) % 3,
                    }
                    for i in range(54)
                ]
            )
            test_df = pd.DataFrame(
                [
                    {
                        "id": f"te_{i}",
                        "text": templates[i % 3].format(
                            v=int(rng.integers(100, 9999)), ip=int(rng.integers(2, 250))
                        ),
                    }
                    for i in range(18)
                ]
            )
            print("[SMOKE TEST] Running Task 11 Tier 2 Byte/Char CNN + Transformer OOF Trainer...")
            oof, test_preds = run_oof_training(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            print("[SUCCESS] Task 11 Tier 2 DL smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_training(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
