#!/usr/bin/env python3
"""
Task 12 — Symbolic, Chemical (SMILES) & Biological Sequences: Tier 2 1D Dilated ResNet + TTA
============================================================================================
Standalone PyTorch OOF trainer implementing:
  1. **1D Dilated Multi-Scale ResNet (`k=3, 5, 7, 11`) + Squeeze-and-Excitation (SE) +
     Self-Attention Sequence Encoder** operating directly on integer-encoded SMILES
     atom/bond tokens or biological residue tokens, fused with dense physicochemical
     features.
  2. **Test-Time Augmentation (TTA)**:
     At validation and test time, averages predictions across `num_tta=4` sequence
     enumerations (non-canonical SMILES branch-swap / cyclic rotations or DNA/RNA
     reverse-complements).
  3. Standardized OOF Contract:
     Saves `oof_tier2_seq_dl.npy`, `test_tier2_seq_dl.npy`, and `sub_tier2_seq_dl.csv`.

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
    """Configuration for Task 12 Tier 2 1D Dilated ResNet + SE + TTA Trainer."""

    exp_name: str = "tier2_seq_dl"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "sequence"
    target_col: str = "label"
    fold_col: str = "fold"

    max_seq_len: int = 256
    embed_dim: int = 48
    branch_channels: int = 24
    hidden_channels: int = 96
    dilations: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.2
    num_tta: int = 4

    epochs: int = 6
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-2
    seed: int = 42
    smoke_test: bool = False


_SMILES_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def tokenize_sequence(seq: str) -> List[str]:
    s = "".join(str(seq or "").split())
    if not s:
        return []
    if any(ch in s for ch in ("=", "#", "(", ")", "[", "]", "1", "2", "c")):
        toks = _SMILES_TOKEN_REGEX.findall(s)
        return toks if toks else list(s)
    return list(s)


def enumerate_tta_views(seq: str, num_views: int = 4) -> List[str]:
    """
    Generate `num_views` TTA representations:
      - For SMILES: sister-branch swaps `(A)(B) -> (B)(A)` and cyclic token shifts.
      - For DNA/RNA: original + reverse-complement + cyclic shifts.
    """
    s = "".join(str(seq or "").split())
    if not s or num_views <= 1:
        return [s]

    views: List[str] = [s]
    upper_set = set(s.upper())
    if upper_set and upper_set.issubset({"A", "C", "G", "T", "U", "N"}) and len(s) >= 6:
        rc = s.translate(_DNA_COMP)[::-1]
        views.append(rc)

    tokens = tokenize_sequence(s)
    adj_branch = re.compile(r"\(([^()]+)\)\(([^()]+)\)")
    if adj_branch.search(s):
        views.append(adj_branch.sub(r"(\2)(\1)", s, count=1))

    v_idx = 1
    while len(views) < num_views:
        if len(tokens) > 3:
            shift = (v_idx * max(1, len(tokens) // num_views)) % len(tokens)
            views.append("".join(tokens[shift:] + tokens[:shift]))
        else:
            views.append(s[::-1])
        v_idx += 1

    return views[:num_views]


def extract_dense_seq_features(seq: str) -> np.ndarray:
    """Compute 10 fast physicochemical & structural summary features per sequence."""
    s = "".join(str(seq or "").split())
    n = max(len(s), 1)
    counts = Counter(s)
    ent = -sum((v / n) * math.log2(v / n) for v in counts.values()) if s else 0.0
    c_arom = sum(1 for ch in s if ch in "cnos") / n
    c_aliph = (s.count("C") - s.count("Cl")) / n
    n_tot = (s.count("N") + s.count("n")) / n
    o_tot = (s.count("O") + s.count("o")) / n
    hal_tot = (s.count("F") + s.count("Cl") + s.count("Br") + s.count("I")) / n
    s_tot = (s.count("S") + s.count("s")) / n
    bonds = (s.count("=") + 2 * s.count("#")) / n
    branches_rings = (s.count("(") + sum(s.count(d) for d in "123456789") * 0.5) / n
    return np.array(
        [
            np.log1p(len(s)),
            ent / 5.0,
            c_arom,
            c_aliph,
            n_tot,
            o_tot,
            hal_tot,
            s_tot,
            bonds,
            branches_rings,
        ],
        dtype=np.float32,
    )


class SequenceVocabulary:
    def __init__(self, max_len: int = 256) -> None:
        self.max_len = max_len
        self.tok2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}

    def fit(self, sequences: Sequence[str]) -> "SequenceVocabulary":
        for s in sequences:
            for t in tokenize_sequence(s):
                if t not in self.tok2id:
                    self.tok2id[t] = len(self.tok2id)
        return self

    def encode(self, seq: str) -> List[int]:
        toks = tokenize_sequence(seq)[: self.max_len - 2]
        return [2] + [self.tok2id.get(t, 1) for t in toks] + [3]

    @property
    def size(self) -> int:
        return max(len(self.tok2id), 16)


class SymbolicSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[str],
        labels: Optional[Sequence[int]],
        vocab: SequenceVocabulary,
        augment: bool = False,
    ) -> None:
        self.sequences = [str(s) for s in sequences]
        self.labels = list(labels) if labels is not None else None
        self.vocab = vocab
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        raw = self.sequences[idx]
        if self.augment:
            views = enumerate_tta_views(raw, num_views=2)
            seq_str = views[idx % len(views)]
        else:
            seq_str = raw
        ids = self.vocab.encode(seq_str)
        aux = extract_dense_seq_features(raw)
        item: Dict[str, object] = {"input_ids": ids, "aux_feats": aux}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def seq_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)  # type: ignore[arg-type]
    max_l = max(max_l, 12)
    ids = [
        list(x["input_ids"]) + [0] * (max_l - len(x["input_ids"]))  # type: ignore[arg-type]
        for x in batch
    ]
    mask = [
        [1.0] * len(x["input_ids"]) + [0.0] * (max_l - len(x["input_ids"]))  # type: ignore[arg-type]
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


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1)
        pooled = (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1e-6)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))).unsqueeze(-1)
        return x * scale


class DilatedResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.se = SqueezeExcitation1D(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        res = x
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        h = self.se(h, mask) * mask.unsqueeze(1)
        return F.silu(h + res)


class DilatedResNet1DSequenceModel(nn.Module):
    """
    1D Dilated Multi-Scale ResNet (`k=3, 5, 7, 11`) + Squeeze-and-Excitation +
    Self-Attention + Mean/Max/Attention Pooling fused with physicochemical features.
    """

    def __init__(self, vocab_size: int, num_classes: int, cfg: CFG, aux_dim: int = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        b_ch = cfg.branch_channels
        self.stem3 = nn.Conv1d(cfg.embed_dim, b_ch, kernel_size=3, padding=1)
        self.stem5 = nn.Conv1d(cfg.embed_dim, b_ch, kernel_size=5, padding=2)
        self.stem7 = nn.Conv1d(cfg.embed_dim, b_ch, kernel_size=7, padding=3)
        self.stem11 = nn.Conv1d(cfg.embed_dim, b_ch, kernel_size=11, padding=5)

        h_ch = cfg.hidden_channels
        self.stem_proj = nn.Sequential(
            nn.Conv1d(b_ch * 4, h_ch, kernel_size=1),
            nn.GroupNorm(min(8, h_ch), h_ch),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedResBlock1D(h_ch, dilation=d, dropout=cfg.dropout * 0.5) for d in cfg.dilations]
        )
        self.self_attn = nn.MultiheadAttention(h_ch, num_heads=4, dropout=cfg.dropout * 0.5, batch_first=True)
        self.attn_norm = nn.LayerNorm(h_ch)
        self.attn_pool = nn.Sequential(nn.Linear(h_ch, h_ch // 2), nn.Tanh(), nn.Linear(h_ch // 2, 1))

        self.aux_mlp = nn.Sequential(nn.Linear(aux_dim, 32), nn.LayerNorm(32), nn.SiLU())
        pooled_dim = h_ch * 3 + 32
        self.head_norm = nn.LayerNorm(pooled_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(4)])
        self.fc = nn.Linear(pooled_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, aux_feats: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embedding(input_ids).transpose(1, 2)
        s = torch.cat(
            [
                F.silu(self.stem3(emb)),
                F.silu(self.stem5(emb)),
                F.silu(self.stem7(emb)),
                F.silu(self.stem11(emb)),
            ],
            dim=1,
        )
        x = self.stem_proj(s) * attention_mask.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, attention_mask)

        seq = (x * attention_mask.unsqueeze(1)).transpose(1, 2)
        attn_out, _ = self.self_attn(seq, seq, seq, need_weights=False)
        seq = self.attn_norm(seq + attn_out) * attention_mask.unsqueeze(-1)

        m_exp = attention_mask.unsqueeze(-1)
        mean_p = (seq * m_exp).sum(dim=1) / m_exp.sum(dim=1).clamp(min=1e-6)
        max_p = seq.masked_fill(m_exp == 0, -1e4).max(dim=1)[0]
        scores = self.attn_pool(seq).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
        att_p = (seq * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)

        feat = self.head_norm(torch.cat([mean_p, max_p, att_p, self.aux_mlp(aux_feats)], dim=-1))
        logits = sum(self.fc(d(feat)) for d in self.dropouts) / len(self.dropouts)
        return logits


@torch.no_grad()
def predict_with_sequence_tta(
    model: nn.Module,
    sequences: Sequence[str],
    vocab: SequenceVocabulary,
    cfg: CFG,
    device: torch.device,
) -> np.ndarray:
    """
    Predict class probabilities using Test-Time Augmentation (TTA) across `cfg.num_tta`
    enumerated SMILES / reverse-complement views.
    """
    model.eval()
    num_tta = max(1, cfg.num_tta)
    tta_views_per_sample = [enumerate_tta_views(s, num_views=num_tta) for s in sequences]

    accum_probs: Optional[np.ndarray] = None
    for v_idx in range(num_tta):
        view_seqs = [views[v_idx] for views in tta_views_per_sample]
        ds = SymbolicSequenceDataset(view_seqs, None, vocab=vocab, augment=False)
        loader = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=seq_collate_fn)
        fold_view_probs: List[np.ndarray] = []
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            aux = batch["aux_feats"].to(device)
            probs = torch.softmax(model(ids, mask, aux), dim=-1).cpu().numpy()
            fold_view_probs.append(probs)
        view_arr = np.vstack(fold_view_probs)
        if accum_probs is None:
            accum_probs = view_arr / num_tta
        else:
            accum_probs += view_arr / num_tta

    assert accum_probs is not None
    return accum_probs


def run_oof_training(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_col = cfg.text_col if cfg.text_col in train_df.columns else "text"
    classes = np.sort(train_df[cfg.target_col].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = train_df[cfg.target_col].map(label_to_idx).values
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_classes = len(classes)

    all_train_seqs = train_df[seq_col].fillna("").astype(str).tolist()
    all_test_seqs = test_df[seq_col].fillna("").astype(str).tolist()

    vocab = SequenceVocabulary(max_len=cfg.max_seq_len).fit(all_train_seqs + all_test_seqs)

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_seqs = [all_train_seqs[i] for i in tr_idx]
        va_seqs = [all_train_seqs[i] for i in va_idx]

        tr_ds = SymbolicSequenceDataset(tr_seqs, y[tr_idx], vocab=vocab, augment=True)
        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=seq_collate_fn)

        model = DilatedResNet1DSequenceModel(
            vocab_size=vocab.size + 4, num_classes=num_classes, cfg=cfg
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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

            va_probs = predict_with_sequence_tta(model, va_seqs, vocab, cfg, device)
            va_f1 = float(f1_score(y[va_idx], np.argmax(va_probs, axis=1), average="macro"))
            if va_f1 >= best_f1:
                best_f1 = va_f1
                best_va_probs = va_probs
                best_state = copy.deepcopy(model.state_dict())

        assert best_va_probs is not None and best_state is not None
        oof_probs[va_idx] = best_va_probs
        model.load_state_dict(best_state)
        test_probs += predict_with_sequence_tta(model, all_test_seqs, vocab, cfg, device) / len(unique_folds)
        print(f"  [Fold {f}] Best Val Macro-F1 (with {cfg.num_tta}x TTA)={best_f1:.4f}")

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
    parser = argparse.ArgumentParser(description="Task 12 Tier 2 1D Dilated ResNet + SE + TTA Trainer")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--num-tta", type=int, default=CFG.num_tta)
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        num_tta=args.num_tta,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.epochs = 4
        cfg.batch_size = 12
        cfg.num_tta = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            rng = np.random.default_rng(cfg.seed)
            scaffolds = {
                0: "COc1cc(C=CC(=O)CC(=O)c2ccc({r})cc2)ccc1O",
                1: "CC(=O)Nc1ccc(S(=O)(=O)N{r})cc1F",
                2: "CCCCCCCC({r})C(=O)O",
            }
            subs = ["C", "CC", "O", "F", "Cl", "Br"]
            train_df = pd.DataFrame(
                [
                    {
                        "id": f"tr_{i}",
                        "sequence": scaffolds[i % 3].format(r=str(rng.choice(subs))),
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
                        "sequence": scaffolds[i % 3].format(r=str(rng.choice(subs))),
                    }
                    for i in range(18)
                ]
            )
            print("[SMOKE TEST] Running Task 12 Tier 2 1D Dilated ResNet + SE + SMILES TTA Trainer...")
            oof, test_preds = run_oof_training(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            print("[SUCCESS] Task 12 Tier 2 DL smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_training(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
