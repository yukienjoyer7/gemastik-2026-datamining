#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Tier 2 Transformer OOF Trainer
===================================================================
Standalone PyTorch + HuggingFace Multi-Label Transformer OOF trainer featuring:
  1. Selectable Multilabel Losses:
     - `BCEWithLogitsLoss`
     - `AsymmetricLossOptimized (ASL)` ($\gamma_- = 4, \gamma_+ = 0, m = 0.05$)
     - `ZLPRLoss` (Zero-bounded Log-sum-exp & Pairwise Rank-based Loss)
  2. **Label-Aware Prompting**:
     Prepends natural-language label names/descriptions to every document input
     (`"labels: economy, monetary policy, technology, ... [SEP] {text}"`) so encoder
     self-attention contextualizes tokens against the candidate label set.
  3. **Label-Wise Attention Head (`LabelWiseAttentionHead`)**:
     Each of the $L$ labels attends over sequence tokens with its own learned query vector.
  4. Offline-compatible `MiniMultilabelBackbone` for fast (<5s) `--smoke-test`.
  5. Standardized output contract: saves `oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     and `sub_<exp_name>.csv`.

Usage:
    python baselines/tier2_strong_dl.py --loss-type asl --train-path train_folds.csv --test-path test.csv
    python baselines/tier2_strong_dl.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import re
import site
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class CFG:
    """Configuration for Tier 2 Multilabel Transformer OOF Trainer."""

    exp_name: str = "tier2_multilabel_dl"
    model_name: str = "microsoft/deberta-v3-small"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    label_prefix: str = "label_"
    label_cols: Optional[Tuple[str, ...]] = None
    fold_col: str = "fold"

    # Loss & Head options
    loss_type: str = "asl"  # "bce" | "asl" | "zlpr"
    use_label_prompting: bool = True
    use_label_wise_attention: bool = True

    max_len: int = 256
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    dropout_rate: float = 0.2
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Multilabel Losses (ASL & ZLPR) and Label-Wise Attention Head
# =============================================================================


class AsymmetricLossOptimized(nn.Module):
    """Memory-efficient Asymmetric Loss (ASL) for imbalanced multilabel classification."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = (1.0 - xs_pos + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        pt = xs_pos * y + xs_neg * (1.0 - y)
        gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
        loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


class ZLPRLoss(nn.Module):
    """Zero-bounded Log-sum-exp & Pairwise Rank-based Loss (Su et al., 2022)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        neg_logits = logits.masked_fill(y > 0.5, -1e9)
        pos_logits = (-logits).masked_fill(y <= 0.5, -1e9)
        zeros = torch.zeros((logits.size(0), 1), device=logits.device, dtype=logits.dtype)
        l_neg = torch.logsumexp(torch.cat([zeros, neg_logits], dim=1), dim=1)
        l_pos = torch.logsumexp(torch.cat([zeros, pos_logits], dim=1), dim=1)
        return (l_neg + l_pos).mean()


class LabelWiseAttentionHead(nn.Module):
    """Dedicated query attention vector per label over token sequence representations."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.scale = hidden_size**-0.5
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.queries = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.out_weight = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keys = self.proj(hidden)
        scores = torch.matmul(self.queries.unsqueeze(0), keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = self.dropout(torch.matmul(attn, hidden))
        return (ctx * self.out_weight.unsqueeze(0)).sum(dim=-1) + self.out_bias


# =============================================================================
# 2. Label-Aware Prompting, Dataset & Model
# =============================================================================


def format_label_aware_prompt(text: str, label_names: Sequence[str], enabled: bool = True) -> str:
    """Prepends human-readable label names to the input text (`labels: l1, l2 ... | text`)."""
    if not enabled:
        return str(text)
    cleaned_labels = ", ".join(name.replace("label_", "").replace("_", " ") for name in label_names)
    return f"candidate labels: {cleaned_labels} | document: {text}"


class MiniMultilabelBackbone(nn.Module):
    """Offline fallback Transformer encoder for fast smoke testing."""

    def __init__(self, vocab_size: int = 2048, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True, dropout=0.1)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        x = self.embeddings(input_ids)
        kpm = attention_mask == 0
        for attn, norm in zip(self.layers, self.norms):
            out, _ = attn(x, x, x, key_padding_mask=kpm, need_weights=False)
            x = norm(x + out)
        return SimpleNamespace(last_hidden_state=x)


class SimpleHashTokenizer:
    cls_token_id: int = 1
    sep_token_id: int = 2
    pad_token_id: int = 0

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = re.findall(r"\w+|\S", str(text).lower())
        return [3 + (hash(t) % (self.vocab_size - 3)) for t in tokens]


class MultilabelDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        label_names: Sequence[str],
        targets: Optional[np.ndarray],
        tokenizer: Any,
        max_len: int = 256,
        use_prompt: bool = True,
    ) -> None:
        self.prompts = [format_label_aware_prompt(t, label_names, enabled=use_prompt) for t in texts]
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_id = int(getattr(tokenizer, "cls_token_id", 101) or 101)
        self.sep_id = int(getattr(tokenizer, "sep_token_id", 102) or 102)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ids = list(self.tokenizer.encode(self.prompts[idx], add_special_tokens=False))[: self.max_len - 2]
        input_ids = [self.cls_id] + ids + [self.sep_id]
        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item


def collate_multilabel(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)
    ids = [x["input_ids"] + [pad_id] * (max_l - len(x["input_ids"])) for x in batch]
    mask = [x["attention_mask"] + [0] * (max_l - len(x["attention_mask"])) for x in batch]
    out: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor(np.array([x["labels"] for x in batch]), dtype=torch.float32)
    return out


class MultilabelTransformer(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int, cfg: CFG) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_label_attn = cfg.use_label_wise_attention
        h = int(backbone.config.hidden_size)
        if self.use_label_attn:
            self.head: nn.Module = LabelWiseAttentionHead(h, num_labels, dropout=cfg.dropout_rate)
        else:
            self.head = nn.Sequential(nn.Dropout(cfg.dropout_rate), nn.Linear(h, num_labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.use_label_attn:
            return self.head(hidden, attention_mask)
        m = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        return self.head(pooled)


# =============================================================================
# 3. Training & OOF Loop
# =============================================================================


def resolve_label_columns(df: pd.DataFrame, cfg: CFG) -> List[str]:
    if cfg.label_cols is not None and len(cfg.label_cols) > 0:
        return list(cfg.label_cols)
    cols = [c for c in df.columns if c.startswith(cfg.label_prefix)]
    if not cols:
        excluded = {cfg.id_col, cfg.text_col, cfg.fold_col}
        cols = [c for c in df.columns if c not in excluded]
    return cols


def build_backbone_and_tokenizer(cfg: CFG) -> Tuple[nn.Module, Any]:
    if not cfg.smoke_test and HAS_TRANSFORMERS:
        try:
            tok = AutoTokenizer.from_pretrained(cfg.model_name)
            conf = AutoConfig.from_pretrained(cfg.model_name)
            return AutoModel.from_pretrained(cfg.model_name, config=conf), tok
        except Exception as exc:
            print(f"[WARN] Falling back to MiniMultilabelBackbone ({exc})")
    return MiniMultilabelBackbone(), SimpleHashTokenizer()


def build_criterion(loss_type: str) -> nn.Module:
    if loss_type == "asl":
        return AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    if loss_type == "zlpr":
        return ZLPRLoss()
    return nn.BCEWithLogitsLoss()


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(preds)


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_cols = resolve_label_columns(train_df, cfg)
    Y = train_df[label_cols].values.astype(np.float32)
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_labels = len(label_cols)

    _, tokenizer = build_backbone_and_tokenizer(cfg)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    te_ds = MultilabelDataset(
        test_df[cfg.text_col].values,
        label_cols,
        None,
        tokenizer,
        max_len=cfg.max_len,
        use_prompt=cfg.use_label_prompting,
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_multilabel(b, pad_id=pad_id),
    )

    oof_probs = np.zeros((len(train_df), num_labels), dtype=np.float32)
    test_probs = np.zeros((len(test_df), num_labels), dtype=np.float32)

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_ds = MultilabelDataset(
            train_df[cfg.text_col].values[tr_idx],
            label_cols,
            Y[tr_idx],
            tokenizer,
            max_len=cfg.max_len,
            use_prompt=cfg.use_label_prompting,
        )
        va_ds = MultilabelDataset(
            train_df[cfg.text_col].values[va_idx],
            label_cols,
            Y[va_idx],
            tokenizer,
            max_len=cfg.max_len,
            use_prompt=cfg.use_label_prompting,
        )
        tr_loader = DataLoader(
            tr_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_multilabel(b, pad_id=pad_id),
        )
        va_loader = DataLoader(
            va_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=lambda b: collate_multilabel(b, pad_id=pad_id),
        )

        backbone, _ = build_backbone_and_tokenizer(cfg)
        model = MultilabelTransformer(backbone, num_labels=num_labels, cfg=cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = build_criterion(cfg.loss_type)

        best_va = None
        best_state = None
        best_f1 = -1.0

        for _ in range(cfg.epochs):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            va_preds = predict_loader(model, va_loader, device)
            va_f1 = float(
                f1_score(Y[va_idx].astype(int), (va_preds >= 0.5).astype(int), average="micro", zero_division=0)
            )
            if va_f1 >= best_f1 or best_va is None:
                best_f1 = va_f1
                best_va = va_preds
                best_state = copy.deepcopy(model.state_dict())

        assert best_va is not None and best_state is not None
        oof_probs[va_idx] = best_va
        model.load_state_dict(best_state)
        test_probs += predict_loader(model, te_loader, device) / len(unique_folds)
        print(f"  [Fold {f}] Best Val Micro-F1 ({cfg.loss_type})={best_f1:.4f}")

    micro_f1 = float(f1_score(Y.astype(int), (oof_probs >= 0.5).astype(int), average="micro", zero_division=0))
    macro_f1 = float(f1_score(Y.astype(int), (oof_probs >= 0.5).astype(int), average="macro", zero_division=0))
    print(f"[OOF SUMMARY] Micro-F1={micro_f1:.4f} | Macro-F1={macro_f1:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs)
    sub_df = pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values})
    for idx, col in enumerate(label_cols):
        sub_df[col] = (test_probs[:, idx] >= 0.5).astype(int)
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
    return oof_probs, test_probs


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Tier 2 Multilabel Transformer OOF Trainer")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--loss-type", type=str, default=CFG.loss_type)
    parser.add_argument("--smoke-test", action="store_true", help="Run offline smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        loss_type=args.loss_type,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.epochs = 2
        cfg.batch_size = 8
        cfg.lr = 3e-3
        cfg.max_len = 64
        tr_df = pd.DataFrame(
            [
                {
                    "id": f"tr_{i}",
                    "text": "inflation bank fiscal bond" if i % 2 == 0 else "transformer neural network gpu",
                    "label_economy": int(i % 2 == 0),
                    "label_finance": int(i % 4 == 0),
                    "label_tech": int(i % 2 == 1),
                    "label_ai": int(i % 3 == 0),
                    "fold": i % 3,
                }
                for i in range(42)
            ]
        )
        te_df = pd.DataFrame(
            [
                {"id": f"te_{i}", "text": "inflation bank neural network"}
                for i in range(12)
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 04 Tier 2 Multilabel Transformer (ASL + Label Prompting + LabelWiseAttention)...")
            oof, test_p = run_oof_pipeline(tr_df, te_df, cfg)
            assert oof.shape == (42, 4)
            assert test_p.shape == (12, 4)
            print("[SUCCESS] Task 04 Tier 2 smoke test passed.")
    else:
        run_oof_pipeline(load_table(cfg.train_path), load_table(cfg.test_path), cfg)


if __name__ == "__main__":
    main()
