#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Tier 2 Cross-Encoder Transformer OOF Pipeline
=============================================================================
Standalone PyTorch + HuggingFace Cross-Encoder OOF pipeline implementing:
  1. Both binary/multiclass pair classification (`BCEWithLogitsLoss` / `CrossEntropyLoss`)
     and continuous semantic similarity (`PearsonCorrelationLoss` + `MSELoss` hybrid).
  2. **Context-Grouping Augmentation** (the #1 winning trick from Kaggle U.S. Patent
     Phrase to Phrase Matching): appends other peer targets sharing the same
     anchor/context group to the prompt (`[CLS] context [SEP] text_a [SEP] text_b [SEP] peers...`)
     so self-attention calibrates similarity relative to the anchor's candidate set.
  3. **Symmetric Test-Time Augmentation (TTA)**: averages predictions across forward
     `(text_a, text_b)` and reversed `(text_b, text_a)` passes (`0.5 * f(A, B) + 0.5 * f(B, A)`).
  4. Offline-compatible `MiniCrossEncoderBackbone` for fast (<5s) `--smoke-test`.
  5. Standardized OOF & prediction contract: saves `oof_<exp_name>.npy`,
     `test_<exp_name>.npy`, and `sub_<exp_name>.csv`.

Usage:
    python baselines/tier2_strong_dl.py --task-mode similarity --train-path train_folds.csv --test-path test.csv
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
from collections import defaultdict
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
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class CFG:
    """Configuration for Tier 2 Cross-Encoder Transformer OOF Pipeline."""

    exp_name: str = "tier2_cross_encoder"
    model_name: str = "microsoft/deberta-v3-small"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_a_col: str = "text_a"
    text_b_col: str = "text_b"
    context_col: str = "context"
    target_col: str = "label"
    fold_col: str = "fold"

    # Task mode: "similarity" (Pearson + MSE) | "binary" (BCEWithLogits) | "multiclass" (CE)
    task_mode: str = "similarity"
    num_classes: int = 1
    pearson_loss_weight: float = 0.7
    mse_loss_weight: float = 0.3

    # Context-grouping & TTA tricks
    use_context_grouping: bool = True
    max_group_peers: int = 6
    use_symmetric_tta: bool = True

    max_len: int = 256
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    dropout_rate: float = 0.2
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Context-Grouping Augmentation (U.S. Patent 1st Place Trick)
# =============================================================================


def build_context_group_prompts(
    df: pd.DataFrame,
    text_a_col: str,
    text_b_col: str,
    context_col: Optional[str] = None,
    max_peers: int = 6,
) -> List[str]:
    """
    Groups rows by `(context, text_a)` (or `text_a` if `context_col` is absent) and constructs
    a context-enriched string containing peer targets `text_b` from the same group:
      `"{context} ; peers: {peer_1}, {peer_2}, ..."`
    Allows the Cross-Encoder to attend over alternative candidates for the same anchor.
    """
    has_ctx = context_col is not None and context_col in df.columns
    group_to_targets: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    texts_a = df[text_a_col].fillna("").astype(str).tolist()
    texts_b = df[text_b_col].fillna("").astype(str).tolist()
    ctxs = (
        df[context_col].fillna("").astype(str).tolist()
        if has_ctx
        else ["" for _ in range(len(df))]
    )

    for c, a, b in zip(ctxs, texts_a, texts_b):
        if b not in group_to_targets[(c, a)]:
            group_to_targets[(c, a)].append(b)

    enriched_contexts: List[str] = []
    for c, a, b in zip(ctxs, texts_a, texts_b):
        peers = [p for p in group_to_targets[(c, a)] if p != b][:max_peers]
        peer_str = " | ".join(peers)
        if c and peer_str:
            enriched_contexts.append(f"{c} ; peers: {peer_str}")
        elif c:
            enriched_contexts.append(c)
        elif peer_str:
            enriched_contexts.append(f"peers: {peer_str}")
        else:
            enriched_contexts.append("")

    return enriched_contexts


# =============================================================================
# 2. Losses, Dataset, and Cross-Encoder Architecture
# =============================================================================


class PearsonMSEHybridLoss(nn.Module):
    """Weighted combination of `PearsonCorrelationLoss` and `MSELoss`."""

    def __init__(self, pearson_weight: float = 0.7, mse_weight: float = 0.3) -> None:
        super().__init__()
        self.pw = pearson_weight
        self.mw = mse_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = preds.view(-1).float()
        y = targets.view(-1).float()
        vx = x - x.mean()
        vy = y - y.mean()
        denom = torch.sqrt((vx.pow(2).sum() + 1e-6) * (vy.pow(2).sum() + 1e-6))
        pearson_loss = 1.0 - (vx * vy).sum() / denom
        mse_loss = F.mse_loss(torch.sigmoid(x), y)
        return self.pw * pearson_loss + self.mw * mse_loss


class MiniCrossEncoderBackbone(nn.Module):
    """Offline fallback Transformer encoder for Cross-Encoder smoke testing."""

    def __init__(self, vocab_size: int = 2048, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.seg_embeddings = nn.Embedding(2, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        x = self.embeddings(input_ids)
        if token_type_ids is not None:
            x = x + self.seg_embeddings(token_type_ids.clamp(0, 1))
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return SimpleNamespace(last_hidden_state=x)


class SimplePairTokenizer:
    cls_token_id: int = 1
    sep_token_id: int = 2
    pad_token_id: int = 0

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = re.findall(r"\w+|\S", str(text).lower())
        return [3 + (hash(t) % (self.vocab_size - 3)) for t in tokens]


class CrossEncoderDataset(Dataset):
    """Balanced-truncation Cross-Encoder dataset supporting forward `(A, B)` and swapped `(B, A)`."""

    def __init__(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
        contexts: Sequence[str],
        labels: Optional[Sequence[float]],
        tokenizer: Any,
        max_len: int = 256,
        swap_ab: bool = False,
    ) -> None:
        self.texts_a = list(texts_b if swap_ab else texts_a)
        self.texts_b = list(texts_a if swap_ab else texts_b)
        self.contexts = list(contexts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_id = int(getattr(tokenizer, "cls_token_id", 101) or 101)
        self.sep_id = int(getattr(tokenizer, "sep_token_id", 102) or 102)

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ctx_ids = list(self.tokenizer.encode(str(self.contexts[idx]), add_special_tokens=False))[:32]
        a_ids = list(self.tokenizer.encode(str(self.texts_a[idx]), add_special_tokens=False))
        b_ids = list(self.tokenizer.encode(str(self.texts_b[idx]), add_special_tokens=False))

        budget = max(4, self.max_len - 4 - len(ctx_ids))
        half = budget // 2
        if len(a_ids) + len(b_ids) > budget:
            if len(a_ids) <= half:
                b_ids = b_ids[: budget - len(a_ids)]
            elif len(b_ids) <= budget - half:
                a_ids = a_ids[: budget - len(b_ids)]
            else:
                a_ids = a_ids[:half]
                b_ids = b_ids[: budget - half]

        input_ids = (
            [self.cls_id]
            + ctx_ids
            + [self.sep_id]
            + a_ids
            + [self.sep_id]
            + b_ids
            + [self.sep_id]
        )
        seg0 = 1 + len(ctx_ids) + 1 + len(a_ids) + 1
        seg1 = len(b_ids) + 1
        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": [0] * seg0 + [1] * seg1,
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def collate_pairs(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)
    ids, mask, ttypes = [], [], []
    for x in batch:
        pad = max_l - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        mask.append(x["attention_mask"] + [0] * pad)
        ttypes.append(x["token_type_ids"] + [0] * pad)
    out = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(ttypes, dtype=torch.long),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.float32)
    return out


class CrossEncoderModel(nn.Module):
    """Cross-Encoder pooling `[CLS]` + MeanPooling + Segment-Difference representation."""

    def __init__(self, backbone: nn.Module, out_dim: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        h = int(backbone.config.hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.head = nn.Linear(h * 2, out_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        except TypeError:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state
        cls_vec = hidden[:, 0, :]
        m = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        mean_vec = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        feat = torch.cat([cls_vec, mean_vec], dim=-1)
        return sum(self.head(d(feat)) for d in self.dropouts) / len(self.dropouts)


# =============================================================================
# 3. OOF Training & Symmetric TTA Evaluation
# =============================================================================


def build_backbone_and_tokenizer(cfg: CFG) -> Tuple[nn.Module, Any]:
    if not cfg.smoke_test and HAS_TRANSFORMERS:
        try:
            tok = AutoTokenizer.from_pretrained(cfg.model_name)
            conf = AutoConfig.from_pretrained(cfg.model_name)
            return AutoModel.from_pretrained(cfg.model_name, config=conf), tok
        except Exception as exc:
            print(f"[WARN] Falling back to MiniCrossEncoderBackbone ({exc})")
    return MiniCrossEncoderBackbone(), SimplePairTokenizer()


@torch.no_grad()
def predict_with_symmetric_tta(
    model: nn.Module,
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    contexts: Sequence[str],
    tokenizer: Any,
    cfg: CFG,
    device: torch.device,
) -> np.ndarray:
    """Computes `0.5 * f(A, B) + 0.5 * f(B, A)` when `use_symmetric_tta=True`."""
    model.eval()
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    def _run_pass(swap: bool) -> np.ndarray:
        ds = CrossEncoderDataset(
            texts_a, texts_b, contexts, None, tokenizer, max_len=cfg.max_len, swap_ab=swap
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=lambda b: collate_pairs(b, pad_id=pad_id),
        )
        preds = []
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            if cfg.task_mode == "multiclass":
                out = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                out = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            preds.append(out)
        return np.concatenate(preds, axis=0)

    p_fwd = _run_pass(swap=False)
    if not cfg.use_symmetric_tta:
        return p_fwd
    p_rev = _run_pass(swap=True)
    return 0.5 * p_fwd + 0.5 * p_rev


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.use_context_grouping:
        tr_contexts = build_context_group_prompts(
            train_df, cfg.text_a_col, cfg.text_b_col, cfg.context_col, cfg.max_group_peers
        )
        te_contexts = build_context_group_prompts(
            test_df, cfg.text_a_col, cfg.text_b_col, cfg.context_col, cfg.max_group_peers
        )
    else:
        tr_contexts = (
            train_df[cfg.context_col].fillna("").astype(str).tolist()
            if cfg.context_col in train_df.columns
            else [""] * len(train_df)
        )
        te_contexts = (
            test_df[cfg.context_col].fillna("").astype(str).tolist()
            if cfg.context_col in test_df.columns
            else [""] * len(test_df)
        )

    y = train_df[cfg.target_col].values.astype(np.float32)
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    out_dim = cfg.num_classes if cfg.task_mode == "multiclass" else 1

    _, tokenizer = build_backbone_and_tokenizer(cfg)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    oof_shape = (len(train_df), out_dim) if out_dim > 1 else (len(train_df),)
    test_shape = (len(test_df), out_dim) if out_dim > 1 else (len(test_df),)
    oof_preds = np.zeros(oof_shape, dtype=np.float32)
    test_preds = np.zeros(test_shape, dtype=np.float32)

    tr_a = train_df[cfg.text_a_col].fillna("").astype(str).tolist()
    tr_b = train_df[cfg.text_b_col].fillna("").astype(str).tolist()
    te_a = test_df[cfg.text_a_col].fillna("").astype(str).tolist()
    te_b = test_df[cfg.text_b_col].fillna("").astype(str).tolist()

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_ds = CrossEncoderDataset(
            [tr_a[i] for i in tr_idx],
            [tr_b[i] for i in tr_idx],
            [tr_contexts[i] for i in tr_idx],
            y[tr_idx],
            tokenizer,
            max_len=cfg.max_len,
        )
        tr_loader = DataLoader(
            tr_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_pairs(b, pad_id=pad_id),
        )

        backbone, _ = build_backbone_and_tokenizer(cfg)
        model = CrossEncoderModel(backbone, out_dim=out_dim, dropout=cfg.dropout_rate).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.task_mode == "similarity":
            criterion: nn.Module = PearsonMSEHybridLoss(cfg.pearson_loss_weight, cfg.mse_loss_weight)
        elif cfg.task_mode == "binary":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        best_va = None
        best_state = None
        best_metric = -1e9

        for _ in range(cfg.epochs):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device),
                )
                targets = batch["labels"].to(device)
                if cfg.task_mode == "multiclass":
                    loss = criterion(logits, targets.long())
                else:
                    loss = criterion(logits.squeeze(-1), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            va_preds = predict_with_symmetric_tta(
                model,
                [tr_a[i] for i in va_idx],
                [tr_b[i] for i in va_idx],
                [tr_contexts[i] for i in va_idx],
                tokenizer,
                cfg,
                device,
            )
            if cfg.task_mode == "similarity":
                score = float(pearsonr(y[va_idx], va_preds)[0]) if np.std(va_preds) > 1e-8 else 0.0
            elif cfg.task_mode == "binary":
                score = float(roc_auc_score(y[va_idx].astype(int), va_preds))
            else:
                score = float(f1_score(y[va_idx].astype(int), np.argmax(va_preds, axis=1), average="macro"))

            if score >= best_metric or best_va is None:
                best_metric = score
                best_va = va_preds
                best_state = copy.deepcopy(model.state_dict())

        assert best_va is not None and best_state is not None
        oof_preds[va_idx] = best_va
        model.load_state_dict(best_state)
        test_preds += (
            predict_with_symmetric_tta(model, te_a, te_b, te_contexts, tokenizer, cfg, device)
            / len(unique_folds)
        )
        print(f"  [Fold {f}] Best Val Score ({cfg.task_mode})={best_metric:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_preds)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_preds)
    sub_col = np.argmax(test_preds, axis=1) if out_dim > 1 else test_preds
    pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values, cfg.target_col: sub_col}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )
    return oof_preds, test_preds


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Tier 2 Cross-Encoder Transformer OOF")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--task-mode", type=str, default=CFG.task_mode)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        task_mode=args.task_mode,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.epochs = 2
        cfg.batch_size = 8
        cfg.lr = 3e-3
        cfg.max_len = 64
        rng = np.random.default_rng(cfg.seed)
        train_rows = [
            {
                "id": f"tr_{i}",
                "context": f"CPC_{i % 3}",
                "text_a": "neural network gradient descent" if i % 2 == 0 else "fiscal policy bank",
                "text_b": "deep learning backpropagation" if i % 2 == 0 else "inflation rate bond",
                "label": float((i % 5) / 4.0),
                "fold": i % 3,
            }
            for i in range(36)
        ]
        test_rows = [
            {
                "id": f"te_{i}",
                "context": f"CPC_{i % 3}",
                "text_a": "neural network gradient descent",
                "text_b": "deep learning optimization",
            }
            for i in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 02 Tier 2 Cross-Encoder (Context Grouping + Symmetric TTA)...")
            oof, test_preds = run_oof_pipeline(pd.DataFrame(train_rows), pd.DataFrame(test_rows), cfg)
            assert oof.shape == (36,)
            assert test_preds.shape == (12,)
            print(f"[SUCCESS] Task 02 Tier 2 smoke test passed. OOF={oof.shape}, Test={test_preds.shape}")
    else:
        run_oof_pipeline(load_table(cfg.train_path), load_table(cfg.test_path), cfg)


if __name__ == "__main__":
    main()
