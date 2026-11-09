#!/usr/bin/env python3
"""
Task 01 — Text Classification: Tier 2 Strong Deep Learning OOF Trainer
======================================================================
Standalone PyTorch + HuggingFace Transformer OOF training script implementing:
  - Pretrained `AutoModel` backbone (e.g., `microsoft/deberta-v3-small`,
    `indobenchmark/indobert-base-p1`, `xlm-roberta-base`) with automatic fallback
    to a lightweight multi-layer `MiniTransformerBackbone` for offline `--smoke-test`.
  - Selectable sequence pooling: `mean`, `max`, `mean_max`, `attention`, `weighted_layer`.
  - `MultiSampleDropout` classification head.
  - Layer-wise Learning Rate Decay (`get_llrd_optimizer_params`) across encoder blocks.
  - Mixed Precision (`fp16` / `bf16` via `torch.amp.autocast`), gradient accumulation,
    cosine warmup scheduler, optional Adversarial Weight Perturbation (`AWP`), and `ModelEMA`.
  - Standardized OOF & prediction contract: saves `oof_<exp_name>.npy`,
    `test_<exp_name>.npy`, and `sub_<exp_name>.csv`.

Usage:
    python baselines/tier2_strong_dl.py --model-name microsoft/deberta-v3-small --train-path train_folds.csv --test-path test.csv
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
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class CFG:
    """Configuration for Tier 2 Transformer OOF Trainer."""

    exp_name: str = "tier2_deberta"
    model_name: str = "microsoft/deberta-v3-small"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"

    # Tokenization & Head+Tail truncation
    max_len: int = 512
    head_len: int = 128
    tail_len: int = 382

    # Architecture
    pooling_type: str = "mean"  # "mean" | "max" | "mean_max" | "attention" | "weighted_layer"
    num_dropouts: int = 5
    dropout_rate: float = 0.2

    # Optimization & LLRD
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 2
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    llrd_factor: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    amp_dtype: str = "fp16"  # "fp16" | "bf16" | "none"

    # Regularization tricks
    use_awp: bool = True
    awp_lr: float = 1e-2
    awp_eps: float = 1e-2
    awp_start_epoch: int = 1
    use_ema: bool = True
    ema_decay: float = 0.995

    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Offline-Compatible Fallback Backbone & Tokenizer
# =============================================================================


class MiniTransformerLayer(nn.Module):
    """Lightweight Transformer encoder block matching HuggingFace naming (`encoder.layer.i`)."""

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformerBackbone(nn.Module):
    """
    Self-contained PyTorch Transformer backbone with `embeddings` and `encoder.layer.{0..L-1}`
    so LLRD, WeightedLayerPooling, and AWP work identically offline or in smoke tests.
    """

    def __init__(self, vocab_size: int = 4096, hidden_size: int = 64, num_hidden_layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [MiniTransformerLayer(hidden_size, num_heads=4) for _ in range(num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        x = self.embeddings(input_ids)
        key_padding_mask = attention_mask == 0
        all_hidden = [x]
        for layer in self.encoder.layer:
            x = layer(x, key_padding_mask=key_padding_mask)
            all_hidden.append(x)
        return SimpleNamespace(last_hidden_state=x, hidden_states=tuple(all_hidden))


class SimpleHashTokenizer:
    """Fast deterministic word-hashing tokenizer for offline smoke tests."""

    cls_token_id: int = 1
    sep_token_id: int = 2
    pad_token_id: int = 0

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = re.findall(r"\w+|\S", str(text).lower())
        ids = [3 + (hash(t) % (self.vocab_size - 3)) for t in tokens]
        if add_special_tokens:
            return [self.cls_token_id] + ids + [self.sep_token_id]
        return ids


# =============================================================================
# 2. Dataset, Pooling, and Classifier Architecture
# =============================================================================


class HeadTailDataset(Dataset):
    """Dataset applying Head+Tail truncation for long sequences."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]],
        tokenizer: Any,
        max_len: int = 512,
        head_len: int = 128,
        tail_len: int = 382,
    ) -> None:
        self.texts = [str(t) for t in texts]
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_len = min(head_len, max(1, (max_len - 2) // 2))
        self.tail_len = min(tail_len, max(1, (max_len - 2) - self.head_len))
        self.cls_id = int(getattr(tokenizer, "cls_token_id", 101) or 101)
        self.sep_id = int(getattr(tokenizer, "sep_token_id", 102) or 102)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_ids = list(self.tokenizer.encode(self.texts[idx], add_special_tokens=False))
        budget = self.max_len - 2
        if len(raw_ids) > budget:
            raw_ids = raw_ids[: self.head_len] + raw_ids[-self.tail_len :]
        input_ids = [self.cls_id] + raw_ids + [self.sep_id]
        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def dynamic_collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_l = max(len(x["input_ids"]) for x in batch)
    ids = [x["input_ids"] + [pad_id] * (max_l - len(x["input_ids"])) for x in batch]
    mask = [x["attention_mask"] + [0] * (max_l - len(x["attention_mask"])) for x in batch]
    out: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return out


class TransformerClassifier(nn.Module):
    """Transformer Encoder + Selectable Pooling + Multi-Sample Dropout Head."""

    def __init__(self, backbone: nn.Module, num_classes: int, cfg: CFG) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_type = cfg.pooling_type
        hidden_size = int(backbone.config.hidden_size)
        num_layers = int(getattr(backbone.config, "num_hidden_layers", 4))

        if self.pooling_type == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            pool_out_dim = hidden_size
        elif self.pooling_type == "weighted_layer":
            self.num_pool_layers = min(4, num_layers)
            self.layer_weights = nn.Parameter(torch.ones(self.num_pool_layers))
            pool_out_dim = hidden_size
        elif self.pooling_type == "mean_max":
            pool_out_dim = hidden_size * 2
        else:
            pool_out_dim = hidden_size

        self.dropouts = nn.ModuleList([nn.Dropout(cfg.dropout_rate) for _ in range(cfg.num_dropouts)])
        self.fc = nn.Linear(pool_out_dim, num_classes)

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

    def _max_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).expand(hidden.size())
        return hidden.masked_fill(m == 0, -1e9).max(dim=1)[0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(self.pooling_type == "weighted_layer"),
        )
        last_hidden = outputs.last_hidden_state

        if self.pooling_type == "mean":
            feat = self._mean_pool(last_hidden, attention_mask)
        elif self.pooling_type == "max":
            feat = self._max_pool(last_hidden, attention_mask)
        elif self.pooling_type == "mean_max":
            feat = torch.cat(
                [self._mean_pool(last_hidden, attention_mask), self._max_pool(last_hidden, attention_mask)],
                dim=-1,
            )
        elif self.pooling_type == "attention":
            scores = self.attn_pool(last_hidden).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat = (last_hidden * weights).sum(dim=1)
        elif self.pooling_type == "weighted_layer":
            states = outputs.hidden_states[-self.num_pool_layers :]
            stacked = torch.stack(list(states), dim=0)
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            blended = (w * stacked).sum(dim=0)
            feat = self._mean_pool(blended, attention_mask)
        else:
            feat = self._mean_pool(last_hidden, attention_mask)

        logits = sum(self.fc(drop(feat)) for drop in self.dropouts) / len(self.dropouts)
        return logits


# =============================================================================
# 3. Layer-wise Learning Rate Decay (LLRD), AWP, EMA & Cosine Schedule
# =============================================================================


def get_llrd_optimizer_params(
    model: TransformerClassifier,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    llrd_factor: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Assigns exponentially decaying learning rates from top transformer layer down to embeddings:
      - Classifier head: `head_lr`
      - Layer `l` in `0..L-1`: `encoder_lr * (llrd_factor ** (L - 1 - l))`
      - Embeddings: `encoder_lr * (llrd_factor ** L)`
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm1.weight", "norm2.weight")
    num_layers = int(getattr(model.backbone.config, "num_hidden_layers", 4))
    param_groups: List[Dict[str, Any]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        if not name.startswith("backbone."):
            lr = head_lr
        elif "embeddings" in name:
            lr = encoder_lr * (llrd_factor**num_layers)
        else:
            layer_match = re.search(r"layer\.(\d+)\.", name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                lr = encoder_lr * (llrd_factor ** max(0, num_layers - 1 - layer_idx))
            else:
                lr = encoder_lr

        param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

    return param_groups


class AWP:
    """Adversarial Weight Perturbation for robust Transformer fine-tuning."""

    def __init__(self, model: nn.Module, adv_lr: float = 1e-2, adv_eps: float = 1e-2) -> None:
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup: Dict[str, torch.Tensor] = {}
        self.backup_eps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def perturb(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "weight" in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (self.backup[name] - grad_eps, self.backup[name] + grad_eps)
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    lo, hi = self.backup_eps[name]
                    param.data = torch.min(torch.max(param.data, lo), hi)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()
        self.backup_eps.clear()


class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, ema_v in self.ema.state_dict().items():
            model_v = model.state_dict()[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 4. Training & OOF Loop
# =============================================================================


def build_backbone_and_tokenizer(cfg: CFG) -> Tuple[nn.Module, Any]:
    if not cfg.smoke_test and HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            config = AutoConfig.from_pretrained(cfg.model_name)
            backbone = AutoModel.from_pretrained(cfg.model_name, config=config)
            return backbone, tokenizer
        except Exception as exc:
            print(f"[WARN] Could not load HuggingFace model '{cfg.model_name}' ({exc}); using MiniTransformerBackbone.")
    return MiniTransformerBackbone(vocab_size=2048, hidden_size=64, num_hidden_layers=4), SimpleHashTokenizer(2048)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds_list: List[np.ndarray] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds_list.append(probs)
    return np.vstack(preds_list)


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

    _, tokenizer = build_backbone_and_tokenizer(cfg)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    test_ds = HeadTailDataset(
        test_df[cfg.text_col].values, None, tokenizer, cfg.max_len, cfg.head_len, cfg.tail_len
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_id=pad_id),
    )

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)

    use_amp = device.type == "cuda" and cfg.amp_dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        tr_ds = HeadTailDataset(
            train_df[cfg.text_col].values[tr_idx],
            y[tr_idx],
            tokenizer,
            cfg.max_len,
            cfg.head_len,
            cfg.tail_len,
        )
        va_ds = HeadTailDataset(
            train_df[cfg.text_col].values[va_idx],
            y[va_idx],
            tokenizer,
            cfg.max_len,
            cfg.head_len,
            cfg.tail_len,
        )
        tr_loader = DataLoader(
            tr_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda b: dynamic_collate_fn(b, pad_id=pad_id),
        )
        va_loader = DataLoader(
            va_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=lambda b: dynamic_collate_fn(b, pad_id=pad_id),
        )

        backbone, _ = build_backbone_and_tokenizer(cfg)
        model = TransformerClassifier(backbone, num_classes=num_classes, cfg=cfg).to(device)

        param_groups = get_llrd_optimizer_params(
            model,
            encoder_lr=cfg.encoder_lr,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
            llrd_factor=cfg.llrd_factor,
        )
        optimizer = torch.optim.AdamW(param_groups)
        total_steps = max(1, (len(tr_loader) * cfg.epochs) // max(1, cfg.grad_accum_steps))
        scheduler = build_cosine_warmup_scheduler(
            optimizer, int(total_steps * cfg.warmup_ratio), total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and cfg.amp_dtype == "fp16"))
        awp = AWP(model, adv_lr=cfg.awp_lr, adv_eps=cfg.awp_eps) if cfg.use_awp else None
        ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        best_va_probs = None
        best_f1 = -1.0
        best_state = None

        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(tr_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)

                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(ids, mask)
                    loss = F.cross_entropy(logits, targets) / cfg.grad_accum_steps

                scaler.scale(loss).backward()

                if awp is not None and epoch >= cfg.awp_start_epoch:
                    awp.perturb()
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        adv_logits = model(ids, mask)
                        adv_loss = F.cross_entropy(adv_logits, targets) / cfg.grad_accum_steps
                    scaler.scale(adv_loss).backward()
                    awp.restore()

                if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(tr_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)

            eval_model = ema.ema if ema is not None else model
            va_probs = predict_loader(eval_model, va_loader, device)
            va_f1 = float(f1_score(y[va_idx], np.argmax(va_probs, axis=1), average="macro"))
            if va_f1 >= best_f1:
                best_f1 = va_f1
                best_va_probs = va_probs
                best_state = copy.deepcopy(eval_model.state_dict())

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
    parser = argparse.ArgumentParser(description="Task 01 Tier 2 Strong DL OOF Trainer")
    parser.add_argument("--model-name", type=str, default=CFG.model_name)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--pooling-type", type=str, default=CFG.pooling_type)
    parser.add_argument("--smoke-test", action="store_true", help="Run fast offline smoke test")
    args = parser.parse_args()

    cfg = CFG(
        model_name=args.model_name,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        pooling_type=args.pooling_type,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.epochs = 2
        cfg.batch_size = 8
        cfg.grad_accum_steps = 1
        cfg.encoder_lr = 3e-3
        cfg.head_lr = 5e-3
        cfg.max_len = 64
        cfg.head_len = 30
        cfg.tail_len = 32
        cfg.pooling_type = "weighted_layer"
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            rng = np.random.default_rng(cfg.seed)
            templates = {
                0: "fiscal central bank inflation interest rate bond market",
                1: "striker stadium tournament goal championship match",
                2: "transformer neural network tensor gradient attention",
            }
            train_df = pd.DataFrame(
                [
                    {
                        "id": f"tr_{i}",
                        "text": " ".join(rng.choice(templates[i % 3].split(), size=25, replace=True)),
                        "label": i % 3,
                        "fold": i % 3,
                    }
                    for i in range(48)
                ]
            )
            test_df = pd.DataFrame(
                [
                    {
                        "id": f"te_{i}",
                        "text": " ".join(rng.choice(templates[i % 3].split(), size=25, replace=True)),
                    }
                    for i in range(15)
                ]
            )
            print("[SMOKE TEST] Running Tier 2 Transformer OOF Trainer (LLRD + WeightedLayerPooling + AWP + EMA)...")
            oof, test_preds = run_oof_training(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            print(f"[SUCCESS] Tier 2 DL smoke test passed. OOF shape={oof.shape}, Test shape={test_preds.shape}")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_training(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
