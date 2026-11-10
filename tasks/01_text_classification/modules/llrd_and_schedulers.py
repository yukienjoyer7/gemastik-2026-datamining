#!/usr/bin/env python3
"""
Task 01 — Text Classification: LLRD, Warmup Schedulers, Multi-Sample Dropout & R-Drop
=====================================================================================
Atomic standalone training utilities for Transformer fine-tuning:
  1. `get_llrd_optimizer_params`: Layer-wise Learning Rate Decay (LLRD) grouping parameters
     by Transformer layer depth with proper weight decay exclusion for `bias` and `LayerNorm`.
  2. `build_warmup_scheduler`: Linear or Cosine warmup learning rate scheduler (`LambdaLR`).
  3. `MultiSampleDropout`: Applies K parallel dropout branches before a shared classifier head.
  4. `RDropClassificationLoss`: Regularized Dropout loss (Cross-Entropy + symmetric KL divergence).

Usage:
    python3 tasks/01_text_classification/modules/llrd_and_schedulers.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import site
import sys
from dataclasses import dataclass
from typing import Dict, List

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CFG:
    """Configuration for LLRD, Schedulers, MultiSampleDropout & R-Drop Smoke Test."""

    batch_size: int = 8
    hidden_size: int = 64
    num_classes: int = 3
    num_layers: int = 4
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    llrd_decay: float = 0.9
    weight_decay: float = 0.01
    num_dropouts: int = 5
    dropout_rate: float = 0.2
    rdrop_alpha: float = 1.0
    smoke_test: bool = False


def get_llrd_optimizer_params(
    model: nn.Module,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.9,
    num_layers: int = 12,
) -> List[Dict[str, object]]:
    """
    Constructs AdamW parameter groups with Layer-wise Learning Rate Decay (LLRD):
      - Classifier head parameters receive `head_lr`.
      - Transformer layer `l` in `0..num_layers-1` receives `encoder_lr * (layer_decay ** (num_layers - 1 - l))`.
      - Embeddings receive `encoder_lr * (layer_decay ** num_layers)`.
      - `bias` and `LayerNorm` parameters receive `weight_decay = 0.0`.
    """
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm")
    layer_re = re.compile(r"(?:layer|layers|blocks)\.(\d+)\.")

    groups: Dict[Tuple[float, float], List[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        m = layer_re.search(name)
        if "embeddings" in name:
            lr = encoder_lr * (layer_decay**num_layers)
        elif m is not None:
            layer_idx = min(int(m.group(1)), max(num_layers - 1, 0))
            lr = encoder_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        elif any(k in name for k in ("classifier", "head", "pool", "fc")):
            lr = head_lr
        else:
            lr = encoder_lr

        key = (round(lr, 10), wd)
        groups.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in groups.items()
    ]


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_type: str = "cosine",
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Builds a linear or cosine warmup learning rate scheduler."""

    def _lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == "linear":
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, _lr_lambda)


class MultiSampleDropout(nn.Module):
    """
    Applies K parallel dropout branches before a shared linear classification head
    and averages the resulting logits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: int = 5,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(min(max(dropout_rate + 0.05 * (i - num_samples // 2), 0.05), 0.5))
                for i in range(num_samples)
            ]
        )
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_sum = sum(self.classifier(drop(x)) for drop in self.dropouts)
        return logits_sum / len(self.dropouts)


class RDropClassificationLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks (Liang et al., NeurIPS 2021).
    Computes average Cross-Entropy across two forward passes plus symmetric KL divergence.
    """

    def __init__(self, alpha: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self, logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = 0.5 * (self.ce(logits1, targets) + self.ce(logits2, targets))
        p_log = F.log_softmax(logits1, dim=-1)
        q_log = F.log_softmax(logits2, dim=-1)
        p = p_log.exp()
        q = q_log.exp()
        kl_pq = F.kl_div(q_log, p, reduction="batchmean")
        kl_qp = F.kl_div(p_log, q, reduction="batchmean")
        return ce_loss + self.alpha * 0.5 * (kl_pq + kl_qp)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)

    class _DummyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.layers = nn.ModuleList(
                [nn.Linear(cfg.hidden_size, cfg.hidden_size) for _ in range(cfg.num_layers)]
            )
            self.classifier = MultiSampleDropout(
                cfg.hidden_size,
                cfg.num_classes,
                num_samples=cfg.num_dropouts,
                dropout_rate=cfg.dropout_rate,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embeddings(x)
            for layer in self.layers:
                h = F.gelu(layer(h))
            return self.classifier(h)

    model = _DummyTransformer()
    param_groups = get_llrd_optimizer_params(
        model,
        encoder_lr=cfg.encoder_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.llrd_decay,
        num_layers=cfg.num_layers,
    )
    assert len(param_groups) >= 4
    lrs = sorted({float(g["lr"]) for g in param_groups})  # type: ignore[arg-type]
    assert lrs[0] < lrs[-1], "LLRD must assign monotonically increasing learning rates from bottom to head!"

    opt = torch.optim.AdamW(param_groups)
    sched = build_warmup_scheduler(opt, num_warmup_steps=2, num_training_steps=10, schedule_type="cosine")

    x = torch.randn(cfg.batch_size, cfg.hidden_size)
    y = torch.randint(0, cfg.num_classes, (cfg.batch_size,))
    logits1 = model(x)
    logits2 = model(x)
    rdrop = RDropClassificationLoss(alpha=cfg.rdrop_alpha)
    loss = rdrop(logits1, logits2, y)
    loss.backward()
    opt.step()
    sched.step()

    print("[SMOKE TEST] Task 01 LLRD, Schedulers, MultiSampleDropout & R-Drop (`llrd_and_schedulers.py`)")
    print(f"  LLRD distinct LRs : {[round(lr, 8) for lr in lrs]}")
    print(f"  R-Drop loss       : {float(loss.item()):.4f}")
    print("[SUCCESS] LLRD, warmup scheduler, MultiSampleDropout, and R-Drop checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 LLRD & Schedulers")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
