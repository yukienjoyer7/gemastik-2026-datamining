#!/usr/bin/env python3
"""
Task 01 — Text Classification: Pooling Heads & Regularization Modules
=====================================================================
Standalone reusable PyTorch components used across winning Kaggle NLP solutions:
  - `MeanPooling`: Attention-masked mean pooling over token embeddings
  - `MaxPooling`: Attention-masked max pooling over token embeddings
  - `MeanMaxPooling`: Concatenation of masked mean and max pooled representations
  - `AttentionPooling`: Learned 2-layer tanh attention pooling over tokens
  - `WeightedLayerPooling`: Learned softmax weights over the last N Transformer layers
  - `MultiSampleDropout`: Parallel dropout branches averaged across a shared classifier
  - `AWP`: Adversarial Weight Perturbation for flatter loss landscapes
  - `ModelEMA`: Exponential Moving Average of model parameters
  - `RDropClassificationLoss`: Regularized Dropout loss (Cross-Entropy + symmetric KL)

Usage:
    python modules/pooling_and_regularization.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import site
import sys
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

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    """Configuration for testing pooling & regularization modules."""

    batch_size: int = 8
    seq_len: int = 32
    hidden_size: int = 64
    num_layers: int = 4
    num_classes: int = 3
    num_dropouts: int = 5
    dropout_rate: float = 0.2
    awp_lr: float = 1e-2
    awp_eps: float = 1e-2
    ema_decay: float = 0.995
    rdrop_alpha: float = 1.0
    smoke_test: bool = False


# =============================================================================
# 1. Transformer Sequence Pooling Heads
# =============================================================================


class MeanPooling(nn.Module):
    """Attention-mask-aware mean pooling over token sequence representations."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class MaxPooling(nn.Module):
    """Attention-mask-aware max pooling over token sequence representations."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        masked_hidden = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
        return torch.max(masked_hidden, dim=1)[0]


class MeanMaxPooling(nn.Module):
    """Concatenates masked MeanPooling and MaxPooling, doubling output dimension (2 * H)."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_pool = MeanPooling()
        self.max_pool = MaxPooling()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mean_emb = self.mean_pool(last_hidden_state, attention_mask)
        max_emb = self.max_pool(last_hidden_state, attention_mask)
        return torch.cat([mean_emb, max_emb], dim=-1)


class AttentionPooling(nn.Module):
    """Learned token-level attention pooling: `softmax(v^T tanh(W h_t))`."""

    def __init__(self, hidden_size: int, bottleneck_size: Optional[int] = None) -> None:
        super().__init__()
        mid_dim = bottleneck_size or hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(last_hidden_state).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        return torch.sum(last_hidden_state * weights, dim=1)


class WeightedLayerPooling(nn.Module):
    """
    Learns scalar weights over the last `num_hidden_layers` transformer layers,
    followed by attention-masked mean pooling.
    """

    def __init__(self, num_total_layers: int = 12, layer_start: int = 8) -> None:
        super().__init__()
        self.layer_start = layer_start
        self.num_layers_to_pool = max(1, num_total_layers - layer_start + 1)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers_to_pool, dtype=torch.float32))
        self.mean_pool = MeanPooling()

    def forward(
        self, all_hidden_states: Sequence[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        selected = all_hidden_states[-self.num_layers_to_pool :]
        stacked = torch.stack(list(selected), dim=0)  # (K, B, L, H)
        norm_weights = torch.softmax(self.layer_weights[: len(selected)], dim=0).view(-1, 1, 1, 1)
        weighted_hidden = (norm_weights * stacked).sum(dim=0)  # (B, L, H)
        return self.mean_pool(weighted_hidden, attention_mask)


# =============================================================================
# 2. Multi-Sample Dropout Head
# =============================================================================


class MultiSampleDropout(nn.Module):
    """
    Applies K parallel dropout masks with different rates/draws before a shared
    linear classification head and averages the resulting logits.
    Accelerates convergence and stabilizes fine-tuning on small datasets.
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
            [nn.Dropout(dropout_rate + 0.05 * (i - num_samples // 2)).clamp_p() if False else nn.Dropout(dropout_rate) for i in range(num_samples)]
        )
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_sum = sum(self.classifier(drop(x)) for drop in self.dropouts)
        return logits_sum / len(self.dropouts)


# =============================================================================
# 3. Adversarial Weight Perturbation (AWP), Model EMA & R-Drop Loss
# =============================================================================


class AWP:
    """
    Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
    Perturbs model weights in the direction of the normalized gradient to maximize
    loss, then updates parameters from that worst-case neighborhood.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = "weight",
        adv_lr: float = 1e-2,
        adv_eps: float = 1e-2,
        start_epoch: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup: Dict[str, torch.Tensor] = {}
        self.backup_eps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def perturb(self) -> None:
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    lo, hi = self.backup_eps[name]
                    param.data = torch.min(torch.max(param.data, lo), hi)

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of the model for evaluation and inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


class RDropClassificationLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks (Liang et al., NeurIPS 2021).
    Computes average Cross-Entropy across two forward passes plus symmetric
    KL-divergence between their output distributions.
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
        kl_loss = 0.5 * (kl_pq + kl_qp)
        return ce_loss + self.alpha * kl_loss


# =============================================================================
# 4. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    B, L, H, C = cfg.batch_size, cfg.seq_len, cfg.hidden_size, cfg.num_classes
    hidden = torch.randn(B, L, H, requires_grad=True)
    all_layers = [torch.randn(B, L, H) for _ in range(cfg.num_layers)]
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, -4:] = 0
    targets = torch.randint(0, C, (B,))

    mean_out = MeanPooling()(hidden, mask)
    max_out = MaxPooling()(hidden, mask)
    mm_out = MeanMaxPooling()(hidden, mask)
    attn_out = AttentionPooling(H)(hidden, mask)
    wlayer_out = WeightedLayerPooling(num_total_layers=cfg.num_layers, layer_start=2)(all_layers, mask)

    assert mean_out.shape == (B, H)
    assert max_out.shape == (B, H)
    assert mm_out.shape == (B, 2 * H)
    assert attn_out.shape == (B, H)
    assert wlayer_out.shape == (B, H)

    msd = MultiSampleDropout(H, C, num_samples=cfg.num_dropouts, dropout_rate=cfg.dropout_rate)
    logits1 = msd(mean_out)
    logits2 = msd(mean_out)
    assert logits1.shape == (B, C)

    rdrop = RDropClassificationLoss(alpha=cfg.rdrop_alpha)
    loss = rdrop(logits1, logits2, targets)

    optimizer = torch.optim.AdamW(msd.parameters(), lr=1e-3)
    awp = AWP(msd, optimizer, adv_lr=cfg.awp_lr, adv_eps=cfg.awp_eps)
    ema = ModelEMA(msd, decay=cfg.ema_decay)

    loss.backward()
    awp.perturb()
    adv_logits = msd(mean_out.detach())
    adv_loss = F.cross_entropy(adv_logits, targets)
    optimizer.zero_grad()
    adv_loss.backward()
    awp.restore()
    optimizer.step()
    ema.update(msd)

    print("[SMOKE TEST] Task 01 Pooling & Regularization Modules")
    print(f"  MeanPooling shape          : {tuple(mean_out.shape)}")
    print(f"  MeanMaxPooling shape       : {tuple(mm_out.shape)}")
    print(f"  AttentionPooling shape     : {tuple(attn_out.shape)}")
    print(f"  WeightedLayerPooling shape : {tuple(wlayer_out.shape)}")
    print(f"  MultiSampleDropout shape   : {tuple(logits1.shape)}")
    print(f"  R-Drop Loss + AWP + EMA    : loss={float(loss.item()):.4f}, adv_loss={float(adv_loss.item()):.4f}")
    print("[SUCCESS] All pooling and regularization checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Pooling & Regularization Modules")
    parser.add_argument("--smoke-test", action="store_true", help="Run module smoke tests")
    args = parser.parse_args()
    cfg = CFG(smoke_test=args.smoke_test)
    run_smoke_test(cfg)


if __name__ == "__main__":
    main()
