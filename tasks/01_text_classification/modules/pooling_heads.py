#!/usr/bin/env python3
"""
Task 01 — Text Classification: Sequence Pooling Heads
=====================================================
Atomic standalone PyTorch sequence pooling heads for Transformer encoders:
  - `MeanPooling`: Attention-mask-aware mean pooling over token representations.
  - `MaxPooling`: Attention-mask-aware max pooling over token representations.
  - `MeanMaxPooling`: Concatenation of masked MeanPooling and MaxPooling (`2 * H`).
  - `AttentionPooling`: Learned 2-layer LayerNorm+GELU attention pooling over tokens.
  - `WeightedLayerPooling`: Learned softmax weights over the last `K` Transformer layers.
  - `GeMPooling`: Generalized Mean Pooling (`(1/N sum_t x_t^p)^(1/p)`) with learnable exponent `p`.

Usage:
    python3 tasks/01_text_classification/modules/pooling_heads.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

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


@dataclass
class CFG:
    """Configuration for Sequence Pooling Heads Smoke Test."""

    batch_size: int = 8
    seq_len: int = 32
    hidden_size: int = 64
    num_layers: int = 6
    gem_p: float = 3.0
    smoke_test: bool = False


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
    """Concatenates masked MeanPooling and MaxPooling, doubling output dimension (`2 * H`)."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_pool = MeanPooling()
        self.max_pool = MaxPooling()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mean_emb = self.mean_pool(last_hidden_state, attention_mask)
        max_emb = self.max_pool(last_hidden_state, attention_mask)
        return torch.cat([mean_emb, max_emb], dim=-1)


class AttentionPooling(nn.Module):
    """Learned token-level attention pooling: `softmax(v^T GELU(LN(W h_t)))`."""

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
    Learns scalar softmax weights over the last `num_layers_to_pool` Transformer layers,
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


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM) over sequence tokens with learnable exponent `p`.
    Interpolates smoothly between MeanPooling (`p=1`) and MaxPooling (`p -> inf`).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else torch.tensor([p])
        self.eps = eps

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        p_clamped = self.p.to(last_hidden_state.device).clamp(min=1.0, max=10.0)
        # Shift/clamp positive magnitude while preserving sign for signed hidden states
        pos_mag = last_hidden_state.clamp(min=self.eps).pow(p_clamped) * mask_expanded
        sum_mask = mask_expanded.sum(dim=1).clamp(min=self.eps)
        return (pos_mag.sum(dim=1) / sum_mask).pow(1.0 / p_clamped)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    B, L, H = cfg.batch_size, cfg.seq_len, cfg.hidden_size
    hidden = torch.randn(B, L, H, requires_grad=True)
    all_layers = [torch.randn(B, L, H) for _ in range(cfg.num_layers)]
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, -4:] = 0

    mean_out = MeanPooling()(hidden, mask)
    max_out = MaxPooling()(hidden, mask)
    mm_out = MeanMaxPooling()(hidden, mask)
    attn_out = AttentionPooling(H)(hidden, mask)
    wlayer_out = WeightedLayerPooling(num_total_layers=cfg.num_layers, layer_start=3)(
        all_layers, mask
    )
    gem_out = GeMPooling(p=cfg.gem_p)(hidden, mask)

    assert mean_out.shape == (B, H)
    assert max_out.shape == (B, H)
    assert mm_out.shape == (B, 2 * H)
    assert attn_out.shape == (B, H)
    assert wlayer_out.shape == (B, H)
    assert gem_out.shape == (B, H)

    loss = (
        mean_out.sum()
        + max_out.sum()
        + mm_out.sum()
        + attn_out.sum()
        + wlayer_out.sum()
        + gem_out.sum()
    )
    loss.backward()

    print("[SMOKE TEST] Task 01 Sequence Pooling Heads (`pooling_heads.py`)")
    print(f"  Mean / Max / Attention / GeM : {tuple(mean_out.shape)}")
    print(f"  MeanMaxPooling               : {tuple(mm_out.shape)}")
    print(f"  WeightedLayerPooling         : {tuple(wlayer_out.shape)}")
    print("[SUCCESS] All 6 sequence pooling heads passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Sequence Pooling Heads")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
