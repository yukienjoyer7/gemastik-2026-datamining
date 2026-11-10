#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Label-Wise Attention Head
==============================================================
Atomic standalone PyTorch module (`LabelWiseAttentionHead`) where each of the $L$
labels possesses a dedicated query vector $q_l \in \mathbb{R}^D$ that attends
independently over sequence token representations $H \in \mathbb{R}^{B \times T \times D}$
before label-specific projection.

Usage:
    python3 tasks/04_multilabel_classification/modules/label_wise_attention_head.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Tuple

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
    """Configuration for LabelWiseAttentionHead Smoke Test."""

    batch_size: int = 8
    seq_len: int = 24
    hidden_size: int = 64
    num_labels: int = 6
    smoke_test: bool = False


class LabelWiseAttentionHead(nn.Module):
    r"""
    Computes label-specific attention weights over sequence tokens $H \in \mathbb{R}^{B \times T \times D}$:
      - $A = \text{softmax}_T(Q K^T / \sqrt{D}) \in \mathbb{R}^{B \times L \times T}$
      - $V = A H \in \mathbb{R}^{B \times L \times D}$
      - $\text{logits}_l = w_l^T V_l + b_l$
    """

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.scale = hidden_size**-0.5

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.out_weight = nn.Parameter(torch.randn(num_labels, hidden_size) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(num_labels))

    def forward_with_attention(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.proj(last_hidden_state)  # (B, T, D)
        scores = torch.matmul(self.label_queries.unsqueeze(0), keys.transpose(1, 2)) * self.scale
        mask = attention_mask.unsqueeze(1) == 0  # (B, 1, T)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, T)

        label_contexts = torch.matmul(attn_weights, last_hidden_state)  # (B, L, D)
        label_contexts = self.dropout(label_contexts)
        logits = (label_contexts * self.out_weight.unsqueeze(0)).sum(dim=-1) + self.out_bias
        return logits, attn_weights

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.forward_with_attention(last_hidden_state, attention_mask)
        return logits


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    B, T, D, L = cfg.batch_size, cfg.seq_len, cfg.hidden_size, cfg.num_labels
    hidden = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[:, -3:] = 0

    head = LabelWiseAttentionHead(hidden_size=D, num_labels=L)
    logits, attn_weights = head.forward_with_attention(hidden, mask)

    assert logits.shape == (B, L)
    assert attn_weights.shape == (B, L, T)
    assert torch.allclose(attn_weights[:, :, -3:], torch.zeros_like(attn_weights[:, :, -3:]), atol=1e-6)
    logits.sum().backward()

    print("[SMOKE TEST] Task 04 LabelWiseAttentionHead (`label_wise_attention_head.py`)")
    print(f"  Logits shape            : {tuple(logits.shape)}")
    print(f"  Attention weights shape : {tuple(attn_weights.shape)}")
    print("[SUCCESS] LabelWiseAttentionHead checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 LabelWiseAttentionHead")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
