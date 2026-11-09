#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Asymmetric Loss, ZLPR, Label-Wise Attention & Hierarchy
============================================================================================
Standalone reusable PyTorch modules and post-processing constraints for multilabel NLP:
  1. `AsymmetricLoss` (ASL — Ridnik et al., ICCV 2021):
     Decouples positive and negative focusing exponents ($\gamma_+ = 0, \gamma_- = 4$) and
     applies probability shifting ($p_m = \max(p - m, 0)$) to discard easy negative
     gradients under severe positive-negative imbalance.
  2. `ZLPRLoss` (Su et al., 2022 — "Zero-bounded Log-sum-exp & Pairwise Rank-based Loss"):
     $\mathcal{L}_{\text{ZLPR}} = \log\left(1 + \sum_{i \in \Omega_{\text{neg}}} e^{s_i}\right) + \log\left(1 + \sum_{j \in \Omega_{\text{pos}}} e^{-s_j}\right)$
     Naturally incorporates intra-sample label correlations and uses $0$ as a calibrated
     logit threshold between positive and negative labels.
  3. `LabelWiseAttentionHead` (CAML / LSAN):
     Each of the $L$ labels possesses a dedicated query vector $q_l \in \mathbb{R}^D$ that
     attends over token hidden states $H \in \mathbb{R}^{B \times T \times D}$ to extract
     label-specific representations before classification.
  4. `HierarchicalConstraintEnforcer`:
     Enforces directed implication rules ($\text{child} \implies \text{parent}$, i.e.,
     $P(\text{parent}) \ge P(\text{child})$) on predicted probability matrices.

Usage:
    python modules/multilabel_losses_and_heads.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    """Configuration for Multilabel Losses & Heads Smoke Test."""

    batch_size: int = 8
    seq_len: int = 24
    hidden_size: int = 64
    num_labels: int = 6
    gamma_neg: float = 4.0
    gamma_pos: float = 0.0
    clip_margin: float = 0.05
    smoke_test: bool = False


# =============================================================================
# 1. Asymmetric Loss (ASL) & ZLPR Loss
# =============================================================================


class AsymmetricLoss(nn.Module):
    r"""
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., ICCV 2021).
      - Positive terms: $(1 - p)^{\gamma_+} \log(p)$
      - Negative terms: $p_m^{\gamma_-} \log(1 - p_m)$ where $p_m = \max(p - m, 0)$
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class ZLPRLoss(nn.Module):
    r"""
    Zero-bounded Log-sum-exp & Pairwise Rank-based Loss (Su et al., 2022):
      $\log\left(1 + \sum_{i \in \Omega_{\text{neg}}} e^{s_i}\right) + \log\left(1 + \sum_{j \in \Omega_{\text{pos}}} e^{-s_j}\right)$
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        neg_logits = logits.masked_fill(y > 0.5, -1e9)
        pos_logits = (-logits).masked_fill(y <= 0.5, -1e9)

        zeros = torch.zeros((logits.size(0), 1), device=logits.device, dtype=logits.dtype)
        neg_loss = torch.logsumexp(torch.cat([zeros, neg_logits], dim=1), dim=1)
        pos_loss = torch.logsumexp(torch.cat([zeros, pos_logits], dim=1), dim=1)
        return (neg_loss + pos_loss).mean()


# =============================================================================
# 2. Label-Wise Attention Head
# =============================================================================


class LabelWiseAttentionHead(nn.Module):
    r"""
    Computes label-specific attention weights over sequence tokens $H \in \mathbb{R}^{B \times T \times D}$:
      - $A = \text{softmax}_T(Q H^T / \sqrt{D}) \in \mathbb{R}^{B \times L \times T}$
      - $V = A H \in \mathbb{R}^{B \times L \times D}$
      - $\text{logits}_l = w_l^T V_l + b_l$
    Each label $l \in \{1, \dots, L\}$ focuses on the exact phrase/clause relevant to that label.
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

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # last_hidden_state: (B, T, D), attention_mask: (B, T)
        keys = self.proj(last_hidden_state)  # (B, T, D)
        # scores: (B, L, T)
        scores = torch.matmul(self.label_queries.unsqueeze(0), keys.transpose(1, 2)) * self.scale
        mask = attention_mask.unsqueeze(1) == 0  # (B, 1, T)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, T)

        label_contexts = torch.matmul(attn_weights, last_hidden_state)  # (B, L, D)
        label_contexts = self.dropout(label_contexts)
        logits = (label_contexts * self.out_weight.unsqueeze(0)).sum(dim=-1) + self.out_bias
        return logits


# =============================================================================
# 3. Hierarchical Constraint Enforcer
# =============================================================================


class HierarchicalConstraintEnforcer:
    r"""
    Enforces parent-child probability consistency for a list of `(child_idx, parent_idx)`
    implication pairs ($\text{child} \implies \text{parent}$, so $P(\text{parent}) \ge P(\text{child})$):
      - `mode="bottom_up"`: $P(\text{parent}) \leftarrow \max(P(\text{parent}), P(\text{child}))$
      - `mode="top_down"`: $P(\text{child}) \leftarrow \min(P(\text{child}), P(\text{parent}))$
      - `mode="average"`: $0.5 \cdot \text{bottom\_up} + 0.5 \cdot \text{top\_down}$
    """

    def __init__(
        self, edges: Sequence[Tuple[int, int]], mode: str = "bottom_up"
    ) -> None:
        self.edges = list(edges)  # list of (child_idx, parent_idx)
        self.mode = mode

    def transform(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float64).copy()

        def _bottom_up(p: np.ndarray) -> np.ndarray:
            out = p.copy()
            for _ in range(max(1, len(self.edges))):
                for child_idx, parent_idx in self.edges:
                    out[:, parent_idx] = np.maximum(out[:, parent_idx], out[:, child_idx])
            return out

        def _top_down(p: np.ndarray) -> np.ndarray:
            out = p.copy()
            for _ in range(max(1, len(self.edges))):
                for child_idx, parent_idx in self.edges:
                    out[:, child_idx] = np.minimum(out[:, child_idx], out[:, parent_idx])
            return out

        if self.mode == "bottom_up":
            return _bottom_up(arr)
        if self.mode == "top_down":
            return _top_down(arr)
        return 0.5 * _bottom_up(arr) + 0.5 * _top_down(arr)


# =============================================================================
# 4. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    B, T, D, L = cfg.batch_size, cfg.seq_len, cfg.hidden_size, cfg.num_labels
    hidden = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[:, -3:] = 0
    targets = torch.randint(0, 2, (B, L)).float()

    head = LabelWiseAttentionHead(hidden_size=D, num_labels=L)
    logits = head(hidden, mask)
    assert logits.shape == (B, L)

    asl = AsymmetricLoss(gamma_neg=cfg.gamma_neg, gamma_pos=cfg.gamma_pos, clip=cfg.clip_margin)
    zlpr = ZLPRLoss()

    loss_asl = asl(logits, targets)
    loss_zlpr = zlpr(logits, targets)
    (loss_asl + loss_zlpr).backward()

    # Test HierarchicalConstraintEnforcer: child=1 => parent=0
    enforcer = HierarchicalConstraintEnforcer(edges=[(1, 0), (3, 2)], mode="bottom_up")
    raw_probs = torch.sigmoid(logits).detach().numpy()
    raw_probs[:, 1] = 0.90
    raw_probs[:, 0] = 0.30
    enforced_probs = enforcer.transform(raw_probs)
    assert np.all(enforced_probs[:, 0] >= enforced_probs[:, 1] - 1e-7)

    print("[SMOKE TEST] Task 04 Multilabel Losses & Heads")
    print(f"  LabelWiseAttentionHead shape : {tuple(logits.shape)}")
    print(f"  AsymmetricLoss (ASL)         : {float(loss_asl.item()):.4f}")
    print(f"  ZLPRLoss                     : {float(loss_zlpr.item()):.4f}")
    print(
        f"  HierarchicalConstraint check : P(parent)={enforced_probs[0, 0]:.2f} >= P(child)={enforced_probs[0, 1]:.2f}"
    )
    print("[SUCCESS] All Task 04 module checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 Multilabel Losses & Heads")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
