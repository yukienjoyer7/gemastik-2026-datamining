#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Ranking, Correlation & Contrastive Losses
=========================================================================
Atomic standalone PyTorch loss functions for semantic similarity and pair ranking:
  1. `CoSENTLoss`: Cosine Sentence Ranking Loss (Su, 2022) directly optimizing pairwise
     rank order on cosine similarities.
  2. `PearsonCorrelationLoss`: Differentiable `1 - PearsonR(preds, targets)` loss (1st-place
     U.S. Patent Phrase to Phrase Matching standard).
  3. `SpearmanSurrogateLoss`: Differentiable sigmoid-rank surrogate loss approximating
     Spearman rank correlation.
  4. `ContrastiveMarginLoss`: Online contrastive cosine margin loss for binary pairs.

Usage:
    python3 tasks/02_text_pair_similarity/modules/similarity_losses.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass

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
    """Configuration for Similarity Losses Smoke Test."""

    batch_size: int = 16
    embed_dim: int = 64
    cosent_scale: float = 20.0
    margin: float = 0.5
    smoke_test: bool = False


class CoSENTLoss(nn.Module):
    """
    CoSENT (Cosine Sentence) Ranking Loss:
      `log(1 + sum_{y_i > y_j} exp(scale * (cos(u_j, v_j) - cos(u_i, v_i))))`
    """

    def __init__(self, scale: float = 20.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self, u: torch.Tensor, v: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        u_norm = F.normalize(u, p=2, dim=-1)
        v_norm = F.normalize(v, p=2, dim=-1)
        cos_sim = (u_norm * v_norm).sum(dim=-1) * self.scale

        y = targets.view(-1).float()
        cos_diff = cos_sim.unsqueeze(0) - cos_sim.unsqueeze(1)
        order_mask = (y.unsqueeze(1) - y.unsqueeze(0)) > 1e-6

        if not order_mask.any():
            return cos_sim.sum() * 0.0

        valid_diffs = cos_diff[order_mask]
        zero_anchor = torch.zeros(1, device=valid_diffs.device, dtype=valid_diffs.dtype)
        return torch.logsumexp(torch.cat([zero_anchor, valid_diffs], dim=0), dim=0)


class PearsonCorrelationLoss(nn.Module):
    """Differentiable `1 - PearsonR(preds, targets)` loss."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = preds.view(-1).float()
        y = targets.view(-1).float()
        vx = x - x.mean()
        vy = y - y.mean()
        denom = torch.sqrt((vx.pow(2).sum() + self.eps) * (vy.pow(2).sum() + self.eps))
        corr = (vx * vy).sum() / denom
        return 1.0 - corr.clamp(-1.0, 1.0)


class SpearmanSurrogateLoss(nn.Module):
    """
    Differentiable surrogate for `1 - SpearmanRho(preds, targets)`.
    Approximates soft ranks via pairwise sigmoid comparisons `sum_j sigmoid((x_i - x_j) / temp)`
    and computes Pearson correlation loss between soft predicted ranks and soft target ranks.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.pearson = PearsonCorrelationLoss()

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(-1, 1).float()
        diff = (v - v.t()) / max(self.temperature, 1e-4)
        return torch.sigmoid(diff).sum(dim=-1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r_pred = self._soft_rank(preds)
        r_true = self._soft_rank(targets.detach())
        return self.pearson(r_pred, r_true)


class ContrastiveMarginLoss(nn.Module):
    """
    Cosine Contrastive Margin Loss for binary pair labels `y in {0, 1}`:
      `0.5 * (y * (1 - cos)^2 + (1 - y) * max(0, cos - margin)^2)`
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, u: torch.Tensor, v: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(u, v, dim=-1)
        y = (targets.view(-1).float() > 0.5).float()
        pos_loss = y * (1.0 - cos_sim).pow(2)
        neg_loss = (1.0 - y) * F.relu(cos_sim - self.margin).pow(2)
        return 0.5 * (pos_loss + neg_loss).mean()


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    u = torch.randn(cfg.batch_size, cfg.embed_dim, requires_grad=True)
    v = torch.randn(cfg.batch_size, cfg.embed_dim, requires_grad=True)
    preds = F.cosine_similarity(u, v, dim=-1)
    y = torch.rand(cfg.batch_size)

    l_cosent = CoSENTLoss(scale=cfg.cosent_scale)(u, v, y)
    l_pearson = PearsonCorrelationLoss()(preds, y)
    l_spearman = SpearmanSurrogateLoss()(preds, y)
    l_contrastive = ContrastiveMarginLoss(margin=cfg.margin)(u, v, (y > 0.5).float())

    total = l_cosent + l_pearson + l_spearman + l_contrastive
    total.backward()

    print("[SMOKE TEST] Task 02 Similarity Losses (`similarity_losses.py`)")
    print(f"  CoSENTLoss            : {float(l_cosent.item()):.4f}")
    print(f"  PearsonCorrelationLoss: {float(l_pearson.item()):.4f}")
    print(f"  SpearmanSurrogateLoss : {float(l_spearman.item()):.4f}")
    print(f"  ContrastiveMarginLoss : {float(l_contrastive.item()):.4f}")
    print("[SUCCESS] All 4 similarity & ranking losses passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Similarity Losses")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
