#!/usr/bin/env python3
r"""
Task 04 — Multilabel Classification: Asymmetric Loss (ASL), ZLPR Loss & Inverse-Prior Weights
=============================================================================================
Atomic standalone PyTorch loss functions for imbalanced multi-label classification:
  1. `AsymmetricLoss` (Ridnik et al., ICCV 2021):
     Decouples positive and negative focusing exponents ($\gamma_+ = 0, \gamma_- = 4$) and
     applies probability shifting ($p_m = \max(p - m, 0)$).
  2. `ZLPRLoss` (Su et al., 2022 — "Zero-bounded Log-sum-exp & Pairwise Rank-based Loss"):
     $\log\left(1 + \sum_{i \in \Omega_{\text{neg}}} e^{s_i}\right) + \log\left(1 + \sum_{j \in \Omega_{\text{pos}}} e^{-s_j}\right)$.
  3. `compute_pos_weight`:
     Computes dampened inverse-prior positive class weights $\left(\frac{N - n_c}{n_c}\right)^\alpha$
     clamped to `[min_w, max_w]` for `BCEWithLogitsLoss(pos_weight=...)`.

Usage:
    python3 tasks/04_multilabel_classification/modules/asymmetric_and_zlpr_loss.py --smoke-test
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

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CFG:
    """Configuration for AsymmetricLoss, ZLPRLoss & pos_weight Smoke Test."""

    batch_size: int = 8
    num_labels: int = 6
    gamma_neg: float = 4.0
    gamma_pos: float = 0.0
    clip_margin: float = 0.05
    smoke_test: bool = False


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


def compute_pos_weight(
    y_train: np.ndarray | torch.Tensor,
    exponent: float = 0.5,
    min_weight: float = 1.0,
    max_weight: float = 20.0,
) -> torch.Tensor:
    """
    Computes dampened inverse-prior positive weights `((N - pos) / max(pos, 1)) ** exponent`
    for `nn.BCEWithLogitsLoss(pos_weight=...)`.
    """
    arr = (
        y_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor)
        else np.asarray(y_train, dtype=np.float64)
    )
    n_samples = float(arr.shape[0])
    pos_counts = np.maximum(arr.sum(axis=0), 1.0)
    neg_counts = np.maximum(n_samples - pos_counts, 1.0)
    raw_weights = np.power(neg_counts / pos_counts, exponent)
    clamped = np.clip(raw_weights, min_weight, max_weight)
    return torch.tensor(clamped, dtype=torch.float32)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    logits = torch.randn(cfg.batch_size, cfg.num_labels, requires_grad=True)
    targets = torch.randint(0, 2, (cfg.batch_size, cfg.num_labels)).float()

    asl = AsymmetricLoss(gamma_neg=cfg.gamma_neg, gamma_pos=cfg.gamma_pos, clip=cfg.clip_margin)
    zlpr = ZLPRLoss()
    pos_w = compute_pos_weight(targets)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    l_asl = asl(logits, targets)
    l_zlpr = zlpr(logits, targets)
    l_bce = bce(logits, targets)
    (l_asl + l_zlpr + l_bce).backward()

    assert pos_w.shape == (cfg.num_labels,)

    print("[SMOKE TEST] Task 04 AsymmetricLoss, ZLPRLoss & pos_weight (`asymmetric_and_zlpr_loss.py`)")
    print(f"  AsymmetricLoss (ASL) : {float(l_asl.item()):.4f}")
    print(f"  ZLPRLoss             : {float(l_zlpr.item()):.4f}")
    print(f"  Weighted BCE         : {float(l_bce.item()):.4f}, pos_weight={pos_w.round(decimals=2).tolist()}")
    print("[SUCCESS] All multilabel loss checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04 AsymmetricLoss & ZLPRLoss")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
