#!/usr/bin/env python3
"""
pytorch_losses.py
=================
Standalone competition PyTorch `nn.Module` custom loss functions:
  1. `FocalLoss`                  — Binary & Multiclass Focal Loss with optional label smoothing & class alpha
  2. `AsymmetricLossOptimized`    — Ridnik et al. (ICCV 2021) ASL for imbalanced multi-label classification
  3. `ZLPRLoss`                   — Su et al. (2022) Zero-bounded Log-sum-exp & Pairwise Rank-based multilabel loss
  4. `PearsonCorrelationLoss`     — 1 - Pearson r for continuous similarity / semantic textual similarity (STS)
  5. `CosineSimilarityLoss`       — 1 - Cosine similarity for embedding alignment
  6. `RDropLoss`                  — Liang et al. (NeurIPS 2021) Regularized Dropout bidirectional KL loss
  7. `BiTemperedLogisticLoss`     — Amid et al. (NeurIPS 2019) Bi-Tempered Logistic Loss for noisy labels

Usage:
  python3 00_core/metrics_and_losses/pytorch_losses.py --smoke-test
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    focal_gamma: float = 2.0
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    rdrop_alpha: float = 1.0
    bi_temp_t1: float = 0.8
    bi_temp_t2: float = 1.2
    seed: int = 42


class FocalLoss(nn.Module):
    """
    Focal Loss supporting both Binary (`mode="binary"`) and Multiclass (`mode="multiclass"`).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        mode: str = "multiclass",
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mode = mode.lower()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":
            y = targets.float().view_as(logits)
            if self.label_smoothing > 0.0:
                y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * y + (1.0 - probs) * (1.0 - y)
            focal_weight = (1.0 - p_t).pow(self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * y + (1.0 - self.alpha) * (1.0 - y)
                focal_weight = alpha_t * focal_weight
            loss = focal_weight * bce
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            targets_long = targets.long()
            ce = F.nll_loss(log_probs, targets_long, reduction="none")
            p_t = probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
            loss = (1.0 - p_t).pow(self.gamma) * ce
            if self.label_smoothing > 0.0:
                smooth_loss = -log_probs.mean(dim=-1)
                loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification with extreme positive-negative imbalance.
    Decouples positive focusing (`gamma_pos`) and negative focusing (`gamma_neg`) with hard
    probability shifting (`clip`) on easy negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
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

        if self.clip is not None and self.clip > 0.0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.sum(dim=-1).mean()


class ZLPRLoss(nn.Module):
    """
    Zero-bounded Log-sum-exp & Pairwise Rank-based Loss (Su et al., 2022) for multi-label classification:
      L_ZLPR = log(1 + sum_{i in Omega_neg} exp(s_i)) + log(1 + sum_{j in Omega_pos} exp(-s_j))
    Directly enforces positive class logits > 0 and negative class logits < 0 without threshold tuning.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.float()
        neg_mask = 1.0 - y
        pos_mask = y

        # Mask out inactive terms with -inf before logsumexp
        neg_logits = logits.masked_fill(neg_mask < 0.5, -1e12)
        pos_logits = (-logits).masked_fill(pos_mask < 0.5, -1e12)

        zeros = torch.zeros((logits.size(0), 1), dtype=logits.dtype, device=logits.device)
        neg_loss = torch.logsumexp(torch.cat([zeros, neg_logits], dim=-1), dim=-1)
        pos_loss = torch.logsumexp(torch.cat([zeros, pos_logits], dim=-1), dim=-1)
        return (neg_loss + pos_loss).mean()


class PearsonCorrelationLoss(nn.Module):
    """
    Differentiable `1 - Pearson r` loss for continuous similarity / regression rank alignment.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = preds.view(-1).float()
        y = targets.view(-1).float()
        vx = x - x.mean()
        vy = y - y.mean()
        corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()) + self.eps)
        return 1.0 - corr


class CosineSimilarityLoss(nn.Module):
    """
    Differentiable `1 - CosineSimilarity(x, y)` loss across feature dimension or batch vector.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)
        return (1.0 - sim).mean()


class RDropLoss(nn.Module):
    """
    R-Drop (Regularized Dropout) Loss combining average CrossEntropy across two stochastic
    forward passes with symmetric KL divergence:
      L = 0.5 * (CE(p1, y) + CE(p2, y)) + alpha * 0.5 * (KL(p1 || p2) + KL(p2 || p1))
    """

    def __init__(self, alpha: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @staticmethod
    def compute_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl_12 = F.kl_div(p1, p2.exp(), reduction="batchmean")
        kl_21 = F.kl_div(p2, p1.exp(), reduction="batchmean")
        return 0.5 * (kl_12 + kl_21)

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = 0.5 * (self.ce(logits1, targets.long()) + self.ce(logits2, targets.long()))
        kl_loss = self.compute_kl_loss(logits1, logits2)
        return ce_loss + self.alpha * kl_loss


class BiTemperedLogisticLoss(nn.Module):
    """
    Bi-Tempered Logistic Loss (Amid et al., NeurIPS 2019) for classification with noisy labels.
    Uses bounded `t1 in (0, 1)` for heavy-tailed robustness to outliers and `t2 > 1` for tail softness.
    """

    def __init__(
        self,
        t1: float = 0.8,
        t2: float = 1.2,
        label_smoothing: float = 0.0,
        num_iters: int = 5,
    ) -> None:
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @staticmethod
    def log_t(u: torch.Tensor, t: float) -> torch.Tensor:
        u_clamped = u.clamp(min=1e-7)
        if abs(t - 1.0) < 1e-6:
            return torch.log(u_clamped)
        return (u_clamped.pow(1.0 - t) - 1.0) / (1.0 - t)

    @staticmethod
    def exp_t(u: torch.Tensor, t: float) -> torch.Tensor:
        if abs(t - 1.0) < 1e-6:
            return torch.exp(u)
        return torch.relu(1.0 + (1.0 - t) * u).pow(1.0 / (1.0 - t))

    def _compute_normalization(self, activations: torch.Tensor) -> torch.Tensor:
        mu = torch.max(activations, dim=-1, keepdim=True).values
        normalized_activations_step_0 = activations - mu
        normalized_activations = normalized_activations_step_0
        for _ in range(self.num_iters):
            logt_partition = torch.sum(self.exp_t(normalized_activations, self.t2), dim=-1, keepdim=True)
            normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - self.t2)
        logt_partition = torch.sum(self.exp_t(normalized_activations, self.t2), dim=-1, keepdim=True)
        return -self.log_t(1.0 / logt_partition, self.t2) + mu

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if targets.dim() == 1:
            y = F.one_hot(targets.long(), num_classes=num_classes).float()
        else:
            y = targets.float()
        if self.label_smoothing > 0.0:
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing / float(num_classes)

        norm_constants = self._compute_normalization(logits)
        probabilities = self.exp_t(logits - norm_constants, self.t2)

        temp1 = (self.log_t(y, self.t1) - self.log_t(probabilities, self.t1)) * y
        temp2 = (1.0 / (2.0 - self.t1)) * (y.pow(2.0 - self.t1) - probabilities.pow(2.0 - self.t1))
        loss_values = torch.sum(temp1 - temp2, dim=-1)
        return loss_values.mean()


def run_smoke_test() -> None:
    torch.manual_seed(42)
    B, C = 16, 4

    # 1. FocalLoss (multiclass & binary)
    logits_mc = torch.randn(B, C, requires_grad=True)
    y_mc = torch.randint(0, C, (B,))
    loss_focal_mc = FocalLoss(gamma=2.0, mode="multiclass", label_smoothing=0.05)(logits_mc, y_mc)
    loss_focal_mc.backward()
    assert torch.isfinite(loss_focal_mc) and logits_mc.grad is not None

    logits_bin = torch.randn(B, requires_grad=True)
    y_bin = torch.randint(0, 2, (B,)).float()
    loss_focal_bin = FocalLoss(gamma=2.0, mode="binary")(logits_bin, y_bin)
    loss_focal_bin.backward()
    assert torch.isfinite(loss_focal_bin) and logits_bin.grad is not None

    # 2. AsymmetricLossOptimized (multilabel)
    logits_ml = torch.randn(B, C, requires_grad=True)
    y_ml = torch.randint(0, 2, (B, C)).float()
    loss_asl = AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)(logits_ml, y_ml)
    loss_asl.backward()
    assert torch.isfinite(loss_asl) and logits_ml.grad is not None

    # 3. ZLPRLoss (multilabel)
    logits_zlpr = torch.randn(B, C, requires_grad=True)
    loss_zlpr = ZLPRLoss()(logits_zlpr, y_ml)
    loss_zlpr.backward()
    assert torch.isfinite(loss_zlpr) and logits_zlpr.grad is not None

    # 4. PearsonCorrelationLoss & CosineSimilarityLoss
    pred_reg = torch.randn(B, requires_grad=True)
    true_reg = torch.randn(B)
    loss_pearson = PearsonCorrelationLoss()(pred_reg, true_reg)
    loss_pearson.backward()
    assert torch.isfinite(loss_pearson) and pred_reg.grad is not None

    emb1 = torch.randn(B, 12, requires_grad=True)
    emb2 = torch.randn(B, 12)
    loss_cos = CosineSimilarityLoss()(emb1, emb2)
    loss_cos.backward()
    assert torch.isfinite(loss_cos) and emb1.grad is not None

    # 5. RDropLoss
    l1 = torch.randn(B, C, requires_grad=True)
    l2 = torch.randn(B, C, requires_grad=True)
    loss_rdrop = RDropLoss(alpha=1.5)(l1, l2, y_mc)
    loss_rdrop.backward()
    assert torch.isfinite(loss_rdrop) and l1.grad is not None and l2.grad is not None

    # 6. BiTemperedLogisticLoss
    l_bt = torch.randn(B, C, requires_grad=True)
    loss_bt = BiTemperedLogisticLoss(t1=0.8, t2=1.2, label_smoothing=0.05)(l_bt, y_mc)
    loss_bt.backward()
    assert torch.isfinite(loss_bt) and l_bt.grad is not None

    print(
        f"[PYTORCH LOSSES] FocalMC={loss_focal_mc.item():.4f} | FocalBin={loss_focal_bin.item():.4f} | "
        f"ASL={loss_asl.item():.4f} | ZLPR={loss_zlpr.item():.4f} | Pearson={loss_pearson.item():.4f} | "
        f"Cosine={loss_cos.item():.4f} | RDrop={loss_rdrop.item():.4f} | BiTempered={loss_bt.item():.4f}"
    )
    print("[SMOKE TEST PASSED] pytorch_losses.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone competition PyTorch custom losses.")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke_test()
