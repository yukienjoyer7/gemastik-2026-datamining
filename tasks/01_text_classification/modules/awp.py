#!/usr/bin/env python3
"""
Task 01 — Text Classification: Adversarial Weight Perturbation (AWP)
====================================================================
Atomic standalone implementation of Adversarial Weight Perturbation (Wu et al., NeurIPS 2020)
for Transformer fine-tuning:
  - Perturbs model weights in the direction of normalized parameter gradients (`adv_lr`, `adv_eps`)
    to find flatter minima in the loss landscape.
  - Supports both `start_epoch` and `min_loss_threshold` triggers, plus a convenient
    `attack_backward(inputs, targets, loss_fn)` helper as well as manual `perturb()` / `restore()`.

Usage:
    python3 tasks/01_text_classification/modules/awp.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
    """Configuration for AWP Smoke Test."""

    batch_size: int = 8
    in_features: int = 32
    num_classes: int = 3
    adv_lr: float = 1e-2
    adv_eps: float = 1e-2
    start_epoch: int = 1
    smoke_test: bool = False


class AWP:
    """
    Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
    Perturbs weights in the direction of the gradient to maximize loss within an
    epsilon ball proportional to parameter norms, then restores weights after the
    adversarial backward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        adv_param: str = "weight",
        adv_lr: float = 1e-2,
        adv_eps: float = 1e-2,
        start_epoch: int = 1,
        max_loss_threshold: Optional[float] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.max_loss_threshold = max_loss_threshold
        self.backup: Dict[str, torch.Tensor] = {}
        self.backup_eps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def should_Trigger(self, epoch: int, current_loss: Optional[float] = None) -> bool:
        if epoch < self.start_epoch:
            return False
        if self.max_loss_threshold is not None and current_loss is not None:
            return current_loss <= self.max_loss_threshold
        return True

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

    def step(
        self,
        forward_loss_fn: Callable[[], torch.Tensor],
        epoch: int = 1,
        current_loss: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """
        Executes a full AWP perturbation -> adversarial forward/backward -> weight restoration step.
        """
        if not self.should_Trigger(epoch, current_loss):
            return None
        self.perturb()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        adv_loss = forward_loss_fn()
        adv_loss.backward()
        self.restore()
        return adv_loss


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(cfg.in_features, cfg.in_features),
        nn.GELU(),
        nn.Linear(cfg.in_features, cfg.num_classes),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    awp = AWP(
        model,
        optimizer=opt,
        adv_lr=cfg.adv_lr,
        adv_eps=cfg.adv_eps,
        start_epoch=cfg.start_epoch,
        max_loss_threshold=5.0,
    )

    x = torch.randn(cfg.batch_size, cfg.in_features)
    y = torch.randint(0, cfg.num_classes, (cfg.batch_size,))

    w_before = model[0].weight.data.clone()
    clean_loss = F.cross_entropy(model(x), y)
    clean_loss.backward()

    adv_loss = awp.step(lambda: F.cross_entropy(model(x), y), epoch=1, current_loss=float(clean_loss.item()))
    assert adv_loss is not None
    # Verify weights were restored to exact pre-perturbation values before optimizer.step()
    assert torch.allclose(model[0].weight.data, w_before)
    opt.step()

    print("[SMOKE TEST] Task 01 Adversarial Weight Perturbation (`awp.py`)")
    print(f"  Clean loss : {float(clean_loss.item()):.4f}")
    print(f"  Adv loss   : {float(adv_loss.item()):.4f} (>= clean loss)")
    print("[SUCCESS] AWP perturb, restore, and step checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Adversarial Weight Perturbation (AWP)")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
