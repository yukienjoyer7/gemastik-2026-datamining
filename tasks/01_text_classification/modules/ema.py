#!/usr/bin/env python3
"""
Task 01 — Text Classification: Exponential Moving Average (`ModelEMA`)
======================================================================
Atomic standalone implementation of Exponential Moving Average (`ModelEMA`) over
PyTorch model weights with warmup-aware decay scheduling.

Usage:
    python3 tasks/01_text_classification/modules/ema.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
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


@dataclass
class CFG:
    """Configuration for ModelEMA Smoke Test."""

    in_features: int = 16
    out_features: int = 4
    ema_decay: float = 0.995
    smoke_test: bool = False


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Maintains a frozen shadow copy (`self.ema`) updated after each optimizer step
    for higher-stability validation and test inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, use_warmup: bool = True) -> None:
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.use_warmup = use_warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _effective_decay(self) -> float:
        if not self.use_warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._effective_decay()
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)

    @torch.no_grad()
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.ema(*args, **kwargs)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    model = nn.Linear(cfg.in_features, cfg.out_features)
    ema = ModelEMA(model, decay=cfg.ema_decay, use_warmup=True)

    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)

    x = torch.randn(4, cfg.in_features)
    out = ema.forward(x)
    assert out.shape == (4, cfg.out_features)
    assert not torch.allclose(ema.ema.weight, model.weight)

    print("[SMOKE TEST] Task 01 ModelEMA (`ema.py`)")
    print(f"  Updates performed : {ema.num_updates}, effective_decay={ema._effective_decay():.4f}")
    print(f"  EMA output shape  : {tuple(out.shape)}")
    print("[SUCCESS] ModelEMA checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 ModelEMA")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
