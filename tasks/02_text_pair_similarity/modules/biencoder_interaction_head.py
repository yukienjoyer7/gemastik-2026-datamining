#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Bi-Encoder Interaction Head
===========================================================
Atomic standalone Sentence-BERT / InferSent interaction layer (`BiEncoderInteractionHead`)
that combines dense embeddings `u` and `v` via:
  `[u, v, |u - v|, u * v, cos(u, v)]`
followed by LayerNorm, Dropout, and a 2-layer GELU MLP classifier/regressor.

Usage:
    python3 tasks/02_text_pair_similarity/modules/biencoder_interaction_head.py --smoke-test
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
    """Configuration for BiEncoderInteractionHead Smoke Test."""

    batch_size: int = 16
    embed_dim: int = 64
    hidden_dim: int = 64
    out_dim: int = 1
    smoke_test: bool = False


class BiEncoderInteractionHead(nn.Module):
    """
    Combines two dense sentence embeddings `u` and `v` via:
      `[u, v, |u - v|, u * v, cos(u, v)]` (dimension `4 * D + 1`)
    followed by LayerNorm, Dropout, and a 2-layer GELU MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.2,
        include_cosine: bool = True,
    ) -> None:
        super().__init__()
        self.include_cosine = include_cosine
        in_dim = embed_dim * 4 + (1 if include_cosine else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(u - v)
        prod = u * v
        if self.include_cosine:
            cos_sim = F.cosine_similarity(u, v, dim=-1).unsqueeze(-1)
            features = torch.cat([u, v, diff, prod, cos_sim], dim=-1)
        else:
            features = torch.cat([u, v, diff, prod], dim=-1)
        return self.net(features)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    u = torch.randn(cfg.batch_size, cfg.embed_dim, requires_grad=True)
    v = torch.randn(cfg.batch_size, cfg.embed_dim, requires_grad=True)
    head = BiEncoderInteractionHead(
        embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim
    )
    out = head(u, v)
    assert out.shape == (cfg.batch_size, cfg.out_dim)
    out.sum().backward()

    print("[SMOKE TEST] Task 02 BiEncoderInteractionHead (`biencoder_interaction_head.py`)")
    print(f"  Input shapes  : u={tuple(u.shape)}, v={tuple(v.shape)}")
    print(f"  Output shape  : {tuple(out.shape)}")
    print("[SUCCESS] BiEncoderInteractionHead checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 BiEncoderInteractionHead")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
