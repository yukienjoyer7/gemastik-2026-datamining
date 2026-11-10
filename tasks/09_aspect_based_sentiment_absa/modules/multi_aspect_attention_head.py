"""
Task 09 — PyTorch Multi-Aspect Cross-Attention Classification Head
==================================================================
Standalone PyTorch module `MultiAspectCrossAttentionHead`:
  - Given token-level encoder hidden states `H` of shape `(batch_size, seq_len, hidden_dim)`
    and an attention mask `(batch_size, seq_len)`, learns `A` aspect-specific query vectors
    that cross-attend over token states and output `(batch_size, num_aspects, num_polarities)` logits
    in a single encoder forward pass.
  - Also optionally returns `(batch_size, num_aspects, seq_len)` aspect-token attention weights
    for interpretability and rationale extraction.

Usage:
  python3 multi_aspect_attention_head.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

_venv_site = (
    Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CFG:
    exp_name: str = "multi_aspect_attention_head"
    seed: int = 42
    hidden_dim: int = 64
    num_aspects: int = 4
    num_polarities: int = 4
    num_heads: int = 4


class MultiAspectCrossAttentionHead(nn.Module):
    """
    Learns `num_aspects` query embeddings `(A, D)` that cross-attend over sequence
    hidden states `(B, L, D)` and output `(B, A, num_polarities)` logits.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_aspects: int = 4,
        num_polarities: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

        self.aspect_queries = nn.Parameter(
            torch.randn(num_aspects, hidden_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.aspect_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, num_polarities) for _ in range(num_aspects)]
        )

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        `token_states`: (B, L, D)
        `attention_mask`: (B, L) where 1 = valid token, 0 = padding
        Returns `logits` (B, A, C), and optionally `attn_weights` (B, A, L).
        """
        bsz = token_states.size(0)
        queries = self.aspect_queries.unsqueeze(0).expand(bsz, -1, -1)  # (B, A, D)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=token_states,
            value=token_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        h = self.norm1(queries + attn_out)
        h = self.norm2(h + self.ffn(h))

        logits = torch.stack(
            [
                head(h[:, a_idx, :])
                for a_idx, head in enumerate(self.aspect_classifiers)
            ],
            dim=1,
        )
        if return_attn_weights:
            return logits, attn_weights
        return logits


def _run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(cfg.seed)
    head = MultiAspectCrossAttentionHead(
        hidden_dim=cfg.hidden_dim,
        num_aspects=cfg.num_aspects,
        num_polarities=cfg.num_polarities,
        num_heads=cfg.num_heads,
    )
    token_states = torch.randn(6, 20, cfg.hidden_dim, requires_grad=True)
    mask = torch.ones((6, 20), dtype=torch.long)
    mask[:, -3:] = 0
    targets = torch.randint(0, cfg.num_polarities, (6, cfg.num_aspects))

    logits, attn_w = head(token_states, attention_mask=mask, return_attn_weights=True)
    loss = F.cross_entropy(logits.reshape(-1, cfg.num_polarities), targets.reshape(-1))
    loss.backward()

    probs = F.softmax(logits.detach(), dim=-1).numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", probs)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", probs)
        pd.DataFrame(probs.argmax(axis=-1), columns=[f"aspect_{i}" for i in range(cfg.num_aspects)]).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] MultiAspectCrossAttentionHead logits={tuple(logits.shape)}, "
        f"attn_weights={tuple(attn_w.shape)}, loss={loss.item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 Multi-Aspect Attention Head Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
