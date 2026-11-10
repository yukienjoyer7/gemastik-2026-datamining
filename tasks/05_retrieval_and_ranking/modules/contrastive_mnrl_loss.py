"""
Task 05 — Contrastive Multiple Negatives Ranking Loss (MNRL / InfoNCE)
======================================================================
Standalone PyTorch module implementing `MultipleNegativesRankingLoss` for Bi-Encoder
contrastive retrieval training:
  - Supports in-batch negatives `(batch_size, dim)`
  - Supports explicit shared `(M, dim)` or per-query `(batch_size, K, dim)` hard negatives
  - Temperature scaling `tau` and optional symmetric bidirectional `(q->d + d->q)` loss

Usage:
  python3 contrastive_mnrl_loss.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    exp_name: str = "contrastive_mnrl_loss"
    seed: int = 42
    temperature: float = 0.05
    symmetric: bool = True


class MultipleNegativesRankingLoss(nn.Module):
    """
    InfoNCE / Multiple Negatives Ranking Loss for Bi-Encoder contrastive training.
    Supports both in-batch negatives `(batch_size, dim)` and explicit hard negatives
    `(batch_size, n_hard_negs, dim)` or `(n_shared_negs, dim)`.
    """

    def __init__(self, temperature: float = 0.05, symmetric: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        query_embs: torch.Tensor,
        pos_doc_embs: torch.Tensor,
        hard_neg_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = F.normalize(query_embs, p=2, dim=-1)
        p = F.normalize(pos_doc_embs, p=2, dim=-1)

        # In-batch similarity matrix: (B, B)
        logits = torch.matmul(q, p.transpose(0, 1)) / self.temperature

        if hard_neg_embs is not None:
            n = F.normalize(hard_neg_embs, p=2, dim=-1)
            if n.dim() == 2:
                # Shared pool of hard negatives: (M, D) -> (B, M)
                neg_logits = torch.matmul(q, n.transpose(0, 1)) / self.temperature
            else:
                # Per-query hard negatives: (B, K, D) -> (B, K)
                neg_logits = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([logits, neg_logits], dim=1)

        labels = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        loss_q2d = F.cross_entropy(logits, labels)

        if self.symmetric and hard_neg_embs is None:
            loss_d2q = F.cross_entropy(logits.transpose(0, 1), labels)
            return 0.5 * (loss_q2d + loss_d2q)
        return loss_q2d


def _run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(cfg.seed)
    loss_fn = MultipleNegativesRankingLoss(
        temperature=cfg.temperature, symmetric=cfg.symmetric
    )
    q_t = torch.randn(8, 16, requires_grad=True)
    p_t = q_t.detach() + 0.1 * torch.randn(8, 16)
    n_t = torch.randn(8, 3, 16)

    loss_inbatch = loss_fn(q_t, p_t)
    loss_hard = loss_fn(q_t, p_t, n_t)
    loss_hard.backward()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        oof_arr = np.array([loss_inbatch.item(), loss_hard.item()], dtype=np.float32)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", oof_arr)
        pd.DataFrame({"metric": ["inbatch", "hard_neg"], "loss": oof_arr}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] MNRL loss(in-batch)={loss_inbatch.item():.4f}, "
        f"loss(hard-neg)={loss_hard.item():.4f}, grad_norm={q_t.grad.norm().item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Contrastive MNRL Loss Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
