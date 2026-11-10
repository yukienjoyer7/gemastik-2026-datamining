#!/usr/bin/env python3
"""
Task 12 — Module: 1D Dilated Multi-Scale ResNet + Squeeze-and-Excitation + Self-Attention
=========================================================================================
Reusable PyTorch sequence encoder (`DilatedResNet1DSequenceModel`) inspired by winning
solutions in Kaggle Leash Bio – BELKA (SMILES binding), OpenVaccine / Ribonanza RNA,
and Protein sequence benchmarks:
  1. Multi-Kernel Stem (`k=3, 5, 7, 11`): Simultaneously captures local chemical bonds
     (2–3 tokens), functional groups / codons (3–5 tokens), and aromatic rings /
     secondary structure motifs (7–11 tokens).
  2. Dilated Residual Blocks (`dilation=1, 2, 4`) with **Squeeze-and-Excitation (SE)**
     channel attention to weight motif channels globally.
  3. Lightweight Self-Attention + Masked **Mean + Max + Attention Pooling** fused with
     optional dense physicochemical features and Multi-Sample Dropout.

Usage:
    python modules/dilated_1d_resnet_encoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

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
    """Configuration for 1D Dilated Multi-Scale ResNet Sequence Encoder."""

    vocab_size: int = 128
    embed_dim: int = 64
    branch_channels: int = 32
    hidden_channels: int = 128
    dilations: Tuple[int, ...] = (1, 2, 4)
    se_reduction: int = 4
    aux_feat_dim: int = 0
    dropout: float = 0.2
    num_dropouts: int = 4
    smoke_test: bool = False


class SqueezeExcitation1D(nn.Module):
    """Channel-wise Squeeze-and-Excitation block for 1D sequence feature maps `(B, C, L)`."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, L), mask: (B, L)
        if mask is not None:
            m = mask.unsqueeze(1)  # (B, 1, L)
            pooled = (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1e-6)
        else:
            pooled = x.mean(dim=-1)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))).unsqueeze(-1)
        return x * scale


class DilatedResBlock1D(nn.Module):
    """1D Dilated Residual Block with BatchNorm/GroupNorm, SiLU, and Squeeze-and-Excitation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        se_reduction: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.se = SqueezeExcitation1D(channels, reduction=se_reduction)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.se(out, mask=mask)
        if mask is not None:
            out = out * mask.unsqueeze(1)
        return F.silu(out + residual)


class DilatedResNet1DSequenceModel(nn.Module):
    """
    Multi-Scale (`k=3, 5, 7, 11`) 1D Dilated ResNet + SE + Self-Attention +
    Mean/Max/Attention Pooling sequence model for SMILES and biological sequences.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        num_classes: int = 2,
        embed_dim: int = 64,
        branch_channels: int = 32,
        hidden_channels: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4),
        aux_feat_dim: int = 0,
        dropout: float = 0.2,
        num_dropouts: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale parallel stem (kernels 3, 5, 7, 11)
        self.stem_k3 = nn.Conv1d(embed_dim, branch_channels, kernel_size=3, padding=1)
        self.stem_k5 = nn.Conv1d(embed_dim, branch_channels, kernel_size=5, padding=2)
        self.stem_k7 = nn.Conv1d(embed_dim, branch_channels, kernel_size=7, padding=3)
        self.stem_k11 = nn.Conv1d(embed_dim, branch_channels, kernel_size=11, padding=5)

        stem_out_ch = branch_channels * 4
        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_out_ch, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
        )

        self.res_blocks = nn.ModuleList(
            [
                DilatedResBlock1D(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout * 0.5,
                )
                for d in dilations
            ]
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=4, dropout=dropout * 0.5, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_channels)

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.aux_feat_dim = aux_feat_dim
        if aux_feat_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(aux_feat_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
            )
            aux_out_dim = 64
        else:
            self.aux_mlp = None
            aux_out_dim = 0

        pooled_dim = hidden_channels * 3 + aux_out_dim
        self.head_norm = nn.LayerNorm(pooled_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_dropouts)])
        self.fc = nn.Linear(pooled_dim, num_classes)

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aux_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids: (B, L), attention_mask: (B, L) float
        mask = attention_mask.float()
        emb = self.embedding(input_ids).transpose(1, 2)  # (B, E, L)

        s3 = F.silu(self.stem_k3(emb))
        s5 = F.silu(self.stem_k5(emb))
        s7 = F.silu(self.stem_k7(emb))
        s11 = F.silu(self.stem_k11(emb))
        x = self.stem_proj(torch.cat([s3, s5, s7, s11], dim=1))
        x = x * mask.unsqueeze(1)

        for block in self.res_blocks:
            x = block(x, mask=mask)

        seq = (x * mask.unsqueeze(1)).transpose(1, 2)  # (B, L, C)
        attn_out, _ = self.self_attn(seq, seq, seq, need_weights=False)
        seq = self.attn_norm(seq + attn_out) * mask.unsqueeze(-1)

        m_exp = mask.unsqueeze(-1)  # (B, L, 1)
        mean_pool = (seq * m_exp).sum(dim=1) / m_exp.sum(dim=1).clamp(min=1e-6)
        max_pool = seq.masked_fill(m_exp == 0, -1e9).max(dim=1)[0]

        scores = self.attn_pool(seq).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        att_pool = (seq * weights).sum(dim=1)

        parts = [mean_pool, max_pool, att_pool]
        if self.aux_mlp is not None and aux_feats is not None:
            parts.append(self.aux_mlp(aux_feats))

        return self.head_norm(torch.cat(parts, dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aux_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.extract_features(input_ids, attention_mask, aux_feats)
        logits = sum(self.fc(drop(feat)) for drop in self.dropouts) / len(self.dropouts)
        return logits


def _run_smoke_test() -> None:
    torch.manual_seed(42)
    batch_size, seq_len, aux_dim, num_classes = 6, 48, 16, 3
    model = DilatedResNet1DSequenceModel(
        vocab_size=64,
        num_classes=num_classes,
        embed_dim=32,
        branch_channels=16,
        hidden_channels=64,
        dilations=(1, 2, 4),
        aux_feat_dim=aux_dim,
    )
    ids = torch.randint(1, 64, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
    mask[:, -6:] = 0.0
    aux = torch.randn(batch_size, aux_dim)

    logits = model(ids, mask, aux)
    assert logits.shape == (batch_size, num_classes)
    loss = logits.sum()
    loss.backward()
    print(f"[SMOKE TEST] DilatedResNet1DSequenceModel output shape: {tuple(logits.shape)}")
    print("[SUCCESS] dilated_1d_resnet_encoder smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="1D Dilated Multi-Scale ResNet Sequence Encoder")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
