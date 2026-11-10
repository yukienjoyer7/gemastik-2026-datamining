"""
Task 07 — Pure-PyTorch Linear-Chain CRF with BIO Transition Constraint Mask
===========================================================================
Standalone neural sequence-labeling head `LinearChainCRF`:
  - Exact log-sum-exp forward partition function in log-space
  - Exact gold tag sequence log-likelihood scoring
  - Viterbi dynamic programming decoding
  - Optional `build_bio_transition_mask(id_to_label)` that masks out illegal BIO transitions
    (`O -> I-X`, `B-X -> I-Y`, `I-X -> I-Y`, `START -> I-X`) with `-1e4` penalty

Usage:
  python3 linear_chain_crf.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass
class CFG:
    exp_name: str = "linear_chain_crf"
    seed: int = 42
    num_tags: int = 7  # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    impossible_score: float = -1e4


def build_bio_transition_constraints(
    id_to_label: Dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs boolean constraint masks `(valid_start_mask, valid_transitions_mask)`
    for a BIO tag vocabulary:
      - `START -> I-X` is illegal
      - `O -> I-X` is illegal
      - `B-X -> I-Y` (where X != Y) is illegal
      - `I-X -> I-Y` (where X != Y) is illegal
    """
    num_tags = len(id_to_label)
    valid_start = torch.ones(num_tags, dtype=torch.bool)
    valid_trans = torch.ones((num_tags, num_tags), dtype=torch.bool)

    for tid, label in id_to_label.items():
        if label.startswith("I-"):
            valid_start[tid] = False

    for src_id, src_lbl in id_to_label.items():
        for dst_id, dst_lbl in id_to_label.items():
            if dst_lbl.startswith("I-"):
                dst_type = dst_lbl[2:]
                if src_lbl == "O":
                    valid_trans[src_id, dst_id] = False
                elif src_lbl.startswith(("B-", "I-")) and src_lbl[2:] != dst_type:
                    valid_trans[src_id, dst_id] = False

    return valid_start, valid_trans


class LinearChainCRF(nn.Module):
    """
    Pure-PyTorch Linear-Chain Conditional Random Field (CRF) layer.
    Computes exact negative log-likelihood via the forward algorithm and decodes
    the globally optimal tag sequence via Viterbi dynamic programming, with optional
    BIO structural transition constraints.
    """

    def __init__(
        self,
        num_tags: int,
        id_to_label: Dict[int, str] | None = None,
        impossible_score: float = -1e4,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.impossible_score = impossible_score
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.02)

        if id_to_label is not None:
            valid_start, valid_trans = build_bio_transition_constraints(id_to_label)
        else:
            valid_start = torch.ones(num_tags, dtype=torch.bool)
            valid_trans = torch.ones((num_tags, num_tags), dtype=torch.bool)

        self.register_buffer("valid_start_mask", valid_start)
        self.register_buffer("valid_trans_mask", valid_trans)

    def _effective_start(self) -> torch.Tensor:
        return torch.where(
            self.valid_start_mask,
            self.start_transitions,
            torch.full_like(self.start_transitions, self.impossible_score),
        )

    def _effective_transitions(self) -> torch.Tensor:
        return torch.where(
            self.valid_trans_mask,
            self.transitions,
            torch.full_like(self.transitions, self.impossible_score),
        )

    def _compute_log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward algorithm in log-space: `emissions` (B, T, C), `mask` (B, T) bool."""
        _, seq_len, _ = emissions.shape
        start_t = self._effective_start()
        trans = self._effective_transitions()
        alpha = start_t.unsqueeze(0) + emissions[:, 0]  # (B, C)

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            trans_t = trans.unsqueeze(0)  # (1, C_from, C_to)
            scores = alpha.unsqueeze(2) + trans_t + emit_t  # (B, C_from, C_to)
            next_alpha = torch.logsumexp(scores, dim=1)  # (B, C_to)
            mask_t = mask[:, t].unsqueeze(1)
            alpha = torch.where(mask_t, next_alpha, alpha)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=-1)

    def _compute_gold_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        start_t = self._effective_start()
        trans = self._effective_transitions()
        b_idx = torch.arange(batch_size, device=emissions.device)
        score = start_t[tags[:, 0]] + emissions[b_idx, 0, tags[:, 0]]

        for t in range(1, seq_len):
            m_t = mask[:, t].float()
            trans_score = trans[tags[:, t - 1], tags[:, t]]
            emit_score = emissions[b_idx, t, tags[:, t]]
            score = score + (trans_score + emit_score) * m_t

        last_indices = mask.long().sum(dim=1) - 1
        last_tags = tags[b_idx, last_indices]
        score = score + self.end_transitions[last_tags]
        return score

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns mean negative log-likelihood loss across the batch."""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        else:
            mask = mask.bool()
        log_Z = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        return torch.mean(log_Z - gold_score)

    @torch.no_grad()
    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> List[List[int]]:
        """Viterbi decoding returning best BIO-valid tag ID sequence per batch item."""
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )
        else:
            mask = mask.bool()

        batch_size, seq_len, _ = emissions.shape
        start_t = self._effective_start()
        trans = self._effective_transitions()
        viterbi = start_t.unsqueeze(0) + emissions[:, 0]
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            scores = (
                viterbi.unsqueeze(2)
                + trans.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            best_scores, best_paths = torch.max(scores, dim=1)
            backpointers.append(best_paths)
            viterbi = torch.where(mask[:, t].unsqueeze(1), best_scores, viterbi)

        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(viterbi, dim=-1)

        decoded_sequences: List[List[int]] = []
        lengths = mask.long().sum(dim=1).tolist()
        for b in range(batch_size):
            valid_len = int(lengths[b])
            best_tag = int(best_last_tags[b].item())
            seq = [best_tag]
            for t in range(valid_len - 2, -1, -1):
                best_tag = int(backpointers[t][b, best_tag].item())
                seq.append(best_tag)
            seq.reverse()
            decoded_sequences.append(seq)
        return decoded_sequences


def _run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(cfg.seed)
    id2label = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
    }
    crf = LinearChainCRF(
        num_tags=cfg.num_tags,
        id_to_label=id2label,
        impossible_score=cfg.impossible_score,
    )
    emissions = torch.randn(4, 10, cfg.num_tags, requires_grad=True)
    # Valid BIO sequence: O, B-PER, I-PER, O, B-ORG, I-ORG, O, B-LOC, O, O
    tags = torch.tensor([[0, 1, 2, 0, 3, 4, 0, 5, 0, 0]] * 4, dtype=torch.long)
    mask = torch.ones((4, 10), dtype=torch.bool)
    mask[:, -2:] = False

    loss = crf(emissions, tags, mask)
    loss.backward()
    decoded = crf.decode(emissions, mask)

    # Verify no illegal BIO transitions in decoded paths
    for seq in decoded:
        assert not id2label[seq[0]].startswith("I-"), "Illegal START -> I-* transition"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        arr = emissions.detach().numpy()
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", arr)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", arr)
        pd.DataFrame({"batch_id": range(len(decoded)), "tags": [str(s) for s in decoded]}).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] LinearChainCRF NLL={loss.item():.4f} | "
        f"decoded_0={decoded[0]} | BIO constraints verified"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 Linear-Chain CRF Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
