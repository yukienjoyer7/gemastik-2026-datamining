"""
Task 07 — NER Heads & Character-Level Span Alignment Modules
============================================================
Reusable neural & alignment components for Token Classification and Span Extraction:
  1. `LinearChainCRF`: Pure-PyTorch Linear-Chain Conditional Random Field layer
     with exact log-sum-exp forward partition function and Viterbi sequence decoding.
  2. `token_probs_to_char_probs`: Converts sliding-window token probabilities
     `(n_tokens, num_classes)` and `offset_mapping` into character-level probability
     arrays `(doc_char_len, num_classes)` — allowing seamless multi-tokenizer ensembling
     (e.g., blending DeBERTa-v3 SentencePiece + RoBERTa BPE + IndoBERT WordPiece).
  3. `char_probs_to_spans`: Decodes character-level class probability matrices into
     clean `(start_char, end_char, label, mean_prob)` entity spans.
  4. `evaluate_entity_spans`: Strict entity-level Micro-F1, Macro-F1, and F_beta
     (e.g., F5 for PII Data Detection).

Usage:
  python3 ner_heads_and_alignment.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

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


SpanTuple = Tuple[int, int, str]


@dataclass
class CFG:
    exp_name: str = "ner_modules"
    seed: int = 42
    num_tags: int = 7  # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    beta: float = 5.0


class LinearChainCRF(nn.Module):
    """
    Pure-PyTorch Linear-Chain Conditional Random Field (CRF) layer.
    Computes exact negative log-likelihood via the forward algorithm and decodes
    the globally optimal tag sequence via Viterbi dynamic programming.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.02)

    def _compute_log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward algorithm in log-space: `emissions` (B, T, C), `mask` (B, T) bool."""
        batch_size, seq_len, _ = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            trans_t = self.transitions.unsqueeze(0)  # (1, C_from, C_to)
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
        b_idx = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[tags[:, 0]] + emissions[b_idx, 0, tags[:, 0]]

        for t in range(1, seq_len):
            m_t = mask[:, t].float()
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
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
        """Viterbi decoding returning best tag ID sequence per batch item."""
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            scores = (
                viterbi.unsqueeze(2)
                + self.transitions.unsqueeze(0)
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


def token_probs_to_char_probs(
    doc_char_len: int,
    window_token_probs: Sequence[np.ndarray],
    window_offsets: Sequence[Sequence[Tuple[int, int]]],
) -> np.ndarray:
    """
    Projects overlapping sliding-window token probabilities onto a unified
    character-level probability matrix `(doc_char_len, num_classes)`.
    Characters not covered by any token default to class 0 (`O` = Outside).
    """
    if not window_token_probs:
        return np.zeros((doc_char_len, 1), dtype=np.float32)

    num_classes = window_token_probs[0].shape[-1]
    char_prob_sum = np.zeros((doc_char_len, num_classes), dtype=np.float32)
    char_counts = np.zeros((doc_char_len, 1), dtype=np.float32)

    for probs, offsets in zip(window_token_probs, window_offsets):
        for tok_idx, (s, e) in enumerate(offsets):
            s_c = max(0, min(doc_char_len, int(s)))
            e_c = max(0, min(doc_char_len, int(e)))
            if e_c > s_c and tok_idx < len(probs):
                char_prob_sum[s_c:e_c] += probs[tok_idx]
                char_counts[s_c:e_c] += 1.0

    unvisited = char_counts[:, 0] == 0.0
    char_counts[unvisited, 0] = 1.0
    char_prob_sum[unvisited, 0] = 1.0
    return char_prob_sum / char_counts


def char_probs_to_spans(
    text: str,
    char_probs: np.ndarray,
    id_to_label: Dict[int, str],
    class_thresholds: Dict[str, float] | None = None,
    min_span_chars: int = 2,
) -> List[Tuple[int, int, str, float]]:
    """
    Decodes a `(doc_char_len, num_classes)` probability array into entity spans
    `(start_char, end_char, label, mean_prob)`, stripping leading/trailing whitespace.
    """
    class_thresholds = class_thresholds or {}
    pred_ids = np.argmax(char_probs, axis=-1)

    # Apply per-class thresholds if provided
    for cid, cname in id_to_label.items():
        if cid == 0 or cname == "O":
            continue
        base_lbl = cname.split("-", 1)[-1]
        thr = class_thresholds.get(base_lbl, class_thresholds.get(cname, 0.0))
        if thr > 0.0:
            low_mask = (pred_ids == cid) & (char_probs[:, cid] < thr)
            pred_ids[low_mask] = 0

    spans: List[Tuple[int, int, str, float]] = []
    n_chars = len(text)
    for k in range(1, n_chars - 1):
        if text[k].isspace() and pred_ids[k - 1] > 0 and pred_ids[k - 1] == pred_ids[k + 1]:
            pred_ids[k] = pred_ids[k - 1]
    i = 0
    while i < n_chars:
        cid = int(pred_ids[i])
        cname = id_to_label.get(cid, "O")
        if cid == 0 or cname == "O":
            i += 1
            continue
        base_lbl = cname.split("-", 1)[-1]
        start = i
        while i < n_chars:
            next_cid = int(pred_ids[i])
            next_name = id_to_label.get(next_cid, "O")
            if next_cid == 0 or next_name == "O" or next_name.split("-", 1)[-1] != base_lbl:
                break
            i += 1
        end = i

        # Trim whitespace boundaries
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if (end - start) >= min_span_chars:
            mean_p = float(np.mean(np.max(char_probs[start:end], axis=-1)))
            spans.append((start, end, base_lbl, mean_p))

    return spans


def evaluate_entity_spans(
    pred_spans_docs: Sequence[Sequence[SpanTuple]],
    gt_spans_docs: Sequence[Sequence[SpanTuple]],
    beta: float = 1.0,
) -> Dict[str, float]:
    """Computes strict entity-level Micro-F1, Macro-F1, and Micro-F_beta."""
    tp_all, fp_all, fn_all = 0, 0, 0
    labels: Set[str] = set()
    for gts in gt_spans_docs:
        for _, _, lbl in gts:
            labels.add(lbl)
    for preds in pred_spans_docs:
        for _, _, lbl in preds:
            labels.add(lbl)

    per_label_f1: List[float] = []
    for lbl in sorted(labels):
        tp, fp, fn = 0, 0, 0
        for preds, gts in zip(pred_spans_docs, gt_spans_docs):
            p_set = {(s, e, l) for s, e, l in preds if l == lbl}
            g_set = {(s, e, l) for s, e, l in gts if l == lbl}
            tp += len(p_set & g_set)
            fp += len(p_set - g_set)
            fn += len(g_set - p_set)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        denom = 2 * tp + fp + fn
        per_label_f1.append((2.0 * tp / denom) if denom > 0 else 0.0)

    prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    beta2 = beta * beta
    micro_fbeta = (
        (1.0 + beta2) * prec * rec / (beta2 * prec + rec)
        if (prec + rec) > 0
        else 0.0
    )

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        f"micro_f{beta:g}": float(micro_fbeta),
    }


def _run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(cfg.seed)

    # 1. Test LinearChainCRF forward & Viterbi decode
    crf = LinearChainCRF(num_tags=cfg.num_tags)
    emissions = torch.randn(4, 10, cfg.num_tags, requires_grad=True)
    tags = torch.randint(0, cfg.num_tags, (4, 10))
    mask = torch.ones((4, 10), dtype=torch.bool)
    mask[:, -2:] = False
    loss = crf(emissions, tags, mask)
    loss.backward()
    decoded = crf.decode(emissions, mask)

    # 2. Test Character-Level Span Probability Aggregation & Decoding
    text = "Budi Santoso works at Bank Mandiri in Jakarta."
    id2label = {0: "O", 1: "PER", 2: "ORG", 3: "LOC"}
    win_offsets = [[(0, 12), (13, 21), (22, 34), (35, 37), (38, 45)]]
    win_probs = [
        np.array(
            [
                [0.02, 0.94, 0.02, 0.02],  # Budi Santoso -> PER
                [0.95, 0.02, 0.02, 0.01],  # works at -> O
                [0.03, 0.02, 0.92, 0.03],  # Bank Mandiri -> ORG
                [0.96, 0.01, 0.02, 0.01],  # in -> O
                [0.02, 0.02, 0.03, 0.93],  # Jakarta -> LOC
            ],
            dtype=np.float32,
        )
    ]
    char_probs = token_probs_to_char_probs(len(text), win_probs, win_offsets)
    spans_with_prob = char_probs_to_spans(text, char_probs, id2label)
    pred_spans = [(s, e, l) for s, e, l, _ in spans_with_prob]
    gt_spans = [(0, 12, "PER"), (22, 34, "ORG"), (38, 45, "LOC")]
    metrics = evaluate_entity_spans([pred_spans], [gt_spans], beta=cfg.beta)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", char_probs)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", char_probs)
        pd.DataFrame(spans_with_prob, columns=["start", "end", "label", "prob"]).to_csv(
            out_dir / f"sub_{cfg.exp_name}.csv", index=False
        )

    print(
        f"[Smoke Test OK] CRF NLL={loss.item():.4f}, Viterbi len={len(decoded[0])}, "
        f"Char-Aligned Micro-F1={metrics['micro_f1']:.4f}, "
        f"Micro-F5={metrics['micro_f5']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 NER Modules")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
