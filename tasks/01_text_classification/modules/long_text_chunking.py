#!/usr/bin/env python3
"""
Task 01 — Text Classification: Long Document Head+Tail Truncation & Sliding Window Chunking
===========================================================================================
Atomic standalone utilities for handling long documents (>512 tokens) in Transformer classifiers:
  1. `head_tail_truncate_ids`: Preserves the opening `head_len` tokens and closing `tail_len`
     tokens (Sun et al., 2019 — "How to Fine-Tune BERT for Text Classification?").
  2. `split_into_sliding_windows`: Splits long token sequences into overlapping chunks of length
     `window_size` with `stride`, preserving `[CLS]` and `[SEP]` boundary tokens.
  3. `SlidingWindowChunkAggregator`: PyTorch module that aggregates variable numbers of chunk
     representations or logits per document back into document-level representations using
     `mean`, `max`, or learned `attention` pooling.

Usage:
    python3 tasks/01_text_classification/modules/long_text_chunking.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import site
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    """Configuration for Long Text Chunking Smoke Test."""

    max_len: int = 64
    head_len: int = 20
    window_size: int = 32
    stride: int = 16
    hidden_size: int = 32
    smoke_test: bool = False


def head_tail_truncate_ids(
    token_ids: Sequence[int],
    max_len: int = 512,
    head_len: int = 128,
    cls_token_id: Optional[int] = None,
    sep_token_id: Optional[int] = None,
) -> List[int]:
    """
    Truncates a token ID sequence to `max_len` by concatenating the first `head_len` tokens
    and the last `max_len - head_len` tokens. Optionally wraps with `cls_token_id` and `sep_token_id`.
    """
    ids = list(token_ids)
    budget = max_len - (1 if cls_token_id is not None else 0) - (1 if sep_token_id is not None else 0)
    if len(ids) > budget:
        h = min(head_len, budget)
        t = max(0, budget - h)
        ids = ids[:h] + (ids[-t:] if t > 0 else [])

    if cls_token_id is not None:
        ids = [cls_token_id] + ids
    if sep_token_id is not None:
        ids = ids + [sep_token_id]
    return ids


def split_into_sliding_windows(
    token_ids: Sequence[int],
    window_size: int = 510,
    stride: int = 256,
    cls_token_id: Optional[int] = 101,
    sep_token_id: Optional[int] = 102,
    max_chunks: int = 8,
) -> List[List[int]]:
    """
    Splits a long sequence of body `token_ids` into overlapping windows of size `window_size`
    with step `stride`, optionally wrapping each window with `[CLS]` and `[SEP]`.
    """
    ids = list(token_ids)
    if not ids:
        chunk = []
        if cls_token_id is not None:
            chunk.append(cls_token_id)
        if sep_token_id is not None:
            chunk.append(sep_token_id)
        return [chunk]

    chunks: List[List[int]] = []
    start = 0
    while start < len(ids) and len(chunks) < max_chunks:
        piece = ids[start : start + window_size]
        if cls_token_id is not None:
            piece = [cls_token_id] + piece
        if sep_token_id is not None:
            piece = piece + [sep_token_id]
        chunks.append(piece)
        if start + window_size >= len(ids):
            break
        start += max(1, stride)
    return chunks


class SlidingWindowChunkAggregator(nn.Module):
    """
    Aggregates chunk-level embeddings or logits `(total_chunks, D)` into document-level
    tensors `(num_docs, D)` given `doc_chunk_counts` (or `chunk_to_doc_idx`), supporting
    `mode="mean"`, `"max"`, or `"attention"`.
    """

    def __init__(self, feature_dim: int, mode: str = "attention") -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.mode = mode
        if mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 1),
            )

    def forward(
        self, chunk_features: torch.Tensor, doc_chunk_counts: Sequence[int]
    ) -> torch.Tensor:
        docs: List[torch.Tensor] = []
        offset = 0
        for count in doc_chunk_counts:
            c = max(1, int(count))
            sub = chunk_features[offset : offset + c]  # (c, D)
            offset += c
            if self.mode == "mean":
                docs.append(sub.mean(dim=0))
            elif self.mode == "max":
                docs.append(sub.max(dim=0)[0])
            else:
                scores = self.attn(sub).squeeze(-1)  # (c,)
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)  # (c, 1)
                docs.append((sub * weights).sum(dim=0))
        return torch.stack(docs, dim=0)


def run_smoke_test(cfg: CFG) -> None:
    torch.manual_seed(42)
    raw_tokens = list(range(1, 150))

    ht = head_tail_truncate_ids(
        raw_tokens, max_len=cfg.max_len, head_len=cfg.head_len, cls_token_id=101, sep_token_id=102
    )
    assert len(ht) == cfg.max_len
    assert ht[0] == 101 and ht[-1] == 102

    windows = split_into_sliding_windows(
        raw_tokens, window_size=cfg.window_size, stride=cfg.stride, cls_token_id=101, sep_token_id=102
    )
    assert len(windows) >= 4
    assert len(windows[0]) == cfg.window_size + 2

    doc_counts = [3, 1, 4]
    total_chunks = sum(doc_counts)
    chunk_embs = torch.randn(total_chunks, cfg.hidden_size, requires_grad=True)

    for mode in ("mean", "max", "attention"):
        agg = SlidingWindowChunkAggregator(cfg.hidden_size, mode=mode)
        doc_embs = agg(chunk_embs, doc_counts)
        assert doc_embs.shape == (len(doc_counts), cfg.hidden_size)
        doc_embs.sum().backward(retain_graph=True)

    print("[SMOKE TEST] Task 01 Long Text Chunking (`long_text_chunking.py`)")
    print(f"  Head+Tail truncated length : {len(ht)} (head={ht[:4]} ... tail={ht[-4:]})")
    print(f"  Sliding window count       : {len(windows)} chunks of size {len(windows[0])}")
    print(f"  Aggregated document shape  : {tuple(doc_embs.shape)}")
    print("[SUCCESS] All long text truncation and sliding window checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 01 Long Text Chunking")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
