#!/usr/bin/env python3
"""
Task 02 — Text Pair Classification & Semantic Similarity: EDA & Dataset Pipeline
================================================================================
Standalone script and reusable module providing:
1. Pair Graph & Transitive Leakage Diagnostics:
   - Union-Find Disjoint Set over `(text_a, text_b)` to detect transitive connected
     components, component size quantiles, node degree distributions, and hub nodes.
   - Pair length ratio & character/word difference diagnostics.
   - Symmetry & reversed-pair overlap checks (`(A, B)` vs `(B, A)`).
2. PyTorch Cross-Encoder Dataset:
   - `TextPairCrossEncoderDataset`: Formats `[CLS] context [SEP] text_a [SEP] text_b [SEP]`
     with **Balanced Token Truncation** so neither `text_a` nor `text_b` is completely
     cut off when the combined sequence exceeds `max_len`.

Usage:
    python eda_and_data.py --train-path train_folds.csv --test-path test.csv
    python eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import site
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_venv_site = (
    pathlib.Path.home()
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
from torch.utils.data import DataLoader, Dataset


@dataclass
class CFG:
    """Configuration for Text Pair Similarity EDA and Cross-Encoder Dataset."""

    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    id_col: str = "id"
    text_a_col: str = "text_a"
    text_b_col: str = "text_b"
    context_col: str = "context"
    target_col: str = "label"
    fold_col: str = "fold"

    max_len: int = 256
    max_context_tokens: int = 32
    cls_token_id: int = 101
    sep_token_id: int = 102
    pad_token_id: int = 0

    batch_size: int = 16
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Union-Find Pair Graph & Transitive Leakage Analyzer
# =============================================================================


class UnionFind:
    """Disjoint-Set (Union-Find) with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def analyze_pair_graph(
    df: pd.DataFrame, text_a_col: str, text_b_col: str, fold_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds an undirected graph over unique texts in `(text_a, text_b)` using Union-Find.
    Reports connected component statistics, degree distribution, hub nodes, and cross-fold
    component leakage if `fold_col` is present.
    """
    uf = UnionFind()
    adj: Dict[str, set] = defaultdict(set)
    texts_a = df[text_a_col].fillna("").astype(str).tolist()
    texts_b = df[text_b_col].fillna("").astype(str).tolist()

    for a, b in zip(texts_a, texts_b):
        uf.union(a, b)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
        else:
            _ = adj[a]

    comp_members: Dict[str, List[str]] = defaultdict(list)
    for node in uf.parent:
        comp_members[uf.find(node)].append(node)

    comp_sizes = np.array([len(v) for v in comp_members.values()], dtype=np.int64)
    degrees = np.array([len(adj[node]) for node in uf.parent], dtype=np.int64)
    top_hubs = sorted(((node, len(adj[node])) for node in uf.parent), key=lambda x: x[1], reverse=True)[:5]

    # Check if connected components span multiple folds (transitive leakage indicator)
    leaked_components = 0
    if fold_col is not None and fold_col in df.columns:
        comp_folds: Dict[str, set] = defaultdict(set)
        for a, f in zip(texts_a, df[fold_col].values):
            comp_folds[uf.find(a)].add(int(f))
        leaked_components = sum(1 for folds_set in comp_folds.values() if len(folds_set) > 1)

    return {
        "num_pairs": len(df),
        "num_unique_nodes": len(uf.parent),
        "num_connected_components": len(comp_sizes),
        "max_component_size": int(comp_sizes.max()) if len(comp_sizes) else 0,
        "p90_component_size": float(np.percentile(comp_sizes, 90)) if len(comp_sizes) else 0.0,
        "mean_node_degree": round(float(degrees.mean()), 3) if len(degrees) else 0.0,
        "max_node_degree": int(degrees.max()) if len(degrees) else 0,
        "cross_fold_leaked_components": leaked_components,
        "top_hub_nodes": [(n[:40], int(d)) for n, d in top_hubs],
    }


def analyze_length_ratios_and_symmetry(
    df: pd.DataFrame,
    text_a_col: str,
    text_b_col: str,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze pair length ratios, word differences, and `(A, B)` vs `(B, A)` symmetry."""
    a_series = df[text_a_col].fillna("").astype(str)
    b_series = df[text_b_col].fillna("").astype(str)

    len_a = np.maximum(a_series.str.split().str.len().values, 1)
    len_b = np.maximum(b_series.str.split().str.len().values, 1)

    len_ratio = np.minimum(len_a, len_b) / np.maximum(len_a, len_b)
    abs_diff = np.abs(len_a - len_b)

    forward_pairs = list(zip(a_series.tolist(), b_series.tolist()))
    forward_set = set(forward_pairs)
    reversed_matches = sum(1 for a, b in forward_pairs if a != b and (b, a) in forward_set)
    identical_self_pairs = sum(1 for a, b in forward_pairs if a == b)

    label_asymmetry = 0
    if target_col is not None and target_col in df.columns:
        pair_to_label = {(a, b): y for (a, b), y in zip(forward_pairs, df[target_col].values)}
        for (a, b), y in pair_to_label.items():
            if (b, a) in pair_to_label and not np.isclose(y, pair_to_label[(b, a)]):
                label_asymmetry += 1

    return {
        "length_ratio_mean": round(float(len_ratio.mean()), 4),
        "length_ratio_p10": round(float(np.percentile(len_ratio, 10)), 4),
        "length_ratio_median": round(float(np.median(len_ratio)), 4),
        "word_diff_mean": round(float(abs_diff.mean()), 2),
        "reversed_pair_count": reversed_matches,
        "self_pair_count": identical_self_pairs,
        "asymmetric_label_pairs": label_asymmetry,
    }


# =============================================================================
# 2. Balanced Token Truncation & Cross-Encoder Dataset
# =============================================================================


def balanced_pair_truncate(
    ids_a: List[int],
    ids_b: List[int],
    ids_ctx: Optional[List[int]],
    max_len: int = 256,
    max_context_tokens: int = 32,
    cls_token_id: int = 101,
    sep_token_id: int = 102,
) -> Tuple[List[int], List[int]]:
    """
    Constructs `[CLS] context [SEP] text_a [SEP] text_b [SEP]` (or `[CLS] text_a [SEP] text_b [SEP]`
    if no context is given) while ensuring balanced token budget allocation between `text_a`
    and `text_b`:
      - Neither `text_a` nor `text_b` can starve the other if both exceed half the budget.
      - If one side is shorter than its half-budget, its unused token budget is donated
        to the longer side.
    Returns `(input_ids, token_type_ids)`.
    """
    has_ctx = ids_ctx is not None and len(ids_ctx) > 0
    if has_ctx:
        ctx_trimmed = list(ids_ctx[:max_context_tokens])
        special_tokens_count = 4  # [CLS] ctx [SEP] a [SEP] b [SEP]
        pair_budget = max(2, max_len - special_tokens_count - len(ctx_trimmed))
    else:
        ctx_trimmed = []
        special_tokens_count = 3  # [CLS] a [SEP] b [SEP]
        pair_budget = max(2, max_len - special_tokens_count)

    half_budget = pair_budget // 2
    if len(ids_a) + len(ids_b) <= pair_budget:
        trunc_a = list(ids_a)
        trunc_b = list(ids_b)
    elif len(ids_a) <= half_budget:
        trunc_a = list(ids_a)
        trunc_b = list(ids_b[: pair_budget - len(trunc_a)])
    elif len(ids_b) <= pair_budget - half_budget:
        trunc_b = list(ids_b)
        trunc_a = list(ids_a[: pair_budget - len(trunc_b)])
    else:
        trunc_a = list(ids_a[:half_budget])
        trunc_b = list(ids_b[: pair_budget - half_budget])

    if has_ctx:
        input_ids = (
            [cls_token_id]
            + ctx_trimmed
            + [sep_token_id]
            + trunc_a
            + [sep_token_id]
            + trunc_b
            + [sep_token_id]
        )
        seg_0_len = 1 + len(ctx_trimmed) + 1 + len(trunc_a) + 1
        seg_1_len = len(trunc_b) + 1
    else:
        input_ids = [cls_token_id] + trunc_a + [sep_token_id] + trunc_b + [sep_token_id]
        seg_0_len = 1 + len(trunc_a) + 1
        seg_1_len = len(trunc_b) + 1

    token_type_ids = [0] * seg_0_len + [1] * seg_1_len
    return input_ids, token_type_ids


def fallback_word_tokenizer(text: str, vocab_size: int = 8000) -> List[int]:
    tokens = re.findall(r"\w+|\S", str(text).lower())
    return [10 + (hash(t) % (vocab_size - 10)) for t in tokens]


class TextPairCrossEncoderDataset(Dataset):
    """
    PyTorch Cross-Encoder dataset formatting `[CLS] context [SEP] text_a [SEP] text_b [SEP]`
    with balanced truncation and optional reversed `(text_b, text_a)` swapping for TTA.
    """

    def __init__(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
        contexts: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[Union[int, float]]] = None,
        tokenizer: Optional[Any] = None,
        max_len: int = 256,
        max_context_tokens: int = 32,
        swap_ab: bool = False,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
    ) -> None:
        self.texts_a = [str(x) for x in (texts_b if swap_ab else texts_a)]
        self.texts_b = [str(x) for x in (texts_a if swap_ab else texts_b)]
        self.contexts = [str(c) for c in contexts] if contexts is not None else None
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_context_tokens = max_context_tokens
        self.cls_token_id = int(getattr(tokenizer, "cls_token_id", cls_token_id) or cls_token_id)
        self.sep_token_id = int(getattr(tokenizer, "sep_token_id", sep_token_id) or sep_token_id)

    def __len__(self) -> int:
        return len(self.texts_a)

    def _encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self.tokenizer is None:
            return fallback_word_tokenizer(text)
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        return fallback_word_tokenizer(text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ids_a = self._encode(self.texts_a[idx])
        ids_b = self._encode(self.texts_b[idx])
        ids_ctx = self._encode(self.contexts[idx]) if self.contexts is not None else None

        input_ids, token_type_ids = balanced_pair_truncate(
            ids_a,
            ids_b,
            ids_ctx,
            max_len=self.max_len,
            max_context_tokens=self.max_context_tokens,
            cls_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
        )
        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": token_type_ids,
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class PairDynamicCollator:
    """Dynamic batch padding collator for `TextPairCrossEncoderDataset`."""

    def __init__(self, pad_token_id: int = 0, label_dtype: torch.dtype = torch.float32) -> None:
        self.pad_token_id = pad_token_id
        self.label_dtype = label_dtype

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_l = max(len(x["input_ids"]) for x in batch)
        ids, mask, ttypes = [], [], []
        for x in batch:
            pad_n = max_l - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_token_id] * pad_n)
            mask.append(x["attention_mask"] + [0] * pad_n)
            ttypes.append(x["token_type_ids"] + [0] * pad_n)

        out: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(ttypes, dtype=torch.long),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=self.label_dtype)
        return out


# =============================================================================
# 3. Synthetic Generator & CLI Entrypoint
# =============================================================================


def generate_synthetic_pairs(n_train: int = 100, n_test: int = 30, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    entities = [f"question_node_{i}: how to optimize transformer gradient descent step {i}" for i in range(35)]
    contexts = ["A47 ML Optimization", "G06 Natural Language", "H04 Signal Processing"]

    train_rows = []
    for i in range(n_train):
        u_idx = int(rng.integers(0, len(entities)))
        v_idx = (u_idx + int(rng.integers(1, 4))) % len(entities)
        # Create occasional long strings to test balanced truncation
        extra_a = " ".join(["extra_token_a"] * (180 if i % 7 == 0 else 5))
        extra_b = " ".join(["extra_token_b"] * (180 if i % 5 == 0 else 5))
        train_rows.append(
            {
                "id": f"tr_{i}",
                "context": contexts[i % len(contexts)],
                "text_a": f"{entities[u_idx]} {extra_a}",
                "text_b": f"{entities[v_idx]} {extra_b}",
                "label": float(1 if abs(u_idx - v_idx) <= 1 else 0),
                "fold": i % 5,
            }
        )

    test_rows = []
    for i in range(n_test):
        u_idx = int(rng.integers(0, len(entities)))
        v_idx = (u_idx + 1) % len(entities)
        test_rows.append(
            {
                "id": f"te_{i}",
                "context": contexts[i % len(contexts)],
                "text_a": entities[u_idx],
                "text_b": entities[v_idx],
            }
        )

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Text Pair EDA & Cross-Encoder Dataset")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--max-len", type=int, default=CFG.max_len)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        max_len=args.max_len,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        print("[SMOKE TEST] Generating synthetic text-pair graph corpus...")
        train_df, _ = generate_synthetic_pairs(seed=cfg.seed)
    else:
        train_df = load_table(cfg.train_path)

    print("=" * 72)
    print("1. PAIR GRAPH CONNECTED COMPONENTS & HUB DIAGNOSTICS")
    print("=" * 72)
    graph_stats = analyze_pair_graph(train_df, cfg.text_a_col, cfg.text_b_col, cfg.fold_col)
    for k, v in graph_stats.items():
        print(f"  {k:<32}: {v}")

    print("\n" + "=" * 72)
    print("2. PAIR LENGTH RATIO & SYMMETRY ANALYSIS")
    print("=" * 72)
    sym_stats = analyze_length_ratios_and_symmetry(
        train_df, cfg.text_a_col, cfg.text_b_col, cfg.target_col
    )
    for k, v in sym_stats.items():
        print(f"  {k:<32}: {v}")

    contexts = train_df[cfg.context_col].values if cfg.context_col in train_df.columns else None
    ds = TextPairCrossEncoderDataset(
        texts_a=train_df[cfg.text_a_col].values,
        texts_b=train_df[cfg.text_b_col].values,
        contexts=contexts,
        labels=train_df[cfg.target_col].values,
        max_len=cfg.max_len,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=PairDynamicCollator())
    batch = next(iter(loader))

    assert batch["input_ids"].shape[1] <= cfg.max_len, "Balanced truncation exceeded max_len!"
    # Verify both segment 0 and segment 1 tokens exist in every sequence
    has_seg_1 = (batch["token_type_ids"].sum(dim=1) > 1).all().item()
    assert has_seg_1, "text_b was completely cut off!"

    print("\n" + "=" * 72)
    print("3. BALANCED CROSS-ENCODER DATASET CHECK")
    print("=" * 72)
    print(f"  Batch input_ids shape     : {tuple(batch['input_ids'].shape)}")
    print(f"  Batch token_type_ids shape: {tuple(batch['token_type_ids'].shape)}")
    print(f"  Both segments preserved   : {has_seg_1}")
    print("[SUCCESS] Task 02 EDA and Cross-Encoder dataset verification passed.")


if __name__ == "__main__":
    main()
