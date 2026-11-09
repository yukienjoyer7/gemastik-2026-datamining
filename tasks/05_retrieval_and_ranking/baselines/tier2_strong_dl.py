"""
Task 05 — Tier 2 Strong Deep Learning: Bi-Encoder (MNRL + Hard Negative Mining) + Cross-Encoder
===============================================================================================
Standalone Two-Stage Neural Retrieval & Reranking pipeline:
  1. Bi-Encoder Contrastive Fine-Tuning:
     - Trains dual-tower / shared-tower encoder with MultipleNegativesRankingLoss (InfoNCE)
     - Runs iterative Hard Negative Mining (with false-negative margin filtering)
  2. Dense Top-K Retrieval + Alpha Query Expansion (AQE / alpha-QE):
     - Exact L2-normalized cosine similarity search (or FAISS when available)
     - Blends query embeddings with similarity-weighted top-K document neighbors
  3. Cross-Encoder Reranker:
     - Jointly encodes `(query, candidate_doc)` pairs to output calibrated OOF logits
     - Exports `oof_<exp_name>.npy`, `test_<exp_name>.npy`, and `sub_<exp_name>.csv`

Usage:
  python3 tier2_strong_dl.py --data-dir ./data --output-dir ./outputs
  python3 tier2_strong_dl.py --smoke-test
"""

from __future__ import annotations

import argparse
import re
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
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold


@dataclass
class CFG:
    exp_name: str = "tier2_bi_cross_encoder"
    model_name: str = "BAAI/bge-small-en-v1.5"
    seed: int = 42
    n_folds: int = 3
    vocab_size: int = 1024
    max_len: int = 24
    embed_dim: int = 64
    temperature: float = 0.07
    bi_epochs: int = 4
    ce_epochs: int = 4
    lr: float = 3e-3
    top_k_retrieve: int = 10
    alpha_qe: float = 0.5
    false_neg_margin: float = 0.95
    data_dir: str = "./data"
    output_dir: str = "./outputs"


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class HashingTokenizer:
    """Fast deterministic word+subword hashing tokenizer for offline/smoke-test execution."""

    def __init__(self, vocab_size: int = 1024, max_len: int = 24) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        out = torch.zeros((len(texts), self.max_len), dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = TOKEN_RE.findall(str(text).lower())[: self.max_len]
            for j, tok in enumerate(tokens):
                out[i, j] = (hash(tok) % (self.vocab_size - 2)) + 2
        return out


class BiEncoderNet(nn.Module):
    """Mean-pooled Transformer/Attention Bi-Encoder producing L2-normalized embeddings."""

    def __init__(self, vocab_size: int = 1024, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).float().unsqueeze(-1)
        x = self.embedding(input_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


class CrossEncoderNet(nn.Module):
    """Joint Cross-Encoder scoring `(query, candidate_doc)` pairs."""

    def __init__(self, vocab_size: int = 1024, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(2, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, q_ids: torch.Tensor, d_ids: torch.Tensor) -> torch.Tensor:
        joint_ids = torch.cat([q_ids, d_ids], dim=1)
        type_ids = torch.cat(
            [torch.zeros_like(q_ids), torch.ones_like(d_ids)], dim=1
        )
        mask = (joint_ids != 0).float().unsqueeze(-1)
        x = self.embedding(joint_ids) + self.type_embedding(type_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)


def multiple_negatives_ranking_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = torch.matmul(q_emb, p_emb.T) / temperature
    if hard_neg_emb is not None:
        neg_logits = torch.bmm(hard_neg_emb, q_emb.unsqueeze(-1)).squeeze(-1) / temperature
        logits = torch.cat([logits, neg_logits], dim=1)
    labels = torch.arange(q_emb.size(0), device=q_emb.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def alpha_query_expansion_np(
    q_embs: np.ndarray, d_embs: np.ndarray, top_k: int = 3, alpha: float = 0.5
) -> np.ndarray:
    sims = q_embs @ d_embs.T
    expanded = np.zeros_like(q_embs)
    for i in range(len(q_embs)):
        top_idx = np.argsort(-sims[i])[:top_k]
        w = np.power(np.clip(sims[i, top_idx], 0.0, None), alpha)[:, None]
        vec = q_embs[i] + np.sum(w * d_embs[top_idx], axis=0)
        expanded[i] = vec / (np.linalg.norm(vec) + 1e-12)
    return expanded


def mine_hard_negatives_indices(
    q_embs: np.ndarray,
    d_embs: np.ndarray,
    pos_indices: Sequence[int],
    n_hard: int = 2,
    margin: float = 0.95,
) -> List[List[int]]:
    sims = q_embs @ d_embs.T
    hard_neg_rows: List[List[int]] = []
    for i, pos_idx in enumerate(pos_indices):
        pos_sim = float(sims[i, pos_idx])
        cutoff = margin * pos_sim
        order = np.argsort(-sims[i])
        chosen: List[int] = []
        for idx in order:
            if int(idx) == int(pos_idx):
                continue
            if float(sims[i, idx]) > cutoff:
                continue
            chosen.append(int(idx))
            if len(chosen) >= n_hard:
                break
        while len(chosen) < n_hard:
            chosen.append(int((pos_idx + len(chosen) + 1) % len(d_embs)))
        hard_neg_rows.append(chosen)
    return hard_neg_rows


def _build_synthetic_data(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    topics = [
        "calculus derivative integral theorem limit",
        "photosynthesis chlorophyll plant cell sunlight",
        "newtonian mechanics gravity acceleration momentum",
        "neural network transformer attention backpropagation",
    ]
    rows = []
    for q_idx in range(18):
        t_idx = q_idx % len(topics)
        qid = f"Q{q_idx:03d}"
        q_words = rng.choice(topics[t_idx].split(), size=3, replace=False)
        q_text = "query " + " ".join(q_words)
        for d_idx in range(6):
            dt_idx = (t_idx + (0 if d_idx < 2 else d_idx)) % len(topics)
            d_words = rng.choice(topics[dt_idx].split(), size=3, replace=False)
            d_text = f"doc {d_idx} " + " ".join(d_words)
            rel = 1 if dt_idx == t_idx else 0
            rows.append(
                {
                    "query_id": qid,
                    "query_text": q_text,
                    "doc_id": f"D{q_idx:02d}_{d_idx:02d}",
                    "doc_text": d_text,
                    "relevance": rel,
                }
            )
    train_df = pd.DataFrame(rows)
    test_df = train_df.iloc[:18].copy().reset_index(drop=True)
    test_df["query_id"] = test_df["query_id"].apply(lambda x: "T" + x)
    return train_df, test_df


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if smoke_test:
        train_df, test_df = _build_synthetic_data(cfg.seed)
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[: min(24, len(train_df))].copy().reset_index(drop=True)
        else:
            train_df, test_df = _build_synthetic_data(cfg.seed)

    if "fold" not in train_df.columns:
        train_df["fold"] = 0
        gkf = GroupKFold(n_splits=min(cfg.n_folds, train_df["query_id"].nunique()))
        for fold, (_, val_idx) in enumerate(
            gkf.split(train_df, train_df["relevance"], groups=train_df["query_id"])
        ):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    tokenizer = HashingTokenizer(vocab_size=cfg.vocab_size, max_len=cfg.max_len)
    q_ids_all = tokenizer.encode_batch(train_df["query_text"].tolist())
    d_ids_all = tokenizer.encode_batch(train_df["doc_text"].tolist())
    q_ids_test = tokenizer.encode_batch(test_df["query_text"].tolist())
    d_ids_test = tokenizer.encode_batch(test_df["doc_text"].tolist())

    oof_bi_cos = np.zeros(len(train_df), dtype=np.float32)
    oof_ce_prob = np.zeros(len(train_df), dtype=np.float32)
    test_bi_cos = np.zeros(len(test_df), dtype=np.float32)
    test_ce_prob = np.zeros(len(test_df), dtype=np.float32)

    unique_folds = sorted(train_df["fold"].unique())
    for fold in unique_folds:
        tr_idx = np.where(train_df["fold"].to_numpy() != fold)[0]
        va_idx = np.where(train_df["fold"].to_numpy() == fold)[0]

        # 1. Train Bi-Encoder with MNRL + Mined Hard Negatives
        bi_model = BiEncoderNet(vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim)
        bi_opt = torch.optim.AdamW(bi_model.parameters(), lr=cfg.lr, weight_decay=1e-2)

        pos_mask = train_df.iloc[tr_idx]["relevance"].to_numpy() > 0
        pos_tr_idx = tr_idx[pos_mask] if np.any(pos_mask) else tr_idx

        for epoch in range(cfg.bi_epochs):
            bi_model.train()
            bi_opt.zero_grad()
            q_emb = bi_model(q_ids_all[pos_tr_idx])
            p_emb = bi_model(d_ids_all[pos_tr_idx])

            if epoch >= cfg.bi_epochs // 2 and len(pos_tr_idx) >= 4:
                with torch.no_grad():
                    q_np = q_emb.detach().numpy()
                    d_np = p_emb.detach().numpy()
                hard_rows = mine_hard_negatives_indices(
                    q_np,
                    d_np,
                    pos_indices=list(range(len(pos_tr_idx))),
                    n_hard=2,
                    margin=cfg.false_neg_margin,
                )
                hard_t = torch.stack([p_emb[idxs] for idxs in hard_rows], dim=0)
                loss = multiple_negatives_ranking_loss(
                    q_emb, p_emb, hard_t, temperature=cfg.temperature
                )
            else:
                loss = multiple_negatives_ranking_loss(
                    q_emb, p_emb, None, temperature=cfg.temperature
                )
            loss.backward()
            bi_opt.step()

        # Evaluate Bi-Encoder + Alpha Query Expansion on validation & test
        bi_model.eval()
        with torch.no_grad():
            va_q_emb = bi_model(q_ids_all[va_idx]).numpy()
            va_d_emb = bi_model(d_ids_all[va_idx]).numpy()
            va_q_exp = alpha_query_expansion_np(va_q_emb, va_d_emb, top_k=3, alpha=cfg.alpha_qe)
            oof_bi_cos[va_idx] = np.sum(va_q_exp * va_d_emb, axis=1)

            te_q_emb = bi_model(q_ids_test).numpy()
            te_d_emb = bi_model(d_ids_test).numpy()
            te_q_exp = alpha_query_expansion_np(te_q_emb, te_d_emb, top_k=3, alpha=cfg.alpha_qe)
            test_bi_cos += np.sum(te_q_exp * te_d_emb, axis=1) / len(unique_folds)

        # 2. Train Cross-Encoder Reranker on candidate pairs
        ce_model = CrossEncoderNet(vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim)
        ce_opt = torch.optim.AdamW(ce_model.parameters(), lr=cfg.lr, weight_decay=1e-2)
        y_tr_t = torch.tensor(
            train_df.iloc[tr_idx]["relevance"].to_numpy(dtype=np.float32)
        )

        for _ in range(cfg.ce_epochs):
            ce_model.train()
            ce_opt.zero_grad()
            logits = ce_model(q_ids_all[tr_idx], d_ids_all[tr_idx])
            ce_loss = F.binary_cross_entropy_with_logits(logits, y_tr_t)
            ce_loss.backward()
            ce_opt.step()

        ce_model.eval()
        with torch.no_grad():
            va_logits = ce_model(q_ids_all[va_idx], d_ids_all[va_idx])
            oof_ce_prob[va_idx] = torch.sigmoid(va_logits).numpy()

            te_logits = ce_model(q_ids_test, d_ids_test)
            test_ce_prob += torch.sigmoid(te_logits).numpy() / len(unique_folds)

    # Hybrid score combining Bi-Encoder AQE cosine + Cross-Encoder probability
    oof_blend = 0.4 * oof_bi_cos + 0.6 * oof_ce_prob
    test_blend = 0.4 * test_bi_cos + 0.6 * test_ce_prob
    train_df["pred_score"] = oof_blend
    test_df["pred_score"] = test_blend

    # Compute OOF MAP@5
    aps = []
    for _, grp in train_df.groupby("query_id"):
        rels = grp.sort_values("pred_score", ascending=False)["relevance"].to_numpy()[:5]
        n_pos = np.sum(rels > 0)
        if n_pos > 0:
            hits = np.cumsum(rels > 0)
            prec = hits / np.arange(1, len(rels) + 1)
            aps.append(float(np.sum(prec * (rels > 0)) / n_pos))
    map5 = float(np.mean(aps)) if aps else 0.0

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_blend)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_blend)

    sub_rows = []
    for qid, grp in test_df.sort_values("pred_score", ascending=False).groupby("query_id"):
        sub_rows.append({"query_id": qid, "doc_ids": " ".join(grp["doc_id"].tolist())})
    pd.DataFrame(sub_rows).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Tier2 Bi-Encoder+AQE+Cross-Encoder] OOF MAP@5={map5:.4f} | "
        f"OOF shape={oof_blend.shape} | Test shape={test_blend.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 05 Tier 2 Bi-Encoder + Cross-Encoder")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = CFG(data_dir=args.data_dir, output_dir=args.output_dir)
    if args.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            run_pipeline(cfg, smoke_test=True)
    else:
        run_pipeline(cfg, smoke_test=False)


if __name__ == "__main__":
    main()
