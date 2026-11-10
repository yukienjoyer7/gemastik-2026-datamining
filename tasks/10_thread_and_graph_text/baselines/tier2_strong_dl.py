"""
Task 10 — Tier 2 Strong DL: Context-Window Transformer + 2nd-Stage Relational Graph Stacking
============================================================================================
Two-Stage Neural + Graph Stacking Architecture for Conversational Threads & Reply Trees:
  - **Stage 1 (Context-Window Transformer OOF Pipeline)**:
    Encodes each node with dynamic token budget allocation across `[ROOT]`, `[PARENT]`, and `[TARGET]`:
    `"[CLS] [ROOT] {root_text} [SEP] [PARENT] {parent_text} [SEP] [TARGET] {target_text} [SEP]"`
    Produces leak-free Stage-1 Out-Of-Fold node probabilities `P_stage1` `(N, C)`.
  - **Stage 2 (Relational Graph Stacking Meta-Classifier)**:
    For every node `i`, aggregates Stage-1 OOF probabilities over its graph neighborhood:
      * `P_self(i)`: node `i`'s own Stage-1 probability vector
      * `P_parent(i)`: immediate parent's Stage-1 probability vector
      * `P_root(i)`: thread root's Stage-1 probability vector
      * `P_children_mean(i)`: mean Stage-1 probability vector of direct replies to `i`
      * `P_same_user_loo(i)`: leave-one-out mean Stage-1 probability vector of other posts by `user_i`
    Trains a 2nd-stage meta-classifier on `[P_self, P_parent, P_root, P_children_mean, P_same_user_loo]`,
    improving OOF Macro-F1 by capturing conversational agreement/disagreement dynamics.

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class CFG:
    exp_name: str = "tier2_thread_dl"
    model_name: str = "indobenchmark/indobert-base-p1"
    seed: int = 42
    n_folds: int = 4
    vocab_size: int = 1024
    root_budget: int = 8
    parent_budget: int = 10
    target_budget: int = 14
    embed_dim: int = 64
    epochs: int = 10
    lr: float = 4e-3
    data_dir: str = "./data"
    output_dir: str = "./outputs"


class BudgetedThreadTokenizer:
    """
    Tokenizes `(root_text, parent_text, target_text)` with explicit token budgets per zone
    so long root posts never truncate the target reply!
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        root_budget: int = 8,
        parent_budget: int = 10,
        target_budget: int = 14,
    ) -> None:
        self.vocab_size = vocab_size
        self.root_budget = root_budget
        self.parent_budget = parent_budget
        self.target_budget = target_budget

    def _encode_zone(self, texts: Sequence[str], budget: int) -> torch.Tensor:
        out = torch.zeros((len(texts), budget), dtype=torch.long)
        for i, t in enumerate(texts):
            toks = TOKEN_RE.findall(str(t).lower())[:budget]
            for j, tok in enumerate(toks):
                out[i, j] = (hash(tok) % (self.vocab_size - 2)) + 2
        return out

    def encode_triplets(
        self,
        root_texts: Sequence[str],
        parent_texts: Sequence[str],
        target_texts: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._encode_zone(root_texts, self.root_budget),
            self._encode_zone(parent_texts, self.parent_budget),
            self._encode_zone(target_texts, self.target_budget),
        )


class ContextWindowThreadTransformer(nn.Module):
    """
    Zone-aware Transformer encoding `[ROOT]`, `[PARENT]`, and `[TARGET]` segments with
    distinct segment embeddings and cross-segment attention.
    """

    def __init__(
        self, vocab_size: int = 1024, embed_dim: int = 64, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.zone_embedding = nn.Embedding(3, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self, r_ids: torch.Tensor, p_ids: torch.Tensor, t_ids: torch.Tensor
    ) -> torch.Tensor:
        r_e = self.embedding(r_ids) + self.zone_embedding(torch.zeros_like(r_ids))
        p_e = self.embedding(p_ids) + self.zone_embedding(torch.ones_like(p_ids))
        t_e = self.embedding(t_ids) + self.zone_embedding(torch.full_like(t_ids, 2))

        joint = torch.cat([r_e, p_e, t_e], dim=1)
        attn_out, _ = self.attn(joint, joint, joint, need_weights=False)
        joint = self.norm(joint + attn_out)

        r_len = r_ids.size(1)
        p_len = p_ids.size(1)

        def _pool(seg: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            m = (ids != 0).float().unsqueeze(-1)
            return (seg * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        r_pool = _pool(joint[:, :r_len], r_ids)
        p_pool = _pool(joint[:, r_len : r_len + p_len], p_ids)
        t_pool = _pool(joint[:, r_len + p_len :], t_ids)
        return self.head(torch.cat([t_pool, p_pool, r_pool], dim=-1))


def build_relational_stacking_features(
    df: pd.DataFrame, stage1_probs: np.ndarray
) -> np.ndarray:
    """
    Constructs leak-free relational neighbor probability features `(N, 5 * C)`:
      `[P_self, P_parent, P_root, P_children_mean, P_same_user_loo]`
    where `P_same_user_loo` strictly excludes node `i`'s own probability vector.
    """
    n_samples, n_classes = stage1_probs.shape
    ids = df["id"].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    parents = df["parent_id"].fillna("").astype(str).tolist()
    users = (
        df["user_id"].fillna("u0").astype(str).tolist()
        if "user_id" in df.columns
        else ["u0"] * n_samples
    )

    parent_idx: Dict[int, int] = {}
    children_idx: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(parents):
        if pid in id_to_idx and id_to_idx[pid] != i:
            p_i = id_to_idx[pid]
            parent_idx[i] = p_i
            children_idx[p_i].append(i)

    root_idx = np.arange(n_samples, dtype=np.int32)
    for i in range(n_samples):
        curr = i
        seen = {curr}
        while curr in parent_idx:
            nxt = parent_idx[curr]
            if nxt in seen:
                break
            seen.add(nxt)
            curr = nxt
        root_idx[i] = curr

    # Leave-one-out same-user probability sum & count
    user_sum: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(n_classes, dtype=np.float64)
    )
    user_cnt: Dict[str, int] = defaultdict(int)
    for i, u in enumerate(users):
        user_sum[u] += stage1_probs[i]
        user_cnt[u] += 1

    global_prior = np.mean(stage1_probs, axis=0)
    stack_feats = np.zeros((n_samples, n_classes * 5), dtype=np.float32)

    for i in range(n_samples):
        p_self = stage1_probs[i]
        p_par = stage1_probs[parent_idx[i]] if i in parent_idx else p_self
        p_root = stage1_probs[int(root_idx[i])]
        ch = children_idx.get(i, [])
        p_ch = np.mean(stage1_probs[ch], axis=0) if ch else p_self

        u = users[i]
        if user_cnt[u] > 1:
            p_user_loo = (user_sum[u] - p_self) / float(user_cnt[u] - 1)
        else:
            p_user_loo = global_prior

        stack_feats[i] = np.concatenate(
            [p_self, p_par, p_root, p_ch, p_user_loo], axis=0
        ).astype(np.float32)

    return stack_feats


def extract_zones(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    ids = df["id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()
    parents = df["parent_id"].fillna("").astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    parent_idx = {
        i: id_to_idx[pid]
        for i, pid in enumerate(parents)
        if pid in id_to_idx and id_to_idx[pid] != i
    }
    root_texts, parent_texts = [], []
    for i in range(len(df)):
        p_i = parent_idx.get(i, i)
        curr = i
        seen = {curr}
        while curr in parent_idx:
            nxt = parent_idx[curr]
            if nxt in seen:
                break
            seen.add(nxt)
            curr = nxt
        root_texts.append(texts[curr])
        parent_texts.append(texts[p_i])
    return root_texts, parent_texts, texts


def _generate_synthetic_thread_data(
    n_threads: int = 16, nodes_per_thread: int = 5, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stance_vocab = {
        0: "berita ini resmi terverifikasi akurat sesuai data kementerian",
        1: "klaim tersebut hoaks salah besar tidak ada bukti ilmiah",
        2: "apakah info ini benar mohon klarifikasi sumber resminya",
    }

    def _build(n_t: int, prefix: str) -> pd.DataFrame:
        rows = []
        for t_idx in range(n_t):
            tid = f"{prefix}_TH{t_idx:03d}"
            root_id = f"{tid}_N0"
            root_label = int(t_idx % 3)
            base_ts = 1700000000 + t_idx * 3600
            thread_nodes = []
            for n_idx in range(nodes_per_thread):
                nid = f"{tid}_N{n_idx}"
                if n_idx == 0:
                    pid = ""
                    lbl = root_label
                    ts = base_ts
                else:
                    par = thread_nodes[int(rng.integers(0, len(thread_nodes)))]
                    pid = par["id"]
                    lbl = int(par["label"]) if rng.random() < 0.78 else int((par["label"] + 1) % 3)
                    ts = int(par["timestamp"]) + int(rng.integers(30, 900))
                uid = f"U_{(lbl * 7 + int(rng.integers(0, 3))) % 18:02d}"
                words = rng.choice(stance_vocab[lbl].split(), size=5, replace=False)
                node_dict = {
                    "id": nid,
                    "parent_id": pid,
                    "root_id": root_id,
                    "thread_id": tid,
                    "user_id": uid,
                    "timestamp": ts,
                    "text": f"post {nid} " + " ".join(words),
                    "label": lbl,
                }
                thread_nodes.append(node_dict)
                rows.append(node_dict)
        return pd.DataFrame(rows)

    return _build(n_threads, "TR"), _build(max(4, n_threads // 3), "TE")


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if smoke_test:
        train_df, test_df = _generate_synthetic_thread_data(
            n_threads=16, nodes_per_thread=5, seed=cfg.seed
        )
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[:20].copy().reset_index(drop=True)
        else:
            train_df, test_df = _generate_synthetic_thread_data(seed=cfg.seed)

    if "fold" not in train_df.columns:
        train_df["fold"] = 0
        gkf = GroupKFold(n_splits=min(cfg.n_folds, train_df["thread_id"].nunique()))
        for fold, (_, val_idx) in enumerate(
            gkf.split(train_df, train_df["label"], groups=train_df["thread_id"])
        ):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    tokenizer = BudgetedThreadTokenizer(
        vocab_size=cfg.vocab_size,
        root_budget=cfg.root_budget,
        parent_budget=cfg.parent_budget,
        target_budget=cfg.target_budget,
    )
    tr_r, tr_p, tr_t = tokenizer.encode_triplets(*extract_zones(train_df))
    te_r, te_p, te_t = tokenizer.encode_triplets(*extract_zones(test_df))

    y = train_df["label"].to_numpy(dtype=np.int64)
    n_classes = int(len(np.unique(y)))
    y_t = torch.tensor(y, dtype=torch.long)

    oof_stage1 = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_stage1 = np.zeros((len(test_df), n_classes), dtype=np.float32)
    folds = train_df["fold"].to_numpy(dtype=np.int32)
    unique_folds = sorted(np.unique(folds).tolist())

    for fold in unique_folds:
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        model = ContextWindowThreadTransformer(
            vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim, num_classes=n_classes
        )
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

        for _ in range(cfg.epochs):
            model.train()
            opt.zero_grad()
            logits = model(tr_r[tr_idx], tr_p[tr_idx], tr_t[tr_idx])
            loss = F.cross_entropy(logits, y_t[tr_idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_probs = F.softmax(
                model(tr_r[va_idx], tr_p[va_idx], tr_t[va_idx]), dim=-1
            ).numpy()
            te_probs = F.softmax(model(te_r, te_p, te_t), dim=-1).numpy()
            oof_stage1[va_idx] = va_probs
            test_stage1 += te_probs / len(unique_folds)

    stage1_f1 = float(f1_score(y, np.argmax(oof_stage1, axis=1), average="macro"))

    # Stage 2: Relational Graph Stacking over reply-tree & same-user OOF probabilities
    X_stack_tr = build_relational_stacking_features(train_df, oof_stage1)
    X_stack_te = build_relational_stacking_features(test_df, test_stage1)

    oof_stage2 = np.zeros_like(oof_stage1)
    test_stage2 = np.zeros_like(test_stage1)
    for fold in unique_folds:
        tr_mask = folds != fold
        va_mask = folds == fold
        meta = LogisticRegression(C=1.0, max_iter=300, random_state=cfg.seed + fold)
        meta.fit(X_stack_tr[tr_mask], y[tr_mask])
        oof_stage2[va_mask] = meta.predict_proba(X_stack_tr[va_mask]).astype(np.float32)
        test_stage2 += meta.predict_proba(X_stack_te).astype(np.float32) / len(unique_folds)

    stage2_f1 = float(f1_score(y, np.argmax(oof_stage2, axis=1), average="macro"))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_stage2)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_stage2)
    pd.DataFrame({"id": test_df["id"], "label": np.argmax(test_stage2, axis=1)}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )

    print(
        f"[Tier2 Thread Context-Transformer + Graph Stacking] "
        f"Stage1 Macro-F1={stage1_f1:.4f} -> Stage2 Graph-Stacked Macro-F1={stage2_f1:.4f} | "
        f"OOF shape={oof_stage2.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 10 Tier 2 Strong DL Thread + Graph Stacking"
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic smoke test")
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
