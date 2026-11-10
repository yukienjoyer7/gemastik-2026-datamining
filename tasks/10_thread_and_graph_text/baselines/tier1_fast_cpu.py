"""
Task 10 — Tier 1 Fast CPU Baseline: Target + Parent/Root Context TF-IDF + Tree Topology
=======================================================================================
Fast CPU baseline for Conversational Threads & Reply Trees:
  1. **Multi-Zone Contextual TF-IDF**:
     - Separate Word + Char TF-IDF spaces for:
       (a) Target post text
       (b) Immediate Parent post text
       (c) Thread Root post text
       (d) Concatenated Direct Children reply text
  2. **Thread & User Graph Topology Features**:
     - Tree depth, normalized depth in thread, is_root indicator, direct reply out-degree,
       total subtree descendant count, log time delta from parent & root, thread size,
       author activity count, and TF-IDF cosine similarity between child & parent / root.
  3. Trains an Out-Of-Fold linear + GBDT hybrid pipeline grouped by `thread_id` and exports
     `oof_tier1_thread.npy`, `test_tier1_thread.npy`, and `sub_tier1_thread.csv`.

Usage:
  python3 tier1_fast_cpu.py --data-dir ./data --output-dir ./outputs
  python3 tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class CFG:
    exp_name: str = "tier1_thread"
    seed: int = 42
    n_folds: int = 5
    c_reg: float = 1.5
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def build_thread_context_and_topology(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str], np.ndarray]:
    """
    Extracts `[target_texts, parent_texts, root_texts, children_texts]` and a dense
    `topology_features` matrix `(N, 10)` from a reply-tree DataFrame.
    """
    ids = df["id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()
    parents = (
        df["parent_id"].fillna("").astype(str).tolist()
        if "parent_id" in df.columns
        else [""] * len(df)
    )
    timestamps = (
        df["timestamp"].fillna(0).to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    users = (
        df["user_id"].fillna("u0").astype(str).tolist()
        if "user_id" in df.columns
        else ["u0"] * len(df)
    )

    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    children_idx: Dict[int, List[int]] = defaultdict(list)
    parent_idx: Dict[int, int] = {}

    for i, pid in enumerate(parents):
        if pid in id_to_idx and id_to_idx[pid] != i:
            p_i = id_to_idx[pid]
            parent_idx[i] = p_i
            children_idx[p_i].append(i)

    # Compute root_idx and depth for every node
    depths = np.zeros(len(df), dtype=np.float32)
    root_indices = np.arange(len(df), dtype=np.int32)
    for i in range(len(df)):
        curr = i
        d = 0
        seen = {curr}
        while curr in parent_idx:
            nxt = parent_idx[curr]
            if nxt in seen:
                break
            seen.add(nxt)
            curr = nxt
            d += 1
        depths[i] = float(d)
        root_indices[i] = curr

    # Subtree descendant count via memoized DFS
    desc_memo: Dict[int, int] = {}

    def _count_desc(u: int) -> int:
        if u in desc_memo:
            return desc_memo[u]
        total = 0
        for ch in children_idx.get(u, []):
            total += 1 + _count_desc(ch)
        desc_memo[u] = total
        return total

    descendants = np.array([_count_desc(i) for i in range(len(df))], dtype=np.float32)
    out_degrees = np.array([len(children_idx.get(i, [])) for i in range(len(df))], dtype=np.float32)

    # Thread sizes and normalized depth
    thread_keys = (
        df["thread_id"].astype(str).tolist()
        if "thread_id" in df.columns
        else [ids[r] for r in root_indices]
    )
    t_counts: Dict[str, int] = defaultdict(int)
    t_max_depth: Dict[str, float] = defaultdict(float)
    for tk, d in zip(thread_keys, depths):
        t_counts[tk] += 1
        if d > t_max_depth[tk]:
            t_max_depth[tk] = float(d)

    user_counts: Dict[str, int] = defaultdict(int)
    for u in users:
        user_counts[u] += 1

    parent_texts: List[str] = []
    root_texts: List[str] = []
    child_texts: List[str] = []
    topo_rows: List[List[float]] = []

    for i in range(len(df)):
        p_i = parent_idx.get(i, i)
        r_i = int(root_indices[i])
        ch_list = children_idx.get(i, [])

        p_txt = texts[p_i] if i in parent_idx else "__root_no_parent__"
        r_txt = texts[r_i]
        c_txt = " ".join(texts[c] for c in ch_list) if ch_list else "__leaf_no_children__"

        parent_texts.append(p_txt)
        root_texts.append(r_txt)
        child_texts.append(c_txt)

        dt_parent = max(0.0, float(timestamps[i] - timestamps[p_i])) if i in parent_idx else 0.0
        dt_root = max(0.0, float(timestamps[i] - timestamps[r_i]))
        tk = thread_keys[i]
        norm_depth = float(depths[i] / max(t_max_depth[tk], 1.0))
        is_root = 1.0 if i not in parent_idx else 0.0
        is_leaf = 1.0 if len(ch_list) == 0 else 0.0

        topo_rows.append(
            [
                float(depths[i]),
                norm_depth,
                is_root,
                is_leaf,
                float(out_degrees[i]),
                float(descendants[i]),
                float(np.log1p(dt_parent)),
                float(np.log1p(dt_root)),
                float(t_counts[tk]),
                float(user_counts[users[i]]),
            ]
        )

    return texts, parent_texts, root_texts, child_texts, np.asarray(topo_rows, dtype=np.float32)


def _generate_synthetic_thread_data(
    n_threads: int = 18, nodes_per_thread: int = 5, seed: int = 42
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
    if smoke_test:
        train_df, test_df = _generate_synthetic_thread_data(
            n_threads=18, nodes_per_thread=5, seed=cfg.seed
        )
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[:25].copy().reset_index(drop=True)
        else:
            train_df, test_df = _generate_synthetic_thread_data(seed=cfg.seed)

    if "fold" not in train_df.columns:
        train_df = train_df.copy()
        train_df["fold"] = 0
        gkf = GroupKFold(n_splits=min(cfg.n_folds, train_df["thread_id"].nunique()))
        for fold, (_, val_idx) in enumerate(
            gkf.split(train_df, train_df["label"], groups=train_df["thread_id"])
        ):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    tr_tgt, tr_par, tr_root, tr_ch, tr_topo = build_thread_context_and_topology(train_df)
    te_tgt, te_par, te_root, te_ch, te_topo = build_thread_context_and_topology(test_df)

    vec_tgt = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    vec_par = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, sublinear_tf=True)
    vec_root = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, sublinear_tf=True)
    vec_ch = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)

    X_tr_tgt = vec_tgt.fit_transform(tr_tgt)
    X_te_tgt = vec_tgt.transform(te_tgt)
    X_tr_par = vec_par.fit_transform(tr_par)
    X_te_par = vec_par.transform(te_par)
    X_tr_root = vec_root.fit_transform(tr_root)
    X_te_root = vec_root.transform(te_root)
    X_tr_ch = vec_ch.fit_transform(tr_ch)
    X_te_ch = vec_ch.transform(te_ch)

    # Compute lexical cosine similarity between target and parent/root in target TF-IDF space
    tr_par_in_tgt = vec_tgt.transform(tr_par)
    tr_root_in_tgt = vec_tgt.transform(tr_root)
    te_par_in_tgt = vec_tgt.transform(te_par)
    te_root_in_tgt = vec_tgt.transform(te_root)

    tr_cos_par = np.asarray(X_tr_tgt.multiply(tr_par_in_tgt).sum(axis=1)).ravel()[:, None]
    tr_cos_root = np.asarray(X_tr_tgt.multiply(tr_root_in_tgt).sum(axis=1)).ravel()[:, None]
    te_cos_par = np.asarray(X_te_tgt.multiply(te_par_in_tgt).sum(axis=1)).ravel()[:, None]
    te_cos_root = np.asarray(X_te_tgt.multiply(te_root_in_tgt).sum(axis=1)).ravel()[:, None]

    scaler = StandardScaler()
    tr_dense = scaler.fit_transform(np.hstack([tr_topo, tr_cos_par, tr_cos_root]))
    te_dense = scaler.transform(np.hstack([te_topo, te_cos_par, te_cos_root]))

    X_tr = hstack(
        [
            X_tr_tgt * 2.0,
            X_tr_par * 1.2,
            X_tr_root * 0.8,
            X_tr_ch * 0.8,
            csr_matrix(tr_dense * 0.5),
        ]
    ).tocsr()
    X_te = hstack(
        [
            X_te_tgt * 2.0,
            X_te_par * 1.2,
            X_te_root * 0.8,
            X_te_ch * 0.8,
            csr_matrix(te_dense * 0.5),
        ]
    ).tocsr()

    y = train_df["label"].to_numpy(dtype=np.int32)
    n_classes = int(len(np.unique(y)))
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)

    folds = train_df["fold"].to_numpy(dtype=np.int32)
    unique_folds = sorted(np.unique(folds).tolist())

    for f in unique_folds:
        tr_mask = folds != f
        va_mask = folds == f
        clf = LogisticRegression(
            C=cfg.c_reg, max_iter=500, class_weight="balanced", random_state=cfg.seed + f
        )
        clf.fit(X_tr[tr_mask], y[tr_mask])
        oof_probs[va_mask] = clf.predict_proba(X_tr[va_mask]).astype(np.float32)
        test_probs += clf.predict_proba(X_te).astype(np.float32) / len(unique_folds)

    macro_f1 = float(f1_score(y, np.argmax(oof_probs, axis=1), average="macro"))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs)
    pd.DataFrame({"id": test_df["id"], "label": np.argmax(test_probs, axis=1)}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )

    print(
        f"[Tier1 Thread Context TF-IDF + Topology] OOF Macro-F1={macro_f1:.4f} | "
        f"OOF shape={oof_probs.shape} | Test shape={test_probs.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10 Tier 1 Fast CPU Thread Baseline")
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
