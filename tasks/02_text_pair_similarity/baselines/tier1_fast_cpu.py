#!/usr/bin/env python3
"""
Task 02 — Text Pair Similarity: Tier 1 Fast CPU Feature-Engineered GBDT Baseline
================================================================================
Standalone Out-Of-Fold (OOF) feature engineering + GBDT baseline combining:
  1. Lexical & Surface Pair Similarity Features:
     - Word (1, 2) TF-IDF row-wise cosine similarity
     - Char-wb (3, 5) TF-IDF row-wise cosine similarity
     - Word 1-gram & 2-gram Jaccard overlap + Sorensen-Dice coefficient
     - Token overlap ratios (`|A ∩ B| / |A|`, `|A ∩ B| / |B|`)
     - Character & word length differences and ratios
     - Normalized character bigram Dice / fast edit similarity (`difflib.SequenceMatcher`)
     - Exact prefix/suffix match indicators
  2. "Magic" Pair-Graph Topological Features (transductive on `train + test` graph):
     - Node degrees `deg(A)`, `deg(B)`, `max(deg)`, `min(deg)`, `|deg(A) - deg(B)|`, `deg_ratio`
     - Shared neighbors count `|N(A) ∩ N(B)|`, neighbor Jaccard overlap
     - Adamic-Adar index `sum_{w in N(A) ∩ N(B)} 1 / log(deg(w))`
     - Connected component size & log component size
     - Power-iteration PageRank (`PR(A)`, `PR(B)`, `PR(A)+PR(B)`) & 2-hop Personalized PageRank affinity
  3. Out-Of-Fold GBDT (`LightGBM` if installed, else `HistGradientBoosting`) supporting both
     binary classification and continuous similarity regression.
  4. Standardized OOF & prediction contract:
     Saves `oof_tier1_pair.npy`, `test_tier1_pair.npy`, and `sub_tier1_pair.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import difflib
import math
import pathlib
import re
import site
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, log_loss, roc_auc_score

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class CFG:
    """Configuration for Tier 1 Pair Similarity Feature + GBDT Baseline."""

    exp_name: str = "tier1_pair"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_a_col: str = "text_a"
    text_b_col: str = "text_b"
    target_col: str = "label"
    fold_col: str = "fold"
    task_type: str = "auto"  # "binary" | "regression" | "auto"

    max_word_features: int = 50000
    max_char_features: int = 60000
    pagerank_damping: float = 0.85
    pagerank_iters: int = 20

    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Lexical Similarity & Magic Pair-Graph Feature Engineering
# =============================================================================


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


def compute_lexical_pair_features(
    texts_a: Sequence[str], texts_b: Sequence[str]
) -> np.ndarray:
    """
    Computes 13 lexical overlap, n-gram Jaccard/Dice, length ratio, and edit similarity
    features per `(text_a, text_b)` pair.
    """
    n = len(texts_a)
    feats = np.zeros((n, 13), dtype=np.float32)

    for i, (ra, rb) in enumerate(zip(texts_a, texts_b)):
        sa = str(ra or "").strip()
        sb = str(rb or "").strip()
        la_c, lb_c = len(sa), len(sb)

        wa = _tokenize_words(sa)
        wb = _tokenize_words(sb)
        la_w, lb_w = len(wa), len(wb)

        set_a, set_b = set(wa), set(wb)
        inter_1 = len(set_a & set_b)
        union_1 = len(set_a | set_b)
        jaccard_1 = inter_1 / max(union_1, 1)
        dice_1 = (2.0 * inter_1) / max(len(set_a) + len(set_b), 1)
        overlap_a = inter_1 / max(len(set_a), 1)
        overlap_b = inter_1 / max(len(set_b), 1)

        # 2-gram Jaccard
        bg_a = set(zip(wa[:-1], wa[1:])) if la_w >= 2 else set()
        bg_b = set(zip(wb[:-1], wb[1:])) if lb_w >= 2 else set()
        inter_2 = len(bg_a & bg_b)
        union_2 = len(bg_a | bg_b)
        jaccard_2 = inter_2 / max(union_2, 1) if union_2 > 0 else 0.0

        # Character 3-gram Dice
        s_low_a, s_low_b = sa.lower(), sb.lower()
        cg_a = {s_low_a[j : j + 3] for j in range(max(0, len(s_low_a) - 2))}
        cg_b = {s_low_b[j : j + 3] for j in range(max(0, len(s_low_b) - 2))}
        c_dice = (2.0 * len(cg_a & cg_b)) / max(len(cg_a) + len(cg_b), 1)

        # Normalized edit similarity (Levenshtein-like longest contiguous matching blocks)
        edit_sim = difflib.SequenceMatcher(None, s_low_a[:256], s_low_b[:256]).quick_ratio()

        char_diff = abs(la_c - lb_c)
        char_ratio = min(la_c, lb_c) / max(la_c, lb_c, 1)
        word_diff = abs(la_w - lb_w)
        word_ratio = min(la_w, lb_w) / max(la_w, lb_w, 1)

        first_match = 1.0 if (la_w > 0 and lb_w > 0 and wa[0] == wb[0]) else 0.0
        last_match = 1.0 if (la_w > 0 and lb_w > 0 and wa[-1] == wb[-1]) else 0.0

        feats[i, 0] = jaccard_1
        feats[i, 1] = dice_1
        feats[i, 2] = overlap_a
        feats[i, 3] = overlap_b
        feats[i, 4] = jaccard_2
        feats[i, 5] = c_dice
        feats[i, 6] = edit_sim
        feats[i, 7] = math.log1p(char_diff)
        feats[i, 8] = char_ratio
        feats[i, 9] = math.log1p(word_diff)
        feats[i, 10] = word_ratio
        feats[i, 11] = first_match
        feats[i, 12] = last_match

    return feats


def compute_magic_graph_features(
    all_texts_a: Sequence[str],
    all_texts_b: Sequence[str],
    damping: float = 0.85,
    n_iters: int = 20,
) -> np.ndarray:
    """
    Computes 15 transductive pair-graph features over all `(text_a, text_b)` edges:
      - `deg(A)`, `deg(B)`, `max_deg`, `min_deg`, `|deg(A)-deg(B)|`, `deg_ratio`
      - Shared neighbors `|N(A) ∩ N(B)|`, neighbor Jaccard, Adamic-Adar index
      - Connected component size & log1p(component_size)
      - PageRank `PR(A)`, `PR(B)`, `PR(A)+PR(B)`, and 2-hop Personalized PageRank affinity
    """
    node_to_id: Dict[str, int] = {}
    for t in list(all_texts_a) + list(all_texts_b):
        s = str(t)
        if s not in node_to_id:
            node_to_id[s] = len(node_to_id)

    n_nodes = len(node_to_id)
    neighbors: List[set] = [set() for _ in range(n_nodes)]
    for a, b in zip(all_texts_a, all_texts_b):
        u, v = node_to_id[str(a)], node_to_id[str(b)]
        if u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)

    degrees = np.array([len(nb) for nb in neighbors], dtype=np.float64)

    # Connected components via BFS
    visited = np.zeros(n_nodes, dtype=bool)
    comp_size = np.ones(n_nodes, dtype=np.float64)
    for start in range(n_nodes):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            for nxt in neighbors[curr]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        c_sz = float(len(queue))
        for member in queue:
            comp_size[member] = c_sz

    # Power-iteration PageRank
    pr = np.full(n_nodes, 1.0 / max(n_nodes, 1), dtype=np.float64)
    teleport = (1.0 - damping) / max(n_nodes, 1)
    for _ in range(n_iters):
        new_pr = np.full(n_nodes, teleport, dtype=np.float64)
        new_pr += pr[degrees == 0].sum() * damping / max(n_nodes, 1)
        for u in range(n_nodes):
            if degrees[u] > 0:
                share = damping * pr[u] / degrees[u]
                for v in neighbors[u]:
                    new_pr[v] += share
        pr = new_pr
    pr = pr * n_nodes

    feats = np.zeros((len(all_texts_a), 15), dtype=np.float32)
    for i, (ra, rb) in enumerate(zip(all_texts_a, all_texts_b)):
        u, v = node_to_id[str(ra)], node_to_id[str(rb)]
        nb_u, nb_v = neighbors[u], neighbors[v]
        du, dv = float(degrees[u]), float(degrees[v])
        max_d, min_d = max(du, dv), min(du, dv)
        common = nb_u & nb_v
        shared = float(len(common))
        union_nb = float(len(nb_u | nb_v))
        adamic = sum(1.0 / math.log(max(float(degrees[w]), 2.0)) for w in common)
        csz = float(max(comp_size[u], comp_size[v]))
        pru, prv = float(pr[u]), float(pr[v])
        two_hop = sum(1.0 / max(float(degrees[w]), 1.0) for w in common)
        ppr_aff = (damping * (1.0 if v in nb_u else 0.0) + (damping**2) * two_hop) / max(du, 1.0)

        feats[i] = [
            du,
            dv,
            max_d,
            min_d,
            abs(du - dv),
            min_d / max(max_d, 1.0),
            shared,
            shared / max(union_nb, 1.0),
            adamic,
            csz,
            math.log1p(csz),
            pru,
            prv,
            pru + prv,
            ppr_aff,
        ]

    return feats


def compute_tfidf_cosine_features(
    fit_texts: Sequence[str],
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    cfg: CFG,
) -> np.ndarray:
    """Computes Word (1, 2) TF-IDF and Char-wb (3, 5) TF-IDF row-wise cosine similarities."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=cfg.max_word_features,
        sublinear_tf=True,
        norm="l2",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=cfg.max_char_features,
        sublinear_tf=True,
        norm="l2",
    )
    word_vec.fit(fit_texts)
    char_vec.fit(fit_texts)

    wa = word_vec.transform(texts_a)
    wb = word_vec.transform(texts_b)
    word_cos = np.asarray(wa.multiply(wb).sum(axis=1)).ravel().astype(np.float32)

    ca = char_vec.transform(texts_a)
    cb = char_vec.transform(texts_b)
    char_cos = np.asarray(ca.multiply(cb).sum(axis=1)).ravel().astype(np.float32)

    return np.column_stack([word_cos, char_cos, word_cos * char_cos, np.abs(word_cos - char_cos)])


# =============================================================================
# 2. Out-Of-Fold GBDT Pipeline
# =============================================================================


def infer_task_type(y: np.ndarray, requested: str) -> str:
    if requested in ("binary", "regression"):
        return requested
    uniq = np.unique(y)
    if len(uniq) <= 2 and set(np.round(uniq, 4)).issubset({0.0, 1.0}):
        return "binary"
    return "regression"


def fit_predict_gbdt(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    task_type: str,
    cfg: CFG,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_LIGHTGBM:
        if task_type == "binary":
            model = lgb.LGBMClassifier(
                n_estimators=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.max_leaf_nodes,
                random_state=seed,
                verbosity=-1,
            )
            model.fit(X_tr, y_tr.astype(int))
            return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        else:
            model = lgb.LGBMRegressor(
                n_estimators=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.max_leaf_nodes,
                random_state=seed,
                verbosity=-1,
            )
            model.fit(X_tr, y_tr)
            return model.predict(X_va), model.predict(X_te)

    min_leaf = 5 if cfg.smoke_test else 20
    if task_type == "binary":
        model = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )
        model.fit(X_tr, y_tr.astype(int))
        return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
    else:
        model = HistGradientBoostingRegressor(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=min_leaf,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        return model.predict(X_va), model.predict(X_te)


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray]:
    n_tr = len(train_df)
    all_a = train_df[cfg.text_a_col].fillna("").astype(str).tolist() + test_df[cfg.text_a_col].fillna("").astype(str).tolist()
    all_b = train_df[cfg.text_b_col].fillna("").astype(str).tolist() + test_df[cfg.text_b_col].fillna("").astype(str).tolist()

    lex_all = compute_lexical_pair_features(all_a, all_b)
    graph_all = compute_magic_graph_features(
        all_a, all_b, damping=cfg.pagerank_damping, n_iters=cfg.pagerank_iters
    )
    base_tr = np.hstack([lex_all[:n_tr], graph_all[:n_tr]])
    base_te = np.hstack([lex_all[n_tr:], graph_all[n_tr:]])

    y = train_df[cfg.target_col].values.astype(np.float64)
    task_type = infer_task_type(y, cfg.task_type)
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))

    oof_preds = np.zeros(n_tr, dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)

    tr_a = train_df[cfg.text_a_col].fillna("").astype(str).tolist()
    tr_b = train_df[cfg.text_b_col].fillna("").astype(str).tolist()
    te_a = test_df[cfg.text_a_col].fillna("").astype(str).tolist()
    te_b = test_df[cfg.text_b_col].fillna("").astype(str).tolist()

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        fit_corpus = [tr_a[i] for i in tr_idx] + [tr_b[i] for i in tr_idx]
        tfidf_tr = compute_tfidf_cosine_features(
            fit_corpus, [tr_a[i] for i in tr_idx], [tr_b[i] for i in tr_idx], cfg
        )
        tfidf_va = compute_tfidf_cosine_features(
            fit_corpus, [tr_a[i] for i in va_idx], [tr_b[i] for i in va_idx], cfg
        )
        tfidf_te = compute_tfidf_cosine_features(fit_corpus, te_a, te_b, cfg)

        X_tr = np.hstack([base_tr[tr_idx], tfidf_tr])
        X_va = np.hstack([base_tr[va_idx], tfidf_va])
        X_te = np.hstack([base_te, tfidf_te])

        p_va, p_te = fit_predict_gbdt(
            X_tr, y[tr_idx], X_va, X_te, task_type=task_type, cfg=cfg, seed=cfg.seed + int(f)
        )
        oof_preds[va_idx] = p_va
        test_preds += p_te / len(unique_folds)
        print(f"  [Fold {f}] val_size={len(va_idx)} | total_features={X_tr.shape[1]}")

    if task_type == "binary":
        auc = float(roc_auc_score(y.astype(int), oof_preds))
        f1 = float(f1_score(y.astype(int), (oof_preds >= 0.5).astype(int)))
        ll = float(log_loss(y.astype(int), np.clip(oof_preds, 1e-6, 1 - 1e-6)))
        print(f"[OOF SUMMARY (Binary)] ROC-AUC={auc:.4f} | F1={f1:.4f} | LogLoss={ll:.4f}")
    else:
        pr = float(pearsonr(y, oof_preds)[0])
        sr = float(spearmanr(y, oof_preds).statistic)
        print(f"[OOF SUMMARY (Regression)] Pearson r={pr:.4f} | Spearman rho={sr:.4f}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_preds.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_preds.astype(np.float32))
    sub_df = pd.DataFrame({cfg.id_col: test_df[cfg.id_col].values, cfg.target_col: test_preds})
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    return oof_preds, test_preds


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 02 Tier 1 Pair Feature + GBDT Baseline")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--task-type", type=str, default=CFG.task_type)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        task_type=args.task_type,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        cfg.max_iter = 40
        rng = np.random.default_rng(cfg.seed)
        clusters = [
            [f"how to train neural network transformer model {k} with AdamW optimizer" for k in range(6)],
            [f"indonesian economy bank central inflation rate policy fiscal {k}" for k in range(6)],
            [f"champions league football striker goal penalty match {k}" for k in range(6)],
        ]
        train_rows, test_rows = [], []
        for i in range(120):
            is_dup = i % 2 == 0
            c1 = i % 3
            c2 = c1 if is_dup else (c1 + 1) % 3
            a = clusters[c1][int(rng.integers(0, 6))]
            b = clusters[c2][int(rng.integers(0, 6))]
            train_rows.append(
                {"id": f"tr_{i}", "text_a": a, "text_b": b, "label": int(is_dup), "fold": i % 5}
            )
        for i in range(30):
            is_dup = i % 2 == 0
            c1 = i % 3
            c2 = c1 if is_dup else (c1 + 1) % 3
            test_rows.append(
                {
                    "id": f"te_{i}",
                    "text_a": clusters[c1][int(rng.integers(0, 6))],
                    "text_b": clusters[c2][int(rng.integers(0, 6))],
                }
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 02 Tier 1 Pair Similarity + Magic Graph GBDT Baseline...")
            oof, test_preds = run_oof_pipeline(pd.DataFrame(train_rows), pd.DataFrame(test_rows), cfg)
            assert oof.shape == (120,)
            assert test_preds.shape == (30,)
            print("[SUCCESS] Task 02 Tier 1 smoke test passed.")
    else:
        run_oof_pipeline(load_table(cfg.train_path), load_table(cfg.test_path), cfg)


if __name__ == "__main__":
    main()
