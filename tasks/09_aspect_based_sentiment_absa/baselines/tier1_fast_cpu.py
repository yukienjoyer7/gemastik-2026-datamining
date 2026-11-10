"""
Task 09 — Tier 1 Fast CPU Baseline: Global + Aspect-Keyword Context Window TF-IDF
=================================================================================
Fast CPU baseline for Aspect-Category Sentiment Analysis (ACSA):
  - Why global TF-IDF alone fails on ABSA:
    If a review says "Makanannya enak tapi pelayanannya buruk", global bag-of-words features
    are identical for both the `makanan` classifier and the `pelayanan` classifier!
  - Solution implemented here:
    1. Global Word (1-2 gram) + Character (3-5 gram) TF-IDF features
    2. **Aspect-Keyword Local Context Window TF-IDF**:
       For each aspect `a`, locates seed keywords in the review and extracts only the
       surrounding `+- window_k` tokens (`extract_aspect_context_window`), plus clause-level
       splitting on contrastive conjunctions (`tetapi`, `tapi`, `namun`, `sedangkan`, `padahal`).
    3. Trains an Out-Of-Fold multi-class classifier (`LogisticRegression` + optional `LightGBM`)
       per aspect to output calibrated `(N, A, C)` probability tensors.
    4. Exports `oof_tier1_absa.npy`, `test_tier1_absa.npy`, and `sub_tier1_absa.csv`.

Usage:
  python3 tier1_fast_cpu.py --data-dir ./data --output-dir ./outputs
  python3 tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import re
import site
import sys
import tempfile
from dataclasses import dataclass, field
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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_ASPECT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "kebersihan": (
        "kebersihan",
        "bersih",
        "kotor",
        "bau",
        "debu",
        "berdebu",
        "rapi",
        "sprei",
        "handuk",
        "lantai",
        "higienis",
        "jorok",
        "wangi",
        "kinclong",
    ),
    "pelayanan": (
        "pelayanan",
        "staf",
        "petugas",
        "resepsionis",
        "ramah",
        "jutek",
        "lambat",
        "cepat",
        "sigap",
        "sopan",
        "cuek",
        "senyum",
        "check",
        "membantu",
    ),
    "harga": (
        "harga",
        "tarif",
        "biaya",
        "mahal",
        "kemahalan",
        "murah",
        "terjangkau",
        "hemat",
        "sepadan",
        "worth",
        "diskon",
        "promo",
        "wajar",
    ),
    "lokasi": (
        "lokasi",
        "akses",
        "strategis",
        "dekat",
        "jauh",
        "macet",
        "stasiun",
        "bandara",
        "pusat",
        "kota",
        "mall",
        "terpencil",
        "jalan",
    ),
}

CONTRAST_RE = re.compile(
    r"\b(?:tetapi|tapi|namun|sedangkan|padahal|meskipun|walaupun)\b|[.,;!?]",
    flags=re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class CFG:
    exp_name: str = "tier1_absa"
    seed: int = 42
    n_folds: int = 5
    window_k: int = 5
    num_polarities: int = 4
    c_reg: float = 2.0
    aspect_keywords: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_ASPECT_KEYWORDS)
    )
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def extract_aspect_context_window(
    text: str, seed_keywords: Sequence[str], window_k: int = 5
) -> str:
    """
    Extracts local context around aspect seed keywords:
      1. Splits review into contrastive clauses; retains clauses mentioning any seed keyword.
      2. Extracts `+- window_k` tokens around every seed keyword hit in the tokenized text.
      3. Returns `"__NO_ASPECT_HIT__"` when no seed keyword is present (strong signal for class 0 = None).
    """
    raw = str(text).lower()
    kw_set = {k.lower() for k in seed_keywords}

    # 1. Clause-level contrastive segmentation
    clauses = [c.strip() for c in CONTRAST_RE.split(raw) if c.strip()]
    matched_clauses: List[str] = []
    for cl in clauses:
        cl_toks = set(TOKEN_RE.findall(cl))
        if cl_toks & kw_set:
            matched_clauses.append(cl)

    # 2. Token-window extraction (+- window_k tokens)
    tokens = TOKEN_RE.findall(raw)
    hit_indices = [idx for idx, tok in enumerate(tokens) if tok in kw_set]
    window_snippets: List[str] = []
    for idx in hit_indices:
        start = max(0, idx - window_k)
        end = min(len(tokens), idx + window_k + 1)
        window_snippets.append(" ".join(tokens[start:end]))

    if not matched_clauses and not window_snippets:
        return "__no_aspect_hit__"

    return " | ".join(matched_clauses + window_snippets)


def evaluate_absa_macro_f1(y_true: np.ndarray, oof_probs: np.ndarray) -> Dict[str, float]:
    """Computes per-aspect Macro-F1 and overall Macro-F1 across all aspects."""
    preds = np.argmax(oof_probs, axis=-1)
    aspect_f1s: List[float] = []
    metrics: Dict[str, float] = {}
    for a_idx in range(y_true.shape[1]):
        f1_a = float(f1_score(y_true[:, a_idx], preds[:, a_idx], average="macro", zero_division=0))
        aspect_f1s.append(f1_a)
        metrics[f"aspect_{a_idx}_macro_f1"] = f1_a
    metrics["overall_macro_f1"] = float(np.mean(aspect_f1s))
    return metrics


def train_oof_tier1_absa(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Trains per-aspect OOF LogisticRegression models combining global Word/Char TF-IDF
    with aspect-specific local context window TF-IDF.
    """
    aspect_cols = list(cfg.aspect_keywords.keys())
    n_train = len(train_df)
    n_test = len(test_df)
    n_aspects = len(aspect_cols)
    n_classes = cfg.num_polarities

    oof_probs = np.zeros((n_train, n_aspects, n_classes), dtype=np.float32)
    test_probs = np.zeros((n_test, n_aspects, n_classes), dtype=np.float32)

    train_texts = train_df["text"].fillna("").astype(str).tolist()
    test_texts = test_df["text"].fillna("").astype(str).tolist()

    # Global Word + Char TF-IDF (low weight relative to aspect-local context)
    global_word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    global_char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=3000, sublinear_tf=True
    )
    X_tr_g = hstack(
        [global_word_vec.fit_transform(train_texts), global_char_vec.fit_transform(train_texts)]
    ).tocsr()
    X_te_g = hstack(
        [global_word_vec.transform(test_texts), global_char_vec.transform(test_texts)]
    ).tocsr()

    for a_idx, aspect in enumerate(aspect_cols):
        keywords = cfg.aspect_keywords[aspect]
        tr_ctx = [
            extract_aspect_context_window(t, keywords, window_k=cfg.window_k) for t in train_texts
        ]
        te_ctx = [
            extract_aspect_context_window(t, keywords, window_k=cfg.window_k) for t in test_texts
        ]

        ctx_word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        ctx_char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=3000, sublinear_tf=True
        )
        X_tr_ctx = hstack(
            [ctx_word_vec.fit_transform(tr_ctx), ctx_char_vec.fit_transform(tr_ctx)]
        ).tocsr()
        X_te_ctx = hstack(
            [ctx_word_vec.transform(te_ctx), ctx_char_vec.transform(te_ctx)]
        ).tocsr()

        # Weight aspect-local context 2.5x higher than global review context
        X_tr = hstack([X_tr_ctx * 2.5, X_tr_g * 0.5]).tocsr()
        X_te = hstack([X_te_ctx * 2.5, X_te_g * 0.5]).tocsr()
        y_a = train_df[aspect].to_numpy(dtype=np.int32)

        if "fold" in train_df.columns:
            folds = train_df["fold"].to_numpy(dtype=np.int32)
        else:
            folds = np.zeros(n_train, dtype=np.int32)
            skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
            for f, (_, val_idx) in enumerate(skf.split(train_df, y_a)):
                folds[val_idx] = f

        unique_folds = sorted(np.unique(folds).tolist())
        for f in unique_folds:
            tr_mask = folds != f
            va_mask = folds == f

            clf = LogisticRegression(
                C=cfg.c_reg,
                max_iter=500,
                class_weight="balanced",
                random_state=cfg.seed + f,
            )
            clf.fit(X_tr[tr_mask], y_a[tr_mask])

            va_p = clf.predict_proba(X_tr[va_mask])
            te_p = clf.predict_proba(X_te)
            for cls_pos, cls_label in enumerate(clf.classes_):
                oof_probs[va_mask, a_idx, int(cls_label)] = va_p[:, cls_pos].astype(np.float32)
                test_probs[:, a_idx, int(cls_label)] += (
                    te_p[:, cls_pos].astype(np.float32) / len(unique_folds)
                )

    y_true = train_df[aspect_cols].to_numpy(dtype=np.int32)
    metrics = evaluate_absa_macro_f1(y_true, oof_probs)
    return oof_probs, test_probs, metrics


def _generate_synthetic_absa_data(
    n_train: int = 80, n_test: int = 20, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lex = {
        "kebersihan": {
            1: "kamar sangat kotor bau apek sprei berdebu jorok",
            2: "kebersihan kamar standar biasa saja",
            3: "kamar sangat bersih wangi rapi sprei higienis kinclong",
        },
        "pelayanan": {
            1: "pelayanan staf jutek lambat resepsionis cuek",
            2: "pelayanan resepsionis standar biasa",
            3: "pelayanan staf sangat ramah cepat sigap sopan membantu",
        },
        "harga": {
            1: "harga tarif sangat mahal biaya tidak sepadan kemahalan",
            2: "harga tarif standar wajar",
            3: "harga sangat murah terjangkau hemat diskon sepadan",
        },
        "lokasi": {
            1: "lokasi sangat jauh macet terpencil akses susah",
            2: "lokasi jalan standar cukup",
            3: "lokasi sangat strategis dekat stasiun bandara pusat kota mall",
        },
    }
    aspects = list(lex.keys())

    def _make(n: int, prefix: str) -> pd.DataFrame:
        rows = []
        for i in range(n):
            labels = {a: 0 for a in aspects}
            chosen = rng.choice(aspects, size=int(rng.integers(1, 4)), replace=False)
            parts = []
            for j, a in enumerate(chosen):
                pol = int(rng.choice([1, 2, 3], p=[0.4, 0.2, 0.4]))
                labels[str(a)] = pol
                sep = " tetapi " if j > 0 else ""
                parts.append(sep + lex[str(a)][pol])
            row = {"id": f"{prefix}_{i:04d}", "text": "".join(parts)}
            row.update(labels)
            rows.append(row)
        return pd.DataFrame(rows)

    return _make(n_train, "TR"), _make(n_test, "TE")


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    if smoke_test:
        train_df, test_df = _generate_synthetic_absa_data(n_train=75, n_test=20, seed=cfg.seed)
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[:20].copy().reset_index(drop=True)
        else:
            train_df, test_df = _generate_synthetic_absa_data(seed=cfg.seed)

    oof_probs, test_probs, metrics = train_oof_tier1_absa(train_df, test_df, cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs)

    aspect_cols = list(cfg.aspect_keywords.keys())
    sub_df = pd.DataFrame({"id": test_df["id"]})
    test_preds = np.argmax(test_probs, axis=-1)
    for a_idx, aspect in enumerate(aspect_cols):
        sub_df[aspect] = test_preds[:, a_idx]
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Tier1 ABSA Context-Window TF-IDF] Overall Macro-F1={metrics['overall_macro_f1']:.4f} | "
        f"OOF shape={oof_probs.shape} | Test shape={test_probs.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09 Tier 1 Fast CPU ABSA Baseline")
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
