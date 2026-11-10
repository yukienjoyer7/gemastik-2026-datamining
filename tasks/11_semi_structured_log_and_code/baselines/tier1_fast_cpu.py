#!/usr/bin/env python3
"""
Task 11 — Semi-Structured Log, Markup & Code: Tier 1 Fast CPU Baseline
======================================================================
Standalone Out-Of-Fold (OOF) pipeline combining:
  1. Structure-Preserving Tokenization:
     - HTML/XML DOM tag hierarchy paths (`dom_path:div>span.price`) & attributes
     - JSON/KV flattened key paths (`json_key:user.role`)
     - Drain-style log template IDs (`[T_0001]`) and variable masks (`<IP>`, `<HEX>`, `<PATH>`)
  2. Multi-Granularity TF-IDF:
     - Structural token Word `(1, 3)` TF-IDF
     - Case-sensitive Byte/Char `(2, 6)` TF-IDF (`analyzer="char"`, `lowercase=False`)
  3. Dense Structural & Entropy Features (18 features):
     - Global & sliding-window Shannon entropy, symbol/digit/uppercase density,
       character-class Markov switch rates, bracket nesting depth, unmatched bracket/quote
       flags, hex/base64 counts, URL parameter counts, and SQL/XSS/Shell/Code keyword hits.
  4. Out-Of-Fold Ensemble:
     - Blends `HistGradientBoostingClassifier` (on SVD + dense structural/entropy features)
       with `LogisticRegression` and `RidgeClassifier` (on full sparse + dense union).
     - Exports `oof_tier1_semistruct.npy`, `test_tier1_semistruct.npy`, and `sub_tier1_semistruct.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import site
import string
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
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
from scipy import sparse
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class CFG:
    """Configuration for Task 11 Tier 1 Semi-Structured CPU Baseline."""

    exp_name: str = "tier1_semistruct"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"

    # TF-IDF & SVD parameters
    struct_ngram_range: Tuple[int, int] = (1, 3)
    char_ngram_range: Tuple[int, int] = (2, 6)
    max_struct_features: int = 40000
    max_char_features: int = 80000
    svd_components: int = 32
    min_df: int = 2

    # Model hyperparameters
    logreg_c: float = 2.0
    ridge_alpha: float = 1.5
    gbdt_max_iter: int = 120
    gbdt_lr: float = 0.06
    blend_weights: Tuple[float, float, float] = (0.40, 0.40, 0.20)  # (GBDT, LogReg, Ridge)

    seed: int = 42
    smoke_test: bool = False


# =============================================================================
# 1. Self-Contained Structure-Preserving Tokenizer (DOM + JSON + Log Masks)
# =============================================================================

_MASK_REGEXES: List[Tuple[str, re.Pattern[str]]] = [
    ("<TIME>", re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")),
    ("<UUID>", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("<IP>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")),
    ("<HEX>", re.compile(r"\b0x[0-9a-fA-F]+\b|\bblk_-?\d+\b")),
    ("<PATH>", re.compile(r"(?:^|(?<=[\s\"'=:(]))(?:/[a-zA-Z0-9_.\-]+){2,}")),
    ("<NUM>", re.compile(r"(?<![a-zA-Z_])-?\b\d+(?:\.\d+)?\b(?![a-zA-Z_])")),
]

_KV_REGEX = re.compile(r'(?:"([a-zA-Z0-9_.\-]+)"|([a-zA-Z0-9_.\-]+))\s*[:=]\s*')
_SQL_REGEX = re.compile(
    r"\b(?:select|union|insert|update|delete|drop|exec|from|where|sleep|benchmark|order\s+by)\b|--|/\*|\*/|'\s*or\s*",
    re.IGNORECASE,
)
_XSS_REGEX = re.compile(
    r"<\s*script|javascript:|onerror\s*=|onload\s*=|onclick\s*=|document\.cookie|eval\s*\(|alert\s*\(",
    re.IGNORECASE,
)
_SHELL_REGEX = re.compile(
    r"(?:\.\./)+|/etc/passwd|/bin/sh|/bin/bash|\b(?:wget|curl|chmod|nc|whoami)\b|&&|\|\|",
    re.IGNORECASE,
)
_CODE_REGEX = re.compile(
    r"\b(?:def|class|return|import|from|try|except|function|const|let|var|public|private|void)\b|=>|->|==|!="
)
_HEX_B64_REGEX = re.compile(
    r"(?:%[0-9a-fA-F]{2}){2,}|\b0x[0-9a-fA-F]{4,}\b|\b[A-Za-z0-9+/]{24,}={0,2}\b"
)
_URL_PARAM_REGEX = re.compile(r"[?&][a-zA-Z0-9_.\-%]+=")


class _MiniDOMExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.tokens: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower().strip()
        attr_map = {k.lower(): (v or "") for k, v in attrs if k}
        if "class" in attr_map and attr_map["class"].strip():
            cls_tok = ".".join(attr_map["class"].split()[:2])
            t = f"{t}.{cls_tok}"
        if len(self.stack) < 10:
            self.stack.append(t)
        self.tokens.append("dom_path:" + ">".join(self.stack))
        for k in attr_map:
            self.tokens.append(f"attr:{k}")

    def handle_endtag(self, tag: str) -> None:
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.tokens.append(cleaned)


def structure_preserving_serialize(raw_text: str) -> str:
    """
    Convert a raw log, HTML fragment, JSON payload, or code snippet into an enriched
    token stream preserving DOM paths, JSON keys, and Drain masked log skeletons.
    """
    text = str(raw_text or "")
    parts: List[str] = []

    if "<" in text and ">" in text:
        parser = _MiniDOMExtractor()
        try:
            parser.feed(text)
            parser.close()
            parts.extend(parser.tokens)
        except Exception:
            pass

    for m in _KV_REGEX.finditer(text):
        key = (m.group(1) or m.group(2) or "").strip()
        if key:
            parts.append(f"kv_key:{key}")

    masked = text
    for placeholder, pattern in _MASK_REGEXES:
        masked = pattern.sub(f" {placeholder} ", masked)
    parts.append(masked)
    return " ".join(parts)


# =============================================================================
# 2. Dense Structural, Entropy & AST Feature Extractor
# =============================================================================


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_structural_and_entropy_features(texts: Sequence[str]) -> np.ndarray:
    """
    Extract 14 dense features per document:
      0. log1p(char_len)
      1. global_shannon_entropy
      2. max_window_shannon_entropy (window=32)
      3. symbol_density
      4. digit_density
      5. uppercase_density
      6. max_bracket_depth
      7. unmatched_bracket_count
      8. odd_quote_flag
      9. hex_or_base64_token_count
     10. url_param_count
     11. sql_keyword_hits
     12. xss_or_html_inj_hits
     13. shell_or_code_ast_hits
    """
    punct_set = set(string.punctuation)
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {v: k for k, v in pairs.items()}
    feats = np.zeros((len(texts), 14), dtype=np.float32)

    for i, raw in enumerate(texts):
        s = str(raw or "")
        n = len(s)
        denom = max(n, 1)

        g_ent = _shannon_entropy(s)
        if n <= 32:
            max_w_ent = g_ent
        else:
            max_w_ent = max(
                _shannon_entropy(s[j : j + 32]) for j in range(0, n - 31, 16)
            )

        sym_cnt = sum(1 for c in s if c in punct_set)
        dig_cnt = sum(1 for c in s if c.isdigit())
        upp_cnt = sum(1 for c in s if c.isupper())

        stack: List[str] = []
        max_depth = 0
        unmatched = 0
        for ch in s:
            if ch in pairs:
                stack.append(ch)
                max_depth = max(max_depth, len(stack))
            elif ch in closing:
                if stack and stack[-1] == closing[ch]:
                    stack.pop()
                else:
                    unmatched += 1
        unmatched += len(stack)

        sq = s.count("'") - 2 * s.count("\\'")
        dq = s.count('"') - 2 * s.count('\\"')
        odd_quote = 1.0 if ((sq % 2 == 1) or (dq % 2 == 1)) else 0.0

        feats[i, 0] = np.log1p(n)
        feats[i, 1] = g_ent
        feats[i, 2] = max_w_ent
        feats[i, 3] = sym_cnt / denom
        feats[i, 4] = dig_cnt / denom
        feats[i, 5] = upp_cnt / denom
        feats[i, 6] = float(max_depth)
        feats[i, 7] = float(unmatched)
        feats[i, 8] = odd_quote
        feats[i, 9] = float(len(_HEX_B64_REGEX.findall(s)))
        feats[i, 10] = float(len(_URL_PARAM_REGEX.findall(s)))
        feats[i, 11] = float(len(_SQL_REGEX.findall(s)))
        feats[i, 12] = float(len(_XSS_REGEX.findall(s)))
        feats[i, 13] = float(len(_SHELL_REGEX.findall(s)) + len(_CODE_REGEX.findall(s)))

    return feats


# =============================================================================
# 3. OOF Training & Evaluation Loop
# =============================================================================


def ridge_to_probs(decision: np.ndarray, num_classes: int) -> np.ndarray:
    if decision.ndim == 1 or num_classes == 2:
        scores = np.vstack([-decision, decision]).T if decision.ndim == 1 else decision
        return softmax(scores, axis=1)
    return softmax(decision, axis=1)


def compute_metrics(y_true: np.ndarray, oof_probs: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(oof_probs, axis=1)
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    ll = float(log_loss(y_true, oof_probs))
    try:
        if oof_probs.shape[1] == 2:
            auc = float(roc_auc_score(y_true, oof_probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, oof_probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {"macro_f1": macro_f1, "log_loss": ll, "roc_auc": auc}


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    classes = np.sort(train_df[cfg.target_col].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = train_df[cfg.target_col].map(label_to_idx).values
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_classes = len(classes)

    raw_train = train_df[cfg.text_col].fillna("").astype(str).tolist()
    raw_test = test_df[cfg.text_col].fillna("").astype(str).tolist()

    struct_train = [structure_preserving_serialize(t) for t in raw_train]
    struct_test = [structure_preserving_serialize(t) for t in raw_test]

    dense_train_all = extract_structural_and_entropy_features(raw_train)
    dense_test_all = extract_structural_and_entropy_features(raw_test)

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
    w_gbdt, w_lr, w_ridge = cfg.blend_weights

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        struct_vec = TfidfVectorizer(
            ngram_range=cfg.struct_ngram_range,
            max_features=cfg.max_struct_features,
            min_df=1 if cfg.smoke_test else cfg.min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\S+",
        )
        char_vec = TfidfVectorizer(
            ngram_range=cfg.char_ngram_range,
            max_features=cfg.max_char_features,
            min_df=1 if cfg.smoke_test else cfg.min_df,
            sublinear_tf=True,
            analyzer="char",
            lowercase=False,
        )

        X_tr_s = struct_vec.fit_transform([struct_train[i] for i in tr_idx])
        X_va_s = struct_vec.transform([struct_train[i] for i in va_idx])
        X_te_s = struct_vec.transform(struct_test)

        X_tr_c = char_vec.fit_transform([raw_train[i] for i in tr_idx])
        X_va_c = char_vec.transform([raw_train[i] for i in va_idx])
        X_te_c = char_vec.transform(raw_test)

        scaler = StandardScaler()
        d_tr = scaler.fit_transform(dense_train_all[tr_idx])
        d_va = scaler.transform(dense_train_all[va_idx])
        d_te = scaler.transform(dense_test_all)

        X_tr_sparse = sparse.hstack([X_tr_s, X_tr_c, sparse.csr_matrix(d_tr)], format="csr")
        X_va_sparse = sparse.hstack([X_va_s, X_va_c, sparse.csr_matrix(d_va)], format="csr")
        X_te_sparse = sparse.hstack([X_te_s, X_te_c, sparse.csr_matrix(d_te)], format="csr")

        n_comp = min(cfg.svd_components, X_tr_sparse.shape[1] - 1, X_tr_sparse.shape[0] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed + int(f))
            svd_tr = svd.fit_transform(X_tr_sparse)
            svd_va = svd.transform(X_va_sparse)
            svd_te = svd.transform(X_te_sparse)
            X_tr_gbdt = np.hstack([dense_train_all[tr_idx], svd_tr])
            X_va_gbdt = np.hstack([dense_train_all[va_idx], svd_va])
            X_te_gbdt = np.hstack([dense_test_all, svd_te])
        else:
            X_tr_gbdt = dense_train_all[tr_idx]
            X_va_gbdt = dense_train_all[va_idx]
            X_te_gbdt = dense_test_all

        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. HistGradientBoostingClassifier (GBDT)
        clf_gbdt = HistGradientBoostingClassifier(
            max_iter=cfg.gbdt_max_iter,
            learning_rate=cfg.gbdt_lr,
            max_leaf_nodes=15,
            min_samples_leaf=5 if cfg.smoke_test else 15,
            random_state=cfg.seed + int(f),
        )
        clf_gbdt.fit(X_tr_gbdt, y_tr)
        p_va_gbdt = clf_gbdt.predict_proba(X_va_gbdt)
        p_te_gbdt = clf_gbdt.predict_proba(X_te_gbdt)

        # 2. LogisticRegression
        clf_lr = LogisticRegression(
            C=cfg.logreg_c, max_iter=1000, random_state=cfg.seed + int(f)
        )
        clf_lr.fit(X_tr_sparse, y_tr)
        p_va_lr = clf_lr.predict_proba(X_va_sparse)
        p_te_lr = clf_lr.predict_proba(X_te_sparse)

        # 3. RidgeClassifier
        clf_ridge = RidgeClassifier(alpha=cfg.ridge_alpha, random_state=cfg.seed + int(f))
        clf_ridge.fit(X_tr_sparse, y_tr)
        p_va_ridge = ridge_to_probs(clf_ridge.decision_function(X_va_sparse), num_classes)
        p_te_ridge = ridge_to_probs(clf_ridge.decision_function(X_te_sparse), num_classes)

        fold_va = w_gbdt * p_va_gbdt + w_lr * p_va_lr + w_ridge * p_va_ridge
        fold_te = w_gbdt * p_te_gbdt + w_lr * p_te_lr + w_ridge * p_te_ridge

        oof_probs[va_idx] = fold_va
        test_probs += fold_te / len(unique_folds)

        fold_f1 = f1_score(y_va, np.argmax(fold_va, axis=1), average="macro")
        print(f"  [Fold {f}] val_size={len(va_idx)} | Macro-F1={fold_f1:.4f} | total_sparse_dim={X_tr_sparse.shape[1]}")

    metrics = compute_metrics(y, oof_probs)
    print(
        f"[OOF SUMMARY] Macro-F1={metrics['macro_f1']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
    )

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs.astype(np.float32))

    sub_df = pd.DataFrame(
        {
            cfg.id_col: test_df[cfg.id_col].values,
            cfg.target_col: [classes[i] for i in np.argmax(test_probs, axis=1)],
        }
    )
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
    return oof_probs, test_probs, metrics


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def generate_synthetic_dataset(
    n_train: int = 120, n_test: int = 36, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    patterns = {
        0: [
            '2026-11-10 10:15:{s:02d} nginx[{p}]: 10.0.0.{ip} "GET /catalog/item_{u}.html HTTP/1.1" 200',
            '<div class="product"><span class="name">Item {u}</span><span class="price">Rp {p}00</span></div>',
        ],
        1: [
            "GET /search.php?q=1'%20UNION%20SELECT%20username,password%20FROM%20users--&id={u} HTTP/1.1",
            '<img src=x onerror="javascript:eval(document.cookie);alert({p})">',
        ],
        2: [
            '{{"service": "auth", "status": "ok", "uid": {u}, "latency_ms": {s}, "host": "10.1.1.{ip}"}}',
            "def validate_token_{u}(tok):\n    if tok == '0x{p:x}':\n        return True\n",
        ],
    }

    def _build(n: int, prefix: str, with_label: bool) -> pd.DataFrame:
        rows = []
        for i in range(n):
            cls = i % 3
            tmpl = patterns[cls][i % len(patterns[cls])]
            txt = tmpl.format(
                s=int(rng.integers(10, 59)),
                p=int(rng.integers(100, 999)),
                ip=int(rng.integers(2, 250)),
                u=int(rng.integers(1, 90)),
            )
            r = {"id": f"{prefix}_{i}", "text": txt}
            if with_label:
                r["label"] = cls
                r["fold"] = i % 4
            rows.append(r)
        return pd.DataFrame(rows)

    return _build(n_train, "tr", True), _build(n_test, "te", False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 11 Tier 1 Semi-Structured Fast CPU Baseline")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 11 Tier 1 Structure-Preserving + Char TF-IDF + GBDT/Linear OOF...")
            train_df, test_df = generate_synthetic_dataset(seed=cfg.seed)
            oof, test_preds, metrics = run_oof_pipeline(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            assert (pathlib.Path(tmpdir) / f"oof_{cfg.exp_name}.npy").exists()
            assert (pathlib.Path(tmpdir) / f"test_{cfg.exp_name}.npy").exists()
            assert (pathlib.Path(tmpdir) / f"sub_{cfg.exp_name}.csv").exists()
            assert metrics["macro_f1"] > 0.85, f"Expected Macro-F1 > 0.85, got {metrics['macro_f1']}"
            print("[SUCCESS] Task 11 Tier 1 smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_pipeline(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
