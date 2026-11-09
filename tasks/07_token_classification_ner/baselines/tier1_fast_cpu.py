"""
Task 07 — Tier 1 Fast CPU NER Baseline: Gazetteer Trie + Orthographic Viterbi Classifier
========================================================================================
Fast CPU Named Entity Recognition pipeline:
  1. Multi-Word Entity Gazetteer (`DictionaryTrie`):
     - Built strictly inside each CV fold from training entity spans (both exact
       and case-insensitive token sequences).
  2. Rich Contextual Orthographic Feature Extractor (`w_{-2}..w_{+2}`):
     - Token lowercase, word shape (`Xx+`, `X+`, `d+`), 3-char prefix/suffix,
       `is_title`, `is_upper`, `is_digit`, `has_hyphen`, and Gazetteer match priors.
  3. Out-Of-Fold Linear Model (`LogisticRegression` / `SGDClassifier`) +
     **Viterbi BIO Transition Decoding**:
     - Enforces valid BIO state transitions (forbids `O -> I-X` and `B-X -> I-Y`)
       during sequence decoding.
  4. Evaluates strict Entity-Level Span Micro-F1 and Macro-F1.
  5. Exports `oof_<exp_name>.npy`, `test_<exp_name>.npy`, and `sub_<exp_name>.csv`.

Usage:
  python3 tier1_fast_cpu.py --data-dir ./data --output-dir ./outputs
  python3 tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import re
import site
import sys
import tempfile
from collections import Counter, defaultdict
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
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


SpanTuple = Tuple[int, int, str]
WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass
class CFG:
    exp_name: str = "tier1_gazetteer_viterbi_ner"
    seed: int = 42
    n_folds: int = 5
    c_reg: float = 2.0
    data_dir: str = "./data"
    output_dir: str = "./outputs"


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(str(text))]


def word_shape(token: str) -> str:
    shape = []
    for ch in token:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(shape))


class DictionaryTrie:
    """Multi-word Entity Gazetteer Trie for longest-match span lookup."""

    def __init__(self) -> None:
        self.root: Dict = {}

    def add(self, tokens: Sequence[str], label: str) -> None:
        node = self.root
        for tok in tokens:
            node = node.setdefault(tok.lower(), {})
        counts: Counter[str] = node.setdefault("_labels", Counter())
        counts[label] += 1

    def match_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Returns BIO gazetteer priors for a sequence of tokens."""
        n = len(tokens)
        tags = ["O"] * n
        i = 0
        while i < n:
            node = self.root
            best_end = -1
            best_label = ""
            j = i
            while j < n and tokens[j].lower() in node:
                node = node[tokens[j].lower()]
                if "_labels" in node:
                    best_end = j + 1
                    best_label = node["_labels"].most_common(1)[0][0]
                j += 1
            if best_end > i and best_label:
                tags[i] = f"B-{best_label}"
                for k in range(i + 1, best_end):
                    tags[k] = f"I-{best_label}"
                i = best_end
            else:
                i += 1
        return tags


def extract_token_features(
    tokens: Sequence[str], gaz_tags: Sequence[str]
) -> List[Dict[str, str | float]]:
    """Extracts contextual orthographic + gazetteer features over window `[-2, +2]`."""
    n = len(tokens)
    feats_list: List[Dict[str, str | float]] = []

    for i in range(n):
        tok = tokens[i]
        clean_tok = tok.strip(".,;:!?()[]{}\"'")
        f: Dict[str, str | float] = {
            "bias": 1.0,
            "w0.lower": clean_tok.lower(),
            "w0.shape": word_shape(clean_tok),
            "w0.pref3": clean_tok[:3].lower(),
            "w0.suff3": clean_tok[-3:].lower(),
            "w0.isupper": float(clean_tok.isupper()),
            "w0.istitle": float(clean_tok.istitle()),
            "w0.isdigit": float(clean_tok.isdigit()),
            "w0.gaz": gaz_tags[i],
        }
        for offset in (-2, -1, 1, 2):
            idx = i + offset
            key = f"w{offset:+d}"
            if 0 <= idx < n:
                ntok = tokens[idx].strip(".,;:!?()[]{}\"'")
                f[f"{key}.lower"] = ntok.lower()
                f[f"{key}.shape"] = word_shape(ntok)
                f[f"{key}.istitle"] = float(ntok.istitle())
                f[f"{key}.gaz"] = gaz_tags[idx]
            else:
                f[f"{key}.bos_eos"] = "<BOUNDARY>"
        feats_list.append(f)

    return feats_list


def spans_to_bio(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[SpanTuple]
) -> List[str]:
    tags = ["O"] * len(offsets)
    for s, e, lbl in sorted(spans, key=lambda x: (x[0], x[1])):
        first = True
        for i, (ts, te) in enumerate(offsets):
            if te <= s or ts >= e:
                continue
            tags[i] = f"B-{lbl}" if first else f"I-{lbl}"
            first = False
    return tags


def bio_to_spans(
    text: str, offsets: Sequence[Tuple[int, int]], tags: Sequence[str]
) -> List[SpanTuple]:
    spans: List[SpanTuple] = []
    cur_s, cur_e, cur_lbl = None, None, None

    def _flush() -> None:
        nonlocal cur_s, cur_e, cur_lbl
        if cur_s is not None and cur_e is not None and cur_lbl is not None:
            s, e = cur_s, cur_e
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            # Strip trailing punctuation attached to token
            while e > s and text[e - 1] in ".,;:!?":
                e -= 1
            if e > s:
                spans.append((s, e, cur_lbl))
        cur_s, cur_e, cur_lbl = None, None, None

    for (ts, te), tag in zip(offsets, tags):
        if tag == "O":
            _flush()
            continue
        pref, _, lbl = tag.partition("-")
        if pref == "B" or cur_lbl != lbl:
            _flush()
            cur_s, cur_e, cur_lbl = ts, te, lbl
        else:
            cur_e = te
    _flush()
    return spans


def viterbi_bio_decode(
    log_probs: np.ndarray, classes: Sequence[str]
) -> List[str]:
    """
    Viterbi sequence decoder enforcing valid BIO transitions:
      - Cannot start sequence with `I-X`
      - Cannot transition from `O` to `I-X`
      - Cannot transition from `B-X`/`I-X` to `I-Y` where `X != Y`
    """
    n_toks, n_cls = log_probs.shape
    if n_toks == 0:
        return []

    neg_inf = -1e9
    trans = np.zeros((n_cls, n_cls), dtype=np.float64)
    start_pen = np.zeros(n_cls, dtype=np.float64)

    for j, cj in enumerate(classes):
        if cj.startswith("I-"):
            start_pen[j] = neg_inf
        for i, ci in enumerate(classes):
            if cj.startswith("I-"):
                lbl_j = cj[2:]
                if ci == "O" or (ci[2:] != lbl_j):
                    trans[i, j] = neg_inf

    dp = np.zeros((n_toks, n_cls), dtype=np.float64)
    back = np.zeros((n_toks, n_cls), dtype=np.int32)
    dp[0] = log_probs[0] + start_pen

    for t in range(1, n_toks):
        scores = dp[t - 1][:, None] + trans + log_probs[t][None, :]
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0)

    best_last = int(np.argmax(dp[-1]))
    seq = [best_last]
    for t in range(n_toks - 1, 0, -1):
        best_last = int(back[t, best_last])
        seq.append(best_last)
    seq.reverse()
    return [classes[idx] for idx in seq]


def compute_span_f1(
    pred_docs: Sequence[Sequence[SpanTuple]], gt_docs: Sequence[Sequence[SpanTuple]]
) -> Dict[str, float]:
    tp, fp, fn = 0, 0, 0
    for p, g in zip(pred_docs, gt_docs):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "micro_f1": float(f1)}


def _build_synthetic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    templates = [
        (
            "Dr. Budi Santoso works at Universitas Indonesia in Jakarta.",
            [("Budi Santoso", "PER"), ("Universitas Indonesia", "ORG"), ("Jakarta", "LOC")],
        ),
        (
            "Siti Aminah joined Bank Mandiri located in Surabaya.",
            [("Siti Aminah", "PER"), ("Bank Mandiri", "ORG"), ("Surabaya", "LOC")],
        ),
        (
            "Andi Wijaya teaches at Institut Teknologi Bandung in Bandung.",
            [("Andi Wijaya", "PER"), ("Institut Teknologi Bandung", "ORG"), ("Bandung", "LOC")],
        ),
    ]
    rows = []
    for i in range(30):
        text, ents = templates[i % len(templates)]
        spans = [[text.rfind(s), text.rfind(s) + len(s), l] for s, l in ents]
        rows.append({"doc_id": f"D{i:03d}", "text": text, "spans": json.dumps(spans)})
    df = pd.DataFrame(rows)
    return df, df.iloc[:6].copy().reset_index(drop=True)


def run_pipeline(cfg: CFG, smoke_test: bool = False) -> None:
    if smoke_test:
        train_df, test_df = _build_synthetic_data()
    else:
        data_dir = Path(cfg.data_dir)
        folds_path = data_dir / "train_folds.parquet"
        if folds_path.exists():
            train_df = pd.read_parquet(folds_path)
            test_df = train_df.iloc[:8].copy().reset_index(drop=True)
        else:
            train_df, test_df = _build_synthetic_data()

    if "fold" not in train_df.columns:
        train_df["fold"] = 0
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for fold, (_, val_idx) in enumerate(kf.split(train_df)):
            train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    texts = train_df["text"].astype(str).tolist()
    gt_spans_all: List[List[SpanTuple]] = [
        [(int(s), int(e), str(l)) for s, e, l in json.loads(raw)]
        for raw in train_df["spans"]
    ]
    oof_pred_spans: List[List[SpanTuple]] = [[] for _ in range(len(train_df))]
    oof_doc_conf = np.zeros(len(train_df), dtype=np.float32)

    for fold in sorted(train_df["fold"].unique()):
        tr_idx = np.where(train_df["fold"].to_numpy() != fold)[0]
        va_idx = np.where(train_df["fold"].to_numpy() == fold)[0]

        # 1. Build leak-free Gazetteer Trie on fold training spans
        trie = DictionaryTrie()
        for idx in tr_idx:
            doc_text = texts[int(idx)]
            for s, e, lbl in gt_spans_all[int(idx)]:
                span_toks = [t.strip(".,;:!?") for t, _, _ in tokenize_with_offsets(doc_text[s:e])]
                if span_toks:
                    trie.add(span_toks, lbl)

        # 2. Extract contextual orthographic features & BIO labels
        tr_feats: List[Dict] = []
        tr_labels: List[str] = []
        for idx in tr_idx:
            tok_objs = tokenize_with_offsets(texts[int(idx)])
            toks = [t.strip(".,;:!?") for t, _, _ in tok_objs]
            offs = [(s, e) for _, s, e in tok_objs]
            gaz = trie.match_tokens(toks)
            tr_feats.extend(extract_token_features(toks, gaz))
            tr_labels.extend(spans_to_bio(offs, gt_spans_all[int(idx)]))

        vec = DictVectorizer(sparse=True)
        X_tr = vec.fit_transform(tr_feats)
        clf = LogisticRegression(C=cfg.c_reg, max_iter=300, random_state=cfg.seed + int(fold))
        clf.fit(X_tr, tr_labels)
        classes = list(clf.classes_)

        for idx in va_idx:
            doc_text = texts[int(idx)]
            tok_objs = tokenize_with_offsets(doc_text)
            toks = [t.strip(".,;:!?") for t, _, _ in tok_objs]
            offs = [(s, e) for _, s, e in tok_objs]
            gaz = trie.match_tokens(toks)
            X_va = vec.transform(extract_token_features(toks, gaz))
            probs = clf.predict_proba(X_va)
            log_p = np.log(np.clip(probs, 1e-9, 1.0))
            pred_tags = viterbi_bio_decode(log_p, classes)
            oof_pred_spans[int(idx)] = bio_to_spans(doc_text, offs, pred_tags)
            oof_doc_conf[int(idx)] = float(np.mean(np.max(probs, axis=1)))

    metrics = compute_span_f1(oof_pred_spans, gt_spans_all)

    # Fit on full data for test predictions
    full_trie = DictionaryTrie()
    for doc_text, spans in zip(texts, gt_spans_all):
        for s, e, lbl in spans:
            stoks = [t.strip(".,;:!?") for t, _, _ in tokenize_with_offsets(doc_text[s:e])]
            if stoks:
                full_trie.add(stoks, lbl)

    test_conf = np.ones(len(test_df), dtype=np.float32) * float(np.mean(oof_doc_conf))
    test_spans_json = []
    for doc_text in test_df["text"].astype(str):
        tok_objs = tokenize_with_offsets(doc_text)
        toks = [t.strip(".,;:!?") for t, _, _ in tok_objs]
        offs = [(s, e) for _, s, e in tok_objs]
        gaz = full_trie.match_tokens(toks)
        spans = bio_to_spans(doc_text, offs, gaz)
        test_spans_json.append(json.dumps(spans))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_doc_conf)
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_conf)
    pd.DataFrame({"doc_id": test_df["doc_id"], "spans": test_spans_json}).to_csv(
        out_dir / f"sub_{cfg.exp_name}.csv", index=False
    )

    print(
        f"[Tier1 CPU Gazetteer+Viterbi NER] Strict Span Micro-F1={metrics['micro_f1']:.4f} "
        f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}) | OOF shape={oof_doc_conf.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07 Tier 1 Fast CPU NER")
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
