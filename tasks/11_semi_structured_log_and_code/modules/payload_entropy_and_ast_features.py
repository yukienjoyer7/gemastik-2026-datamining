#!/usr/bin/env python3
"""
Task 11 — Module: Payload Entropy, Character Markov & AST/Security Feature Extractor
====================================================================================
Extracts statistical, information-theoretic, and syntactic features designed for
cybersecurity payloads (SQLi, XSS, Shell injection, malicious URLs), obfuscated scripts,
system logs, and source code snippets:
  1. Global & Sliding-Window Shannon Entropy (catches Base64/Hex obfuscation, DGA domains,
     and encrypted shellcode blobs).
  2. Character-Class Transition Markov Score (measures transition probabilities between
     lowercase, uppercase, digits, whitespace, and punctuation/symbols).
  3. Bracket & Quote Balance Diagnostics (detects unmatched `'`, `"`, `()`, `{}`, `<>`
     characteristic of injection attacks and truncated logs).
  4. SQL / XSS / Shell / Source-Code AST Keyword & Operator Counters.

Usage:
    python modules/payload_entropy_and_ast_features.py --smoke-test
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
from typing import Dict, List, Optional, Sequence

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
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class CFG:
    """Configuration for Payload Entropy and AST/Security Feature Extractor."""

    window_size: int = 32
    window_stride: int = 16
    smoke_test: bool = False


_SQL_KEYWORDS = re.compile(
    r"\b(?:select|union|insert|update|delete|drop|alter|exec|execute|from|where|"
    r"group\s+by|order\s+by|having|waitfor|delay|benchmark|sleep|information_schema|"
    r"xp_cmdshell|concat|char|cast|convert|declare|varchar)\b|--|/\*|\*/|'\s*or\s*'?\d+'\s*=\s*'?\d+",
    re.IGNORECASE,
)

_XSS_AND_HTML_INJ = re.compile(
    r"<\s*script|javascript:|onerror\s*=|onload\s*=|onclick\s*=|onmouseover\s*=|"
    r"document\.cookie|document\.write|window\.location|eval\s*\(|fromcharcode|"
    r"alert\s*\(|<\s*iframe|<\s*svg|<\s*img\s+[^>]*src",
    re.IGNORECASE,
)

_SHELL_AND_PATH_INJ = re.compile(
    r"\.\./|\.\.\\|etc/passwd|etc/shadow|/bin/sh|/bin/bash|cmd\.exe|"
    r"powershell|\b(?:wget|curl|chmod|chown|nc|netcat|bash|whoami|uname)\b|"
    r"&&|\|\||;\s*(?:id|ls|cat|rm)\b|\$\([^)]+\)",
    re.IGNORECASE,
)

_CODE_AST_TOKENS = re.compile(
    r"\b(?:def|class|return|import|from|try|except|finally|raise|async|await|"
    r"lambda|yield|for|while|if|elif|else|function|const|let|var|public|private|"
    r"static|void|int|float|double|struct|interface|namespace|using)\b|=>|->|==|!=|<=|>=",
)

_HEX_OR_ENCODED = re.compile(
    r"(?:%[0-9a-fA-F]{2}){2,}|(?:\\x[0-9a-fA-F]{2}){2,}|(?:\\u[0-9a-fA-F]{4}){2,}|"
    r"\b0x[0-9a-fA-F]{4,}\b|\b[A-Za-z0-9+/]{24,}={0,2}\b"
)

_URL_PARAMS = re.compile(r"[?&][a-zA-Z0-9_.\-%]+=")


def compute_shannon_entropy(text: str) -> float:
    """Compute character-level Shannon entropy in bits per character."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def compute_sliding_window_entropy_stats(
    text: str, window_size: int = 32, stride: int = 16
) -> Dict[str, float]:
    """
    Compute global, maximum, and standard deviation of character Shannon entropy
    across sliding windows to detect localized obfuscated/Base64/hex payloads.
    """
    if not text:
        return {"global_entropy": 0.0, "max_window_entropy": 0.0, "std_window_entropy": 0.0}
    global_ent = compute_shannon_entropy(text)
    if len(text) <= window_size:
        return {
            "global_entropy": global_ent,
            "max_window_entropy": global_ent,
            "std_window_entropy": 0.0,
        }
    win_ents = [
        compute_shannon_entropy(text[i : i + window_size])
        for i in range(0, max(1, len(text) - window_size + 1), max(1, stride))
    ]
    return {
        "global_entropy": global_ent,
        "max_window_entropy": float(max(win_ents)),
        "std_window_entropy": float(np.std(win_ents)),
    }


def _char_class_id(ch: str) -> int:
    """Map character into 5 coarse classes: 0=lower, 1=upper, 2=digit, 3=space, 4=symbol."""
    if ch.islower():
        return 0
    if ch.isupper():
        return 1
    if ch.isdigit():
        return 2
    if ch.isspace():
        return 3
    return 4


def compute_char_transition_features(text: str) -> Dict[str, float]:
    """
    Compute character-class transition rates (e.g., rapid alternation between digits,
    uppercase, lowercase, and punctuation characteristic of obfuscation, hashes, or DGA).
    """
    if len(text) < 2:
        return {"class_switch_rate": 0.0, "alnum_to_symbol_rate": 0.0, "case_switch_rate": 0.0}

    classes = [_char_class_id(c) for c in text]
    n_trans = len(classes) - 1
    switches = sum(1 for a, b in zip(classes[:-1], classes[1:]) if a != b)
    alnum_sym = sum(
        1
        for a, b in zip(classes[:-1], classes[1:])
        if (a in (0, 1, 2) and b == 4) or (a == 4 and b in (0, 1, 2))
    )
    case_sw = sum(
        1
        for a, b in zip(classes[:-1], classes[1:])
        if (a == 0 and b == 1) or (a == 1 and b == 0)
    )
    return {
        "class_switch_rate": switches / n_trans,
        "alnum_to_symbol_rate": alnum_sym / n_trans,
        "case_switch_rate": case_sw / n_trans,
    }


def compute_bracket_and_quote_balance(text: str) -> Dict[str, float]:
    """
    Compute max nesting depth, bracket imbalance count, and odd-quote indicator.
    Injection payloads (`' OR 1=1 --`, `"><script>`) routinely exhibit unmatched quotes
    or asymmetric brackets.
    """
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = {v: k for k, v in pairs.items()}
    stack: List[str] = []
    max_depth = 0
    unmatched_brackets = 0

    for ch in text:
        if ch in pairs:
            stack.append(ch)
            max_depth = max(max_depth, len(stack))
        elif ch in closing:
            if stack and stack[-1] == closing[ch]:
                stack.pop()
            else:
                unmatched_brackets += 1

    unmatched_brackets += len(stack)
    single_quotes = text.count("'") - 2 * text.count("\\'")
    double_quotes = text.count('"') - 2 * text.count('\\"')
    odd_quote_flag = 1.0 if ((single_quotes % 2 == 1) or (double_quotes % 2 == 1)) else 0.0

    return {
        "max_bracket_depth": float(max_depth),
        "unmatched_brackets": float(unmatched_brackets),
        "odd_quote_flag": odd_quote_flag,
    }


def extract_payload_and_ast_feature_dict(
    raw_text: str, window_size: int = 32, stride: int = 16
) -> Dict[str, float]:
    """Extract 18 structural, entropy, Markov, and security/AST features for a single string."""
    text = str(raw_text or "")
    c_len = len(text)
    denom = max(c_len, 1)
    punct_set = set(string.punctuation)

    ent = compute_sliding_window_entropy_stats(text, window_size=window_size, stride=stride)
    trans = compute_char_transition_features(text)
    bal = compute_bracket_and_quote_balance(text)

    symbol_cnt = sum(1 for c in text if c in punct_set)
    digit_cnt = sum(1 for c in text if c.isdigit())
    upper_cnt = sum(1 for c in text if c.isupper())

    sql_hits = len(_SQL_KEYWORDS.findall(text))
    xss_hits = len(_XSS_AND_HTML_INJ.findall(text))
    shell_hits = len(_SHELL_AND_PATH_INJ.findall(text))
    code_hits = len(_CODE_AST_TOKENS.findall(text))
    hex_hits = len(_HEX_OR_ENCODED.findall(text))
    url_params = len(_URL_PARAMS.findall(text))

    lines = text.splitlines() or [text]
    indent_spaces = [len(line) - len(line.lstrip(" \t")) for line in lines if line.strip()]
    max_indent = float(max(indent_spaces)) if indent_spaces else 0.0

    return {
        "log_char_len": float(np.log1p(c_len)),
        "global_entropy": ent["global_entropy"],
        "max_window_entropy": ent["max_window_entropy"],
        "std_window_entropy": ent["std_window_entropy"],
        "symbol_density": symbol_cnt / denom,
        "digit_density": digit_cnt / denom,
        "upper_density": upper_cnt / denom,
        "class_switch_rate": trans["class_switch_rate"],
        "alnum_to_symbol_rate": trans["alnum_to_symbol_rate"],
        "case_switch_rate": trans["case_switch_rate"],
        "max_bracket_depth": bal["max_bracket_depth"],
        "unmatched_brackets": bal["unmatched_brackets"],
        "odd_quote_flag": bal["odd_quote_flag"],
        "sql_keyword_hits": float(sql_hits),
        "xss_keyword_hits": float(xss_hits),
        "shell_path_hits": float(shell_hits),
        "code_ast_hits": float(code_hits),
        "hex_or_encoded_hits": float(hex_hits + url_params) + 0.01 * max_indent,
    }


FEATURE_NAMES: List[str] = [
    "log_char_len",
    "global_entropy",
    "max_window_entropy",
    "std_window_entropy",
    "symbol_density",
    "digit_density",
    "upper_density",
    "class_switch_rate",
    "alnum_to_symbol_rate",
    "case_switch_rate",
    "max_bracket_depth",
    "unmatched_brackets",
    "odd_quote_flag",
    "sql_keyword_hits",
    "xss_keyword_hits",
    "shell_path_hits",
    "code_ast_hits",
    "hex_or_encoded_hits",
]


class PayloadEntropyAndASTFeaturizer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer outputting the 18 payload/entropy/AST features as a CSR matrix."""

    def __init__(self, window_size: int = 32, stride: int = 16) -> None:
        self.window_size = window_size
        self.stride = stride

    def fit(self, X: Sequence[str], y: Optional[np.ndarray] = None) -> "PayloadEntropyAndASTFeaturizer":
        return self

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        mat = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float32)
        for i, raw in enumerate(X):
            d = extract_payload_and_ast_feature_dict(
                raw, window_size=self.window_size, stride=self.stride
            )
            for j, k in enumerate(FEATURE_NAMES):
                mat[i, j] = d[k]
        return sparse.csr_matrix(mat)


def _run_smoke_test() -> None:
    samples = [
        "SELECT * FROM users WHERE username = 'admin' OR '1'='1' -- AND password = ''",
        "<script>eval(String.fromCharCode(97,108,101,114,116,40,49,41))</script>",
        "GET /index.php?page=../../../../etc/passwd&tok=%41%42%43%44 HTTP/1.1",
        "def forward(self, x):\n    if x is not None:\n        return self.net(x)\n",
        "Normal clean customer support message asking about delivery status.",
    ]
    fe = PayloadEntropyAndASTFeaturizer()
    X_csr = fe.fit_transform(samples)
    dense = X_csr.toarray()
    assert dense.shape == (5, 18)
    assert dense[0, FEATURE_NAMES.index("sql_keyword_hits")] >= 3.0
    assert dense[1, FEATURE_NAMES.index("xss_keyword_hits")] >= 2.0
    assert dense[2, FEATURE_NAMES.index("shell_path_hits")] >= 2.0
    assert dense[3, FEATURE_NAMES.index("code_ast_hits")] >= 3.0
    print(f"[SMOKE TEST] Extracted {dense.shape[1]} features across {dense.shape[0]} samples.")
    print("[SUCCESS] payload_entropy_and_ast_features smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Payload Entropy & AST Feature Extractor")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
