#!/usr/bin/env python3
"""
Task 11 — Module: Drain-Style Log Template Miner
================================================
Pure-Python implementation of Drain-style fixed-depth parse tree for log template
mining and parameter extraction:
  1. Masks variable tokens (IPv4/IPv6 addresses, ISO/syslog timestamps, UUIDs,
     hex addresses `0x...`, file paths `/var/...`, URLs, emails, and integers/floats)
     into canonical placeholders (`<IP>`, `<TIME>`, `<UUID>`, `<HEX>`, `<PATH>`,
     `<URL>`, `<NUM>`).
  2. Clusters log messages by token length and leading token prefixes into a
     fixed-depth tree, merging tokens that differ above a similarity threshold `sim_th`
     into `<*>`.
  3. Extracts both the deterministic `template_id` (`T_0001`), the structural template
     string, and the list of variable parameter values per log line.

Usage:
    python modules/drain_log_template_miner.py --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import site
import sys
from collections import Counter
from dataclasses import dataclass, field
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
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class CFG:
    """Configuration for Drain-style Log Template Miner."""

    depth: int = 4
    sim_th: float = 0.5
    max_children: int = 100
    max_tokens: int = 64
    smoke_test: bool = False


_MASK_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "<TIME>",
        re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\b"
        ),
    ),
    ("<UUID>", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("<IP>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")),
    ("<URL>", re.compile(r"\bhttps?://[^\s\"'<>]+")),
    ("<HEX>", re.compile(r"\b0x[0-9a-fA-F]+\b|\bblk_-?\d+\b")),
    ("<PATH>", re.compile(r"(?:^|(?<=[\s\"'=:(]))(?:/[a-zA-Z0-9_.\-]+){2,}(?:/[a-zA-Z0-9_.\-]*)?")),
    ("<NUM>", re.compile(r"(?<![a-zA-Z_])-?\b\d+(?:\.\d+)?\b(?![a-zA-Z_])")),
]


def mask_log_variables(raw_line: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Replace variable substrings (timestamps, IPs, UUIDs, hex IDs, file paths, numbers)
    with typed placeholders (`<TIME>`, `<IP>`, `<HEX>`, `<PATH>`, `<NUM>`) and return
    both the masked log string and the extracted `(placeholder_type, raw_value)` list.
    """
    line = str(raw_line or "").strip()
    extracted_params: List[Tuple[str, str]] = []

    for placeholder, pattern in _MASK_PATTERNS:
        for match in pattern.finditer(line):
            extracted_params.append((placeholder, match.group(0)))
        line = pattern.sub(placeholder, line)

    return line, extracted_params


@dataclass
class LogCluster:
    """Represents a discovered log template cluster in Drain."""

    cluster_id: int
    template_tokens: List[str]
    size: int = 1

    @property
    def template_id(self) -> str:
        sig = " ".join(self.template_tokens)
        short_hash = hashlib.md5(sig.encode("utf-8")).hexdigest()[:6]
        return f"T_{self.cluster_id:04d}_{short_hash}"

    @property
    def template_str(self) -> str:
        return " ".join(self.template_tokens)


@dataclass
class _DrainNode:
    children: Dict[str, "_DrainNode"] = field(default_factory=dict)
    clusters: List[LogCluster] = field(default_factory=list)


class DrainLogTemplateMiner:
    """
    Pure-Python Drain Log Parser (He et al., ICWS 2017):
      - Layer 1: Groups by token count.
      - Layers 2..depth: Routes by leading non-wildcard tokens.
      - Leaf: Computes token-sequence similarity against candidate `LogCluster`s
        and merges into the best cluster if `similarity >= sim_th`, updating differing
        positions to `<*>`.
    """

    def __init__(
        self,
        depth: int = 4,
        sim_th: float = 0.5,
        max_children: int = 100,
        max_tokens: int = 64,
    ) -> None:
        self.depth = max(3, depth)
        self.sim_th = sim_th
        self.max_children = max_children
        self.max_tokens = max_tokens
        self.root = _DrainNode()
        self.clusters: List[LogCluster] = []

    @staticmethod
    def _is_variable_token(tok: str) -> bool:
        return (tok.startswith("<") and tok.endswith(">")) or any(ch.isdigit() for ch in tok)

    def _tokenize(self, line: str) -> List[str]:
        masked, _ = mask_log_variables(line)
        return masked.split()[: self.max_tokens]

    def _tree_search(self, tokens: List[str]) -> Optional[LogCluster]:
        seq_len_key = str(len(tokens))
        if seq_len_key not in self.root.children:
            return None

        curr = self.root.children[seq_len_key]
        max_prefix_depth = min(self.depth - 2, len(tokens))
        for i in range(max_prefix_depth):
            tok = tokens[i]
            key = "<*>" if self._is_variable_token(tok) else tok
            if key in curr.children:
                curr = curr.children[key]
            elif "<*>" in curr.children:
                curr = curr.children["<*>"]
            else:
                return None

        return self._fast_match(curr.clusters, tokens)

    def _fast_match(self, clusters: List[LogCluster], tokens: List[str]) -> Optional[LogCluster]:
        best_sim = -1.0
        best_param_count = -1
        best_cluster: Optional[LogCluster] = None

        for cluster in clusters:
            sim, param_cnt = self._seq_dist(cluster.template_tokens, tokens)
            if sim > best_sim or (sim == best_sim and param_cnt > best_param_count):
                best_sim = sim
                best_param_count = param_cnt
                best_cluster = cluster

        if best_sim >= self.sim_th:
            return best_cluster
        return None

    @staticmethod
    def _seq_dist(template: List[str], tokens: List[str]) -> Tuple[float, int]:
        if len(template) == 0:
            return 1.0, 0
        sim_tokens = 0
        param_count = 0
        for t_tok, m_tok in zip(template, tokens):
            if t_tok == "<*>" or (t_tok.startswith("<") and t_tok.endswith(">") and t_tok == m_tok):
                param_count += 1
                if t_tok == m_tok:
                    sim_tokens += 1
            elif t_tok == m_tok:
                sim_tokens += 1
        return float(sim_tokens) / len(template), param_count

    def _add_seq_to_prefix_tree(self, cluster: LogCluster) -> None:
        tokens = cluster.template_tokens
        seq_len_key = str(len(tokens))
        if seq_len_key not in self.root.children:
            self.root.children[seq_len_key] = _DrainNode()
        curr = self.root.children[seq_len_key]

        max_prefix_depth = min(self.depth - 2, len(tokens))
        for i in range(max_prefix_depth):
            tok = tokens[i]
            key = "<*>" if self._is_variable_token(tok) else tok
            if key not in curr.children:
                if len(curr.children) < self.max_children:
                    curr.children[key] = _DrainNode()
                elif "<*>" in curr.children:
                    key = "<*>"
                else:
                    curr.children["<*>"] = _DrainNode()
                    key = "<*>"
            curr = curr.children[key]

        curr.clusters.append(cluster)

    @staticmethod
    def _update_template(template: List[str], tokens: List[str]) -> None:
        for i, (t_tok, m_tok) in enumerate(zip(template, tokens)):
            if t_tok != m_tok:
                template[i] = "<*>"

    def fit(self, log_lines: Sequence[str]) -> "DrainLogTemplateMiner":
        self.root = _DrainNode()
        self.clusters = []
        for line in log_lines:
            tokens = self._tokenize(line)
            if not tokens:
                continue
            matched = self._tree_search(tokens)
            if matched is None:
                new_cluster = LogCluster(cluster_id=len(self.clusters) + 1, template_tokens=list(tokens), size=1)
                self.clusters.append(new_cluster)
                self._add_seq_to_prefix_tree(new_cluster)
            else:
                self._update_template(matched.template_tokens, tokens)
                matched.size += 1
        return self

    def parse_line(self, line: str) -> Dict[str, object]:
        masked, params = mask_log_variables(line)
        tokens = masked.split()[: self.max_tokens]
        if not tokens:
            return {
                "template_id": "T_EMPTY",
                "template_str": "",
                "masked_str": "",
                "params": [],
                "is_unseen_template": True,
                "cluster_size": 0,
            }
        matched = self._tree_search(tokens)
        if matched is None:
            return {
                "template_id": f"T_UNSEEN_LEN{len(tokens)}",
                "template_str": masked,
                "masked_str": masked,
                "params": params,
                "is_unseen_template": True,
                "cluster_size": 0,
            }
        return {
            "template_id": matched.template_id,
            "template_str": matched.template_str,
            "masked_str": masked,
            "params": params,
            "is_unseen_template": False,
            "cluster_size": matched.size,
        }

    def transform_to_annotated_strings(self, log_lines: Sequence[str]) -> List[str]:
        """Convert raw log lines into `[T_ID] <template_str> [PARAMS: <IP> <NUM>]` strings."""
        out: List[str] = []
        for line in log_lines:
            info = self.parse_line(line)
            param_types = " ".join(p[0] for p in info["params"])  # type: ignore[index]
            out.append(f"[{info['template_id']}] {info['template_str']} {param_types}".strip())
        return out


class DrainLogFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that fits `DrainLogTemplateMiner` on training
    logs and extracts 8 dense log-template & parameter statistics per line:
      0. log1p(template_cluster_frequency_in_train)
      1. is_unseen_template_flag
      2. num_extracted_parameters
      3. count(<IP>)
      4. count(<HEX>)
      5. count(<PATH>)
      6. count(<NUM>)
      7. wildcard_ratio_in_template
    """

    def __init__(self, depth: int = 4, sim_th: float = 0.5) -> None:
        self.depth = depth
        self.sim_th = sim_th
        self.miner_: Optional[DrainLogTemplateMiner] = None

    def fit(self, X: Sequence[str], y: Optional[np.ndarray] = None) -> "DrainLogFeatureExtractor":
        self.miner_ = DrainLogTemplateMiner(depth=self.depth, sim_th=self.sim_th)
        self.miner_.fit(X)
        return self

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        assert self.miner_ is not None, "DrainLogFeatureExtractor must be fitted before transform."
        feats = np.zeros((len(X), 8), dtype=np.float32)
        for i, line in enumerate(X):
            info = self.miner_.parse_line(line)
            params: List[Tuple[str, str]] = info["params"]  # type: ignore[assignment]
            p_counts = Counter(p[0] for p in params)
            tmpl_tokens = str(info["template_str"]).split()
            wc_cnt = sum(1 for t in tmpl_tokens if t == "<*>" or (t.startswith("<") and t.endswith(">")))

            feats[i, 0] = np.log1p(float(info["cluster_size"]))  # type: ignore[arg-type]
            feats[i, 1] = 1.0 if info["is_unseen_template"] else 0.0
            feats[i, 2] = float(len(params))
            feats[i, 3] = float(p_counts.get("<IP>", 0))
            feats[i, 4] = float(p_counts.get("<HEX>", 0))
            feats[i, 5] = float(p_counts.get("<PATH>", 0))
            feats[i, 6] = float(p_counts.get("<NUM>", 0))
            feats[i, 7] = wc_cnt / max(len(tmpl_tokens), 1)

        return sparse.csr_matrix(feats)


def _run_smoke_test() -> None:
    sample_logs = [
        "2026-10-12 08:15:01 sshd[1024]: Failed password for root from 192.168.1.50 port 54321 ssh2",
        "2026-10-12 08:15:04 sshd[1029]: Failed password for admin from 10.0.0.18 port 49112 ssh2",
        "2026-10-12 08:16:10 dfs.DataNode: Receiving block blk_-1608999687919862906 src: 10.250.19.102:50010 dest: /var/data/hdfs/block01",
        "2026-10-12 08:16:15 dfs.DataNode: Receiving block blk_7503483334202473044 src: 10.250.10.6:50010 dest: /var/data/hdfs/block02",
    ]
    miner = DrainLogTemplateMiner(depth=4, sim_th=0.5).fit(sample_logs)
    assert len(miner.clusters) == 2, f"Expected 2 distinct log templates, got {len(miner.clusters)}"

    parsed_0 = miner.parse_line(sample_logs[0])
    parsed_1 = miner.parse_line(sample_logs[1])
    assert parsed_0["template_id"] == parsed_1["template_id"]
    assert "<*>" in str(parsed_0["template_str"])

    fe = DrainLogFeatureExtractor().fit(sample_logs)
    mat = fe.transform(sample_logs + ["2026-10-12 09:00:00 kernel: Kernel panic - not syncing: Fatal exception 0xdeadbeef"])
    assert mat.shape == (5, 8)
    assert mat[4, 1] == 1.0  # unseen template flag on line 5
    print(f"[SMOKE TEST] Discovered {len(miner.clusters)} templates:")
    for c in miner.clusters:
        print(f"  {c.template_id} (size={c.size}): {c.template_str}")
    print("[SUCCESS] drain_log_template_miner smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drain-Style Log Template Miner")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
