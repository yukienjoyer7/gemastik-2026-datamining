#!/usr/bin/env python3
"""
Task 11 — Module: DOM, Markup (HTML/XML) & JSON Structure-Preserving Tokenizer
==============================================================================
Parses raw HTML/XML fragments and JSON/Key-Value dumps without losing structural
hierarchy:
  1. Extracts ancestor tag hierarchy paths (`html>body>div.price`), attribute
     names (`attr:href`, `attr:onclick`), and depth statistics.
  2. Extracts flattened JSON key paths (`user.profile.role`) and primitive value
     type markers (`json_type:str`, `json_type:num`, `json_type:bool`).
  3. Serializes clean, structure-explicit strings (`[TAG:div.price] Rp 50.000 [/TAG]`)
     ready for both TF-IDF vectorizers and Transformer tokenizers (`DeBERTa` / `CodeBERT`).

Usage:
    python modules/dom_and_markup_tokenizer.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import site
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Configuration for DOM and Markup Structure-Preserving Tokenizer."""

    max_tag_depth: int = 12
    include_class_in_path: bool = True
    include_id_in_path: bool = False
    max_classes_per_tag: int = 2
    emit_closing_tags: bool = True
    smoke_test: bool = False


class _SafeDOMStreamParser(HTMLParser):
    """Fault-tolerant HTML/XML stream parser that builds tag paths and serialized tokens."""

    def __init__(self, cfg: CFG) -> None:
        super().__init__(convert_charrefs=True)
        self.cfg = cfg
        self.stack: List[str] = []
        self.tag_paths: List[str] = []
        self.attr_tokens: List[str] = []
        self.serialized_parts: List[str] = []
        self.text_segments: List[str] = []
        self.max_depth_seen: int = 0
        self.total_tags: int = 0

    def _format_tag_node(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> str:
        node = tag.lower().strip()
        attr_dict = {k.lower(): (v or "") for k, v in attrs if k}
        if self.cfg.include_class_in_path and "class" in attr_dict:
            classes = [c.strip() for c in attr_dict["class"].split() if c.strip()]
            if classes:
                node += "." + ".".join(classes[: self.cfg.max_classes_per_tag])
        if self.cfg.include_id_in_path and "id" in attr_dict and attr_dict["id"].strip():
            node += f"#{attr_dict['id'].strip()[:20]}"
        return node

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.total_tags += 1
        node = self._format_tag_node(tag, attrs)
        if len(self.stack) < self.cfg.max_tag_depth:
            self.stack.append(node)
        self.max_depth_seen = max(self.max_depth_seen, len(self.stack))
        path_str = ">".join(self.stack)
        self.tag_paths.append(f"dom_path:{path_str}")

        attr_names = []
        for k, v in attrs:
            if not k:
                continue
            k_low = k.lower().strip()
            self.attr_tokens.append(f"attr:{k_low}")
            attr_names.append(k_low)
            if v:
                v_low = v.strip().lower()
                if v_low.startswith("javascript:"):
                    self.attr_tokens.append("attr_val:javascript_uri")
                elif v_low.startswith(("http://", "https://")):
                    self.attr_tokens.append("attr_val:http_uri")

        attr_suffix = (" " + " ".join(f"@{a}" for a in attr_names[:4])) if attr_names else ""
        self.serialized_parts.append(f"[TAG:{node}{attr_suffix}]")

    def handle_endtag(self, tag: str) -> None:
        tag_low = tag.lower().strip()
        if self.stack:
            for idx in range(len(self.stack) - 1, -1, -1):
                if self.stack[idx].split(".")[0].split("#")[0] == tag_low:
                    popped = self.stack[idx]
                    del self.stack[idx:]
                    if self.cfg.emit_closing_tags:
                        self.serialized_parts.append(f"[/TAG:{popped}]")
                    break

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.text_segments.append(cleaned)
            self.serialized_parts.append(cleaned)


def parse_html_or_xml_structure(raw_text: str, cfg: Optional[CFG] = None) -> Dict[str, Any]:
    """
    Parse raw HTML/XML string into structural tokens, ancestor paths, and a
    Transformer-friendly serialized string (`[TAG:div.price @class] Rp 50.000 [/TAG:div.price]`).
    """
    cfg = cfg or CFG()
    text = str(raw_text or "")
    parser = _SafeDOMStreamParser(cfg)
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        pass

    return {
        "tag_paths": parser.tag_paths,
        "attr_tokens": parser.attr_tokens,
        "serialized_text": " ".join(parser.serialized_parts) if parser.serialized_parts else text,
        "plain_text": " ".join(parser.text_segments),
        "max_depth": parser.max_depth_seen,
        "total_tags": parser.total_tags,
    }


def _flatten_json_obj(
    obj: Any,
    prefix: str = "",
    key_paths: Optional[List[str]] = None,
    type_tokens: Optional[List[str]] = None,
    serialized: Optional[List[str]] = None,
    depth: int = 0,
    max_depth: int = 12,
) -> int:
    if key_paths is None:
        key_paths = []
    if type_tokens is None:
        type_tokens = []
    if serialized is None:
        serialized = []

    max_d = depth
    if depth > max_depth:
        return max_d

    if isinstance(obj, dict):
        for k, v in obj.items():
            k_clean = re.sub(r"\s+", "_", str(k).strip())
            path = f"{prefix}.{k_clean}" if prefix else k_clean
            key_paths.append(f"json_key:{path}")
            serialized.append(f"[KEY:{path}]")
            d = _flatten_json_obj(v, path, key_paths, type_tokens, serialized, depth + 1, max_depth)
            max_d = max(max_d, d)
    elif isinstance(obj, list):
        type_tokens.append(f"json_type:{prefix or 'root'}:list_len_{min(len(obj), 10)}")
        for item in obj[:16]:
            d = _flatten_json_obj(
                item, f"{prefix}[]", key_paths, type_tokens, serialized, depth + 1, max_depth
            )
            max_d = max(max_d, d)
    elif isinstance(obj, bool):
        type_tokens.append(f"json_type:{prefix}:bool")
        serialized.append(f"[VAL:BOOL:{str(obj).lower()}]")
    elif obj is None:
        type_tokens.append(f"json_type:{prefix}:null")
        serialized.append("[VAL:NULL]")
    elif isinstance(obj, (int, float)):
        type_tokens.append(f"json_type:{prefix}:num")
        serialized.append(f"[VAL:NUM:{obj}]")
    else:
        type_tokens.append(f"json_type:{prefix}:str")
        val_str = " ".join(str(obj).split())
        serialized.append(f"[VAL:{val_str}]")

    return max_d


_REGEX_KV_PAIR = re.compile(
    r"""(?:"([a-zA-Z0-9_.\-]+)"|([a-zA-Z0-9_.\-]+))\s*[:=]\s*("[^"]*"|'[^']*'|[^\s,;}\]]+)"""
)


def parse_json_or_kv_structure(raw_text: str, cfg: Optional[CFG] = None) -> Dict[str, Any]:
    """
    Parse raw JSON or semi-structured Key-Value logs into flattened key paths
    (`user.profile.role`), value type tokens, and `[KEY:...] [VAL:...]` sequences.
    Falls back to regex key-value extraction if JSON is truncated or malformed.
    """
    cfg = cfg or CFG()
    text = str(raw_text or "").strip()
    key_paths: List[str] = []
    type_tokens: List[str] = []
    serialized: List[str] = []
    max_depth = 0

    parsed_ok = False
    if ("{" in text and "}" in text) or (text.startswith("[") and text.endswith("]")):
        start = text.find("{") if "{" in text else text.find("[")
        end = text.rfind("}") + 1 if "}" in text else text.rfind("]") + 1
        candidate = text[start:end] if end > start else text
        try:
            obj = json.loads(candidate)
            max_depth = _flatten_json_obj(
                obj,
                prefix="",
                key_paths=key_paths,
                type_tokens=type_tokens,
                serialized=serialized,
                depth=1,
                max_depth=cfg.max_tag_depth,
            )
            parsed_ok = True
        except Exception:
            parsed_ok = False

    if not parsed_ok:
        for match in _REGEX_KV_PAIR.finditer(text):
            k = (match.group(1) or match.group(2) or "").strip()
            v = (match.group(3) or "").strip().strip("\"'")
            if k:
                key_paths.append(f"kv_key:{k}")
                serialized.append(f"[KEY:{k}] [VAL:{v}]")
        max_depth = max(1, text.count("{") + text.count("["))

    return {
        "key_paths": key_paths,
        "type_tokens": type_tokens,
        "serialized_text": " ".join(serialized) if serialized else text,
        "max_depth": max_depth,
        "num_keys": len(key_paths),
        "is_valid_json": parsed_ok,
    }


def serialize_structure_preserving_text(raw_text: str, cfg: Optional[CFG] = None) -> str:
    """
    Unified serializer that converts raw HTML/DOM, JSON/KV, or hybrid strings into
    a single enriched token stream containing both structural tokens (`dom_path:...`,
    `json_key:...`, `[TAG:...]`, `[KEY:...]`) and original content.
    """
    cfg = cfg or CFG()
    text = str(raw_text or "")
    tokens: List[str] = []

    if "<" in text and ">" in text:
        dom_info = parse_html_or_xml_structure(text, cfg)
        tokens.extend(dom_info["tag_paths"])
        tokens.extend(dom_info["attr_tokens"])
        tokens.append(dom_info["serialized_text"])

    if ("{" in text and ":" in text) or ("=" in text and len( _REGEX_KV_PAIR.findall(text) ) >= 2):
        kv_info = parse_json_or_kv_structure(text, cfg)
        tokens.extend(kv_info["key_paths"])
        tokens.extend(kv_info["type_tokens"])
        tokens.append(kv_info["serialized_text"])

    if not tokens:
        return text
    return " ".join(tokens)


class DOMAndMarkupStructuralTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer extracting dense structural summary features
    from HTML/XML/JSON strings:
      0. log1p(total_html_tags)
      1. max_dom_depth
      2. log1p(num_html_attributes)
      3. log1p(num_json_or_kv_keys)
      4. max_json_nesting_depth
      5. is_valid_json_flag
      6. tag_to_text_char_ratio
    """

    def __init__(self, max_tag_depth: int = 12) -> None:
        self.max_tag_depth = max_tag_depth

    def fit(self, X: Sequence[str], y: Optional[np.ndarray] = None) -> "DOMAndMarkupStructuralTransformer":
        return self

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        cfg = CFG(max_tag_depth=self.max_tag_depth)
        feats = np.zeros((len(X), 7), dtype=np.float32)
        for i, raw in enumerate(X):
            text = str(raw or "")
            dom = parse_html_or_xml_structure(text, cfg) if ("<" in text and ">" in text) else {
                "total_tags": 0,
                "max_depth": 0,
                "attr_tokens": [],
                "plain_text": text,
            }
            kv = parse_json_or_kv_structure(text, cfg)
            plain_len = len(dom["plain_text"])
            total_len = max(len(text), 1)

            feats[i, 0] = np.log1p(dom["total_tags"])
            feats[i, 1] = float(dom["max_depth"])
            feats[i, 2] = np.log1p(len(dom["attr_tokens"]))
            feats[i, 3] = np.log1p(kv["num_keys"])
            feats[i, 4] = float(kv["max_depth"])
            feats[i, 5] = 1.0 if kv["is_valid_json"] else 0.0
            feats[i, 6] = max(0.0, float(total_len - plain_len)) / total_len

        return sparse.csr_matrix(feats)


def _run_smoke_test() -> None:
    sample_html = (
        '<div class="product-card featured" id="item-99">'
        '<a href="https://shop.id/p/99" onclick="javascript:stealCookie()">'
        '<span class="price">Rp 125.000</span></a></div>'
    )
    dom_res = parse_html_or_xml_structure(sample_html)
    assert "dom_path:div.product-card.featured>a>span.price" in dom_res["tag_paths"]
    assert "attr:onclick" in dom_res["attr_tokens"]
    assert "attr_val:javascript_uri" in dom_res["attr_tokens"]
    assert "[TAG:span.price @class] Rp 125.000 [/TAG:span.price]" in dom_res["serialized_text"]

    sample_json = '{"event": "auth_login", "user": {"id": 42, "role": "admin", "mfa": false}}'
    json_res = parse_json_or_kv_structure(sample_json)
    assert json_res["is_valid_json"] is True
    assert "json_key:user.role" in json_res["key_paths"]
    assert "[KEY:user.role] [VAL:admin]" in json_res["serialized_text"]

    tf = DOMAndMarkupStructuralTransformer()
    mat = tf.fit_transform([sample_html, sample_json])
    assert mat.shape == (2, 7)
    print(f"[SMOKE TEST] DOM paths: {dom_res['tag_paths']}")
    print(f"[SMOKE TEST] JSON keys: {json_res['key_paths']}")
    print(f"[SMOKE TEST] Structural feature matrix shape: {mat.shape}")
    print("[SUCCESS] dom_and_markup_tokenizer smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DOM, Markup & JSON Structure-Preserving Tokenizer")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
