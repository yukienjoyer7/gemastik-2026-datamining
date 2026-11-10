#!/usr/bin/env python3
"""
Indonesian Online Gambling (Judol), Pinjol & Toxic Text De-Obfuscator
=====================================================================
Standalone de-obfuscator and numeric feature extractor for adversarial Indonesian
text (online gambling / *Judi Online* spam, illegal *Pinjol* spam, and censored hate speech):
  1. Strips zero-width characters (`\\u200b`, `\\u200c`, `\\u200d`, `\\ufeff`, `\\u2060`).
  2. Normalizes Unicode homoglyphs (Cyrillic/Greek/Mathematical lookalikes -> ASCII).
  3. Collapses spaced/punctuated evasion sequences (`s.l.o.t`, `s_l_o_t`, `s l o t` -> `slot`,
     `g-a-c-o-r` -> `gacor`, `j*d*l` -> `judol`).
  4. Decodes Indonesian leetspeak & censored tokens (`g4c0r` -> `gacor`, `m4xw1n` -> `maxwin`,
     `p1nj0l` -> `pinjol`, `j*d*l` -> `judol`, `zeus`, `scatter`, `jp`, `wd`, `depo`).
  5. Extracts GBDT-ready obfuscation features:
     `leetspeak_ratio`, `spaced_char_sequence_count`, `homoglyph_count`,
     `zero_width_count`, `censored_token_count`, `judol_keyword_hits`,
     `pinjol_keyword_hits`, `toxic_keyword_hits`.

Usage:
    python3 00_core/indonesian_nlp/deobfuscator_judol_toxic.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import site
import sys
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class _FallbackMatrix(list):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__([[0.0] * cols for _ in range(rows)])
        self.shape = (rows, cols)

    def __getitem__(self, idx: object) -> object:
        if isinstance(idx, tuple):
            r, c = idx
            return super().__getitem__(r)[c]
        return super().__getitem__(idx)  # type: ignore[arg-type]

    def __setitem__(self, idx: object, val: float) -> None:
        if isinstance(idx, tuple):
            r, c = idx
            super().__getitem__(r)[c] = float(val)
        else:
            super().__setitem__(idx, val)  # type: ignore[arg-type]


@dataclass
class CFG:
    """Configuration for Indonesian Judol/Pinjol/Toxic De-Obfuscator."""

    strip_zero_width: bool = True
    normalize_homoglyphs: bool = True
    collapse_spaced_chars: bool = True
    decode_leetspeak: bool = True
    smoke_test: bool = False


# =============================================================================
# 1. Zero-Width, Homoglyph & Leetspeak Tables
# =============================================================================

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u00ad]")

HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic lookalikes
    "а": "a",
    "А": "A",
    "е": "e",
    "Е": "E",
    "о": "o",
    "О": "O",
    "р": "p",
    "Р": "P",
    "с": "c",
    "С": "C",
    "у": "y",
    "У": "Y",
    "х": "x",
    "Х": "X",
    "і": "i",
    "І": "I",
    "ј": "j",
    "к": "k",
    "м": "m",
    "н": "h",
    "т": "t",
    "в": "b",
    # Greek lookalikes
    "α": "a",
    "β": "b",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    # Fullwidth ASCII lookalikes
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
}

LEETSPEAK_CHAR_MAP: Dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
}

# Explicit censored/obfuscated Indonesian domain patterns
CENSORED_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bj[\*\.\-_u0]+d[\*\.\-_o0]+l\b", re.IGNORECASE), "judol"),
    (re.compile(r"\bg[\*\.\-_a4@]+c[\*\.\-_o0]+r\b", re.IGNORECASE), "gacor"),
    (re.compile(r"\bm[\*\.\-_a4@]+x[\*\.\-_]?w[\*\.\-_i1!]+n\b", re.IGNORECASE), "maxwin"),
    (re.compile(r"\bs[\*\.\-_]+l[\*\.\-_o0]+t\b", re.IGNORECASE), "slot"),
    (re.compile(r"\bp[\*\.\-_i1!]+n[\*\.\-_]?j[\*\.\-_o0]+l\b", re.IGNORECASE), "pinjol"),
    (re.compile(r"\bs[\*\.\-_]+c[\*\.\-_a4@]+t[\*\.\-_]+t?[\*\.\-_e3]+r\b", re.IGNORECASE), "scatter"),
    (re.compile(r"\bj[\*\.\-_a4@]+c?k[\*\.\-_]?p[\*\.\-_o0]+t\b", re.IGNORECASE), "jackpot"),
    (re.compile(r"\ba[\*\.\-_n]+j[\*\.\-_i1!]+r?\b", re.IGNORECASE), "anjir"),
    (re.compile(r"\ba[\*\.\-_n]+j[\*\.\-_i1!]+n[\*\.\-_g69]\b", re.IGNORECASE), "anjing"),
    (re.compile(r"\bg[\*\.\-_o0]+b[\*\.\-_l1]+[\*\.\-_o0]+k\b", re.IGNORECASE), "goblok"),
    (re.compile(r"\bt[\*\.\-_o0]+l[\*\.\-_o0]+l\b", re.IGNORECASE), "tolol"),
    (re.compile(r"\bb[\*\.\-_a4@]+n[\*\.\-_g6]+s[\*\.\-_a4@]+t\b", re.IGNORECASE), "bangsat"),
)

# Domain keyword sets for GBDT classification features
JUDOL_KEYWORDS = {
    "judol",
    "gacor",
    "maxwin",
    "slot",
    "scatter",
    "zeus",
    "olympus",
    "pragmatic",
    "pgsoft",
    "mahjong",
    "jackpot",
    "jp",
    "wd",
    "withdraw",
    "depo",
    "deposit",
    "rtp",
    "togel",
    "toto",
    "bandar",
    "bet",
    "spin",
    "freespin",
    "bocoran",
    "pola",
    "situs",
}

PINJOL_KEYWORDS = {
    "pinjol",
    "pinjaman",
    "cair",
    "tenor",
    "bunga",
    "galbay",
    "dc",
    "limit",
    "ktp",
    "dana",
    "kredit",
    "cicilan",
    "angsuran",
    "ilegal",
    "ojk",
    "verifikasi",
}

TOXIC_KEYWORDS = {
    "anjing",
    "anjir",
    "bangsat",
    "goblok",
    "tolol",
    "bodoh",
    "idiot",
    "bego",
    "kampret",
    "bajingan",
    "keparat",
    "brengsek",
    "sampah",
    "bacot",
    "tai",
}

# Spaced or punctuated single-character evasion: e.g. "s.l.o.t", "s_l_o_t", "s l o t"
_PUNCT_SPACED_RE = re.compile(
    r"\b([a-zA-Z0-9])(?:[\.\-_\/]+([a-zA-Z0-9])){2,}\b"
)
_WHITESPACE_SPACED_RE = re.compile(
    r"(?<![a-zA-Z0-9\*_\.\-])([a-zA-Z])(?:\s+([a-zA-Z])){3,}(?![a-zA-Z0-9\*_\.\-])"
)


# =============================================================================
# 2. De-Obfuscation Functions
# =============================================================================


def strip_zero_width_chars(text: str) -> Tuple[str, int]:
    """Removes zero-width Unicode characters and returns `(cleaned_text, count)`."""
    matches = _ZERO_WIDTH_RE.findall(text)
    return _ZERO_WIDTH_RE.sub("", text), len(matches)


def normalize_unicode_homoglyphs(text: str) -> Tuple[str, int]:
    """Maps Cyrillic/Greek/Fullwidth homoglyphs to standard ASCII letters."""
    count = 0
    out_chars = []
    for ch in text:
        if ch in HOMOGLYPH_MAP:
            out_chars.append(HOMOGLYPH_MAP[ch])
            count += 1
        else:
            out_chars.append(ch)
    return "".join(out_chars), count


def collapse_spaced_and_punctuated_sequences(text: str) -> Tuple[str, int]:
    """
    Collapses tokens split by dots, underscores, hyphens, asterisks, or spaces:
      `s.l.o.t` -> `slot`, `s_l_o_t` -> `slot`, `s l o t` -> `slot`.
    """
    seq_count = 0

    def _punct_repl(m: re.Match) -> str:
        nonlocal seq_count
        seq_count += 1
        return re.sub(r"[\.\-_\/\*]+", "", m.group(0))

    def _space_repl(m: re.Match) -> str:
        nonlocal seq_count
        seq_count += 1
        return re.sub(r"\s+", "", m.group(0))

    s = _PUNCT_SPACED_RE.sub(_punct_repl, text)
    s = _WHITESPACE_SPACED_RE.sub(_space_repl, s)
    return s, seq_count


def decode_indonesian_leetspeak(text: str) -> Tuple[str, int]:
    """
    Decodes censored tokens and mixed alphanumeric leetspeak tokens (`g4c0r` -> `gacor`,
    `m4xw1n` -> `maxwin`, `p1nj0l` -> `pinjol`, `j*d*l` -> `judol`) without corrupting
    pure numbers or valid codes (e.g., `2026`, `rp15000`, `rt03`).
    """
    s = text
    leet_token_count = 0

    for pat, repl in CENSORED_PATTERNS:
        s, n_sub = pat.subn(repl, s)
        leet_token_count += n_sub

    def _decode_mixed_token(match: re.Match) -> str:
        nonlocal leet_token_count
        tok = match.group(0)
        low = tok.lower()
        # Skip pure numbers, common units, or reduplication 'kata2'
        if low.isdigit() or re.match(r"^(?:rp|rt|rw|no|uu|ps|th|id|v)\d+$", low):
            return tok
        if re.match(r"^[a-z]{2,}2(?:nya|an|kan|lah|kah)?$", low):
            return tok

        has_alpha = any(c.isalpha() for c in low)
        has_leet = any(c in LEETSPEAK_CHAR_MAP for c in low)
        if has_alpha and has_leet:
            leet_token_count += 1
            return "".join(LEETSPEAK_CHAR_MAP.get(c, c) for c in low)
        return tok

    s = re.sub(r"\b[a-zA-Z0-9@\$!]{3,}\b", _decode_mixed_token, s)
    return s, leet_token_count


# =============================================================================
# 3. End-to-End Deobfuscator & Feature Extractor Class
# =============================================================================


class JudolToxicDeobfuscator:
    """
    De-obfuscates adversarial Indonesian text and extracts numeric obfuscation features
    for downstream GBDT or neural classifiers.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "leetspeak_ratio",
        "spaced_char_sequence_count",
        "homoglyph_count",
        "zero_width_count",
        "censored_token_count",
        "judol_keyword_hits",
        "pinjol_keyword_hits",
        "toxic_keyword_hits",
    )

    def __init__(self, cfg: CFG | None = None) -> None:
        self.cfg = cfg or CFG()

    def deobfuscate_with_stats(self, text: str) -> Tuple[str, Dict[str, float]]:
        raw = str(text)
        s = raw

        zw_count = 0
        if self.cfg.strip_zero_width:
            s, zw_count = strip_zero_width_chars(s)

        homo_count = 0
        if self.cfg.normalize_homoglyphs:
            s, homo_count = normalize_unicode_homoglyphs(s)

        censored_hits = len(re.findall(r"[a-zA-Z0-9][\*\._\-]+[a-zA-Z0-9]", s))

        spaced_count = 0
        if self.cfg.collapse_spaced_chars:
            s, spaced_count = collapse_spaced_and_punctuated_sequences(s)

        leet_count = 0
        if self.cfg.decode_leetspeak:
            s, leet_count = decode_indonesian_leetspeak(s)

        clean_tokens = re.findall(r"\b[a-z0-9]+\b", s.lower())
        n_toks = max(len(clean_tokens), 1)

        judol_hits = sum(1 for t in clean_tokens if t in JUDOL_KEYWORDS)
        pinjol_hits = sum(1 for t in clean_tokens if t in PINJOL_KEYWORDS)
        toxic_hits = sum(1 for t in clean_tokens if t in TOXIC_KEYWORDS)

        stats = {
            "leetspeak_ratio": float(leet_count) / n_toks,
            "spaced_char_sequence_count": float(spaced_count),
            "homoglyph_count": float(homo_count),
            "zero_width_count": float(zw_count),
            "censored_token_count": float(censored_hits),
            "judol_keyword_hits": float(judol_hits),
            "pinjol_keyword_hits": float(pinjol_hits),
            "toxic_keyword_hits": float(toxic_hits),
        }
        return s, stats

    def deobfuscate(self, text: str) -> str:
        clean_text, _ = self.deobfuscate_with_stats(text)
        return clean_text

    def extract_features(self, texts: Sequence[str]) -> object:
        feats = (
            np.zeros((len(texts), len(self.FEATURE_NAMES)), dtype=np.float32)
            if np is not None
            else _FallbackMatrix(len(texts), len(self.FEATURE_NAMES))
        )
        for i, t in enumerate(texts):
            _, st = self.deobfuscate_with_stats(t)
            for j, name in enumerate(self.FEATURE_NAMES):
                feats[i, j] = st[name]
        return feats


# =============================================================================
# 4. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    deob = JudolToxicDeobfuscator(cfg)

    sample_judol = (
        "Info s.l.o.t g4c0r hari ini! M4xw1n bareng kakek z\u200beus, "
        "s_l_o_t s l o t j*d*l p1nj0l cair cepat pakai сyrillic о!"
    )
    clean_text, stats = deob.deobfuscate_with_stats(sample_judol)

    assert "slot" in clean_text.lower()
    assert "gacor" in clean_text.lower()
    assert "maxwin" in clean_text.lower()
    assert "judol" in clean_text.lower()
    assert "pinjol" in clean_text.lower()
    assert "zeus" in clean_text.lower()
    assert stats["zero_width_count"] >= 1.0
    assert stats["homoglyph_count"] >= 1.0
    assert stats["spaced_char_sequence_count"] >= 2.0
    assert stats["judol_keyword_hits"] >= 5.0
    assert stats["pinjol_keyword_hits"] >= 2.0

    feats = deob.extract_features([sample_judol, "Artikel berita ekonomi nasional hari ini."])
    assert feats.shape == (2, len(JudolToxicDeobfuscator.FEATURE_NAMES))
    assert feats[0, 5] > feats[1, 5]

    print("[SMOKE TEST] Indonesian Judol, Pinjol & Toxic De-Obfuscator")
    print(f"  Original obfuscated : {sample_judol}")
    print(f"  Cleaned output      : {clean_text}")
    print(f"  Obfuscation stats   : {stats}")
    print("[SUCCESS] All de-obfuscation and feature extraction checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indonesian Judol/Pinjol/Toxic De-Obfuscator")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
