#!/usr/bin/env python3
"""
Indonesian Prosody, Syllabification & Pantun / Syair / Gurindam Analyzer
========================================================================
Deterministic Indonesian phonological syllabifier (*Pemenggalan Suku Kata* based on PUEBI/EYD)
and stanza prosody feature extractor for Indonesian literary/poetry tasks (Pantun, Syair,
Gurindam, Puisi):
  1. **Indonesian Phonological Syllabifier (`syllabify_indonesian_word`)**:
     - Protects consonant digraphs (`ng`, `ny`, `sy`, `kh`) and diphthongs (`ai`, `au`, `oi`, `ei`).
     - Splits V-CV (`ma-kan`), VC-CV (`man-di`, `bang-sa`), VCC-CV (`ins-tru-men`), and V-V hiatus (`sa-at`).
  2. **Per-Line & Per-Stanza Prosodic Features**:
     - Syllable count per line (*jumlah suku kata*), checking the classic 8–12 Pantun rule.
     - Final syllable rhyme (*rima akhir*: exact final open/closed syllable match + vowel assonance match).
     - Internal/medial rhyme (*rima tengah*: matching syllable before the caesura / middle word of each line).
     - Rhyme scheme classifier: `a-b-a-b` (Pantun), `a-a-a-a` (Syair), `a-a` (Gurindam), `a-b-b-a`, or `free`.
     - *Sampiran* (lines 1–2) vs. *Isi* (lines 3–4) phonetic & lexical similarity features.

Usage:
    python3 00_core/indonesian_nlp/prosody_pantun_puisi.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import site
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import math

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

    def __setitem__(self, idx: object, val: object) -> None:
        if isinstance(idx, tuple):
            r, c = idx
            super().__getitem__(r)[c] = float(val)  # type: ignore[arg-type]
        else:
            super().__setitem__(idx, list(val))  # type: ignore[arg-type,call-overload]


@dataclass
class CFG:
    """Configuration for Indonesian Prosody & Pantun/Syair Analyzer."""

    min_pantun_syllables: int = 8
    max_pantun_syllables: int = 12
    protect_diphthongs: bool = True
    smoke_test: bool = False


# =============================================================================
# 1. Deterministic Indonesian Phonological Syllabifier (Pemenggalan Suku Kata)
# =============================================================================

_DIGRAPH_TO_TOKEN = {
    "ng": "ŋ",
    "ny": "ɲ",
    "sy": "ʃ",
    "kh": "x",
}
_TOKEN_TO_DIGRAPH = {v: k for k, v in _DIGRAPH_TO_TOKEN.items()}

# True diphthongs in standard Indonesian syllables (excluding common -i/-an suffix boundaries)
_DIPHTHONG_TO_TOKEN = {
    "ai": "Æ",
    "au": "Å",
    "oi": "Œ",
    "ei": "Ë",
}
_TOKEN_TO_DIPHTHONG = {v: k for k, v in _DIPHTHONG_TO_TOKEN.items()}

_VOWELS = set("aiueoÆÅŒË")


def syllabify_indonesian_word(word: str, protect_diphthongs: bool = True) -> List[str]:
    """
    Splits a single Indonesian word into phonological syllables according to PUEBI rules:
      - `makan` -> `['ma', 'kan']`
      - `bangsa` -> `['bang', 'sa']`
      - `menyanyi` -> `['me', 'nya', 'nyi']`
      - `pulau` -> `['pu', 'lau']`
    """
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return []

    # 1. Encode consonant digraphs as single phoneme symbols
    s = clean
    for dg, sym in _DIGRAPH_TO_TOKEN.items():
        s = s.replace(dg, sym)

    # 2. Encode diphthongs (mostly in open final or root positions like pulau, pantai, sepoi)
    if protect_diphthongs:
        for dp, sym in _DIPHTHONG_TO_TOKEN.items():
            # Avoid merging vowel + suffix '-i' in words like 'mulai' vs 'mengenai' when followed by nothing,
            # though in Pantun prosody word-final 'ai'/'au' (pantai, pulau, kalau) counts as 1 diphthong nucleus
            s = re.sub(rf"{dp}(?=$|[^aiueo])", sym, s)

    # Find indices of all vowel nuclei
    vowel_positions = [idx for idx, ch in enumerate(s) if ch in _VOWELS]
    if len(vowel_positions) <= 1:
        return [clean]

    syllable_Encoded: List[str] = []
    start = 0
    for k in range(len(vowel_positions) - 1):
        v1 = vowel_positions[k]
        v2 = vowel_positions[k + 1]
        num_consonants = v2 - v1 - 1

        if num_consonants == 0:
            # V-V hiatus: sa-at, ma-af, bu-ah
            cut = v1 + 1
        elif num_consonants == 1:
            # V-CV: ma-kan, me-nya-nyi
            cut = v1 + 1
        elif num_consonants == 2:
            # VC-CV unless second consonant is 'r'/'l' in onset cluster (pr, kr, tr, br, dr, pl, kl)
            c1, c2 = s[v1 + 1], s[v1 + 2]
            if c1 in "pbtkgd" and c2 in "rl":
                cut = v1 + 1
            else:
                cut = v1 + 2
        else:
            # VCC-CV: ins-tru-men, bang-krut
            cut = v1 + 2

        syllable_Encoded.append(s[start:cut])
        start = cut

    syllable_Encoded.append(s[start:])

    # Decode phoneme symbols back to orthographic digraphs/diphthongs
    decoded: List[str] = []
    for syl in syllable_Encoded:
        out = syl
        for sym, dp in _TOKEN_TO_DIPHTHONG.items():
            out = out.replace(sym, dp)
        for sym, dg in _TOKEN_TO_DIGRAPH.items():
            out = out.replace(sym, dg)
        if out:
            decoded.append(out)
    return decoded


def syllabify_line(line: str, protect_diphthongs: bool = True) -> List[List[str]]:
    """Returns a list of syllabified words for a line of poetry."""
    words: List[str] = []
    for raw_tok in re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", line.lower()):
        for part in raw_tok.split("-"):
            if part:
                words.append(part)
    return [syllabify_indonesian_word(w, protect_diphthongs=protect_diphthongs) for w in words if w]


# =============================================================================
# 2. Rhyme, Assonance & Stanza Prosody Analyzer
# =============================================================================


def extract_rhyme_signature(syllable: str) -> Tuple[str, str]:
    """
    Extracts `(rime_key, vowel_nucleus)` from the final syllable of a line:
      - `rime_key`: nucleus vowel(s) + coda consonant(s), e.g. `'kan'` -> `'an'`,
        `'ti'` -> `'i'`, `'lau'` -> `'au'`, `'bang'` -> `'ang'`.
      - `vowel_nucleus`: last vowel sound (`'a'`, `'i'`, `'u'`, `'e'`, `'o'`, `'ai'`, `'au'`).
    """
    s = re.sub(r"[^a-z]", "", syllable.lower())
    if not s:
        return ("", "")

    m = re.search(r"([aiueo]+[^aiueo]*)$", s)
    rime = m.group(1) if m else s
    v_match = re.search(r"([aiueo]+)", rime)
    vowel = v_match.group(1) if v_match else rime
    return (rime, vowel)


def _rhyme_match_score(sig_a: Tuple[str, str], sig_b: Tuple[str, str]) -> float:
    """Returns 1.0 for exact rime match (`-an` == `-an`), 0.5 for vowel assonance, 0.0 otherwise."""
    if not sig_a[0] or not sig_b[0]:
        return 0.0
    if sig_a[0] == sig_b[0]:
        return 1.0
    if sig_a[1] == sig_b[1]:
        return 0.5
    return 0.0


def classify_stanza_rhyme_scheme(end_signatures: Sequence[Tuple[str, str]]) -> str:
    """
    Classifies a 2-line or 4-line stanza into canonical Indonesian poetry schemes:
      - `a-b-a-b` (Pantun)
      - `a-a-a-a` (Syair)
      - `a-a` (Gurindam / Karmina)
      - `a-b-b-a` / `a-a-b-b`
      - `free`
    """
    n = len(end_signatures)
    if n == 2:
        return "a-a" if _rhyme_match_score(end_signatures[0], end_signatures[1]) >= 0.5 else "free"
    if n == 4:
        s13 = _rhyme_match_score(end_signatures[0], end_signatures[2]) >= 0.5
        s24 = _rhyme_match_score(end_signatures[1], end_signatures[3]) >= 0.5
        s12 = _rhyme_match_score(end_signatures[0], end_signatures[1]) >= 0.5
        s34 = _rhyme_match_score(end_signatures[2], end_signatures[3]) >= 0.5
        s14 = _rhyme_match_score(end_signatures[0], end_signatures[3]) >= 0.5
        s23 = _rhyme_match_score(end_signatures[1], end_signatures[2]) >= 0.5

        if s13 and s24 and not (s12 and s34):
            return "a-b-a-b"
        if s12 and s23 and s34:
            return "a-a-a-a"
        if s14 and s23 and not s12:
            return "a-b-b-a"
        if s12 and s34 and not s23:
            return "a-a-b-b"
    return "free"


class IndonesianProsodyAnalyzer:
    """
    Extracts phonological and prosodic features for Indonesian Pantun, Syair, Gurindam, and Puisi.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "num_lines",
        "mean_syllables_per_line",
        "std_syllables_per_line",
        "pantun_8_12_compliance_ratio",
        "end_rhyme_abab_score",
        "end_rhyme_aaaa_score",
        "internal_rhyme_abab_score",
        "sampiran_isi_phonetic_sim",
        "sampiran_isi_lexical_jaccard",
        "is_valid_pantun",
        "is_valid_syair",
    )

    def __init__(self, cfg: CFG | None = None) -> None:
        self.cfg = cfg or CFG()

    def analyze_stanza(self, stanza_text: str) -> Dict[str, object]:
        lines = [
            ln.strip()
            for ln in re.split(r"[\r\n]+|(?:\s*[/|;]\s*)", str(stanza_text).strip())
            if ln.strip()
        ]
        if not lines:
            return {
                "lines": [],
                "syllable_counts": [],
                "scheme": "free",
                "features": (
                    np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
                    if np is not None
                    else [0.0] * len(self.FEATURE_NAMES)
                ),
            }

        line_words_syls = [
            syllabify_line(ln, protect_diphthongs=self.cfg.protect_diphthongs) for ln in lines
        ]
        syl_counts = [sum(len(w) for w in ws) for ws in line_words_syls]

        # Final syllable rhyme signature per line
        end_sigs: List[Tuple[str, str]] = []
        # Internal (caesura / mid-line word) final syllable signature
        mid_sigs: List[Tuple[str, str]] = []
        for ws in line_words_syls:
            flat_syls = [s for w in ws for s in w]
            if flat_syls:
                end_sigs.append(extract_rhyme_signature(flat_syls[-1]))
            else:
                end_sigs.append(("", ""))

            if len(ws) >= 2 and ws[len(ws) // 2 - 1]:
                mid_word_last_syl = ws[len(ws) // 2 - 1][-1]
                mid_sigs.append(extract_rhyme_signature(mid_word_last_syl))
            elif flat_syls:
                mid_sigs.append(extract_rhyme_signature(flat_syls[len(flat_syls) // 2]))
            else:
                mid_sigs.append(("", ""))

        scheme = classify_stanza_rhyme_scheme(end_sigs)

        n_lines = len(lines)
        mean_syl = float(sum(syl_counts) / n_lines) if syl_counts else 0.0
        std_syl = (
            float(math.sqrt(sum((c - mean_syl) ** 2 for c in syl_counts) / n_lines))
            if syl_counts
            else 0.0
        )
        in_8_12 = [
            1.0
            if self.cfg.min_pantun_syllables <= c <= self.cfg.max_pantun_syllables
            else 0.0
            for c in syl_counts
        ]
        compliance_8_12 = float(sum(in_8_12) / len(in_8_12)) if in_8_12 else 0.0

        if n_lines >= 4:
            abab_end = 0.5 * (
                _rhyme_match_score(end_sigs[0], end_sigs[2])
                + _rhyme_match_score(end_sigs[1], end_sigs[3])
            )
            aaaa_end = float(
                (
                    _rhyme_match_score(end_sigs[0], end_sigs[1])
                    + _rhyme_match_score(end_sigs[1], end_sigs[2])
                    + _rhyme_match_score(end_sigs[2], end_sigs[3])
                )
                / 3.0
            )
            abab_mid = 0.5 * (
                _rhyme_match_score(mid_sigs[0], mid_sigs[2])
                + _rhyme_match_score(mid_sigs[1], mid_sigs[3])
            )

            # Sampiran (lines 0,1) vs Isi (lines 2,3) phonetic & lexical similarity
            sampiran_text = (lines[0] + " " + lines[1]).lower()
            isi_text = (lines[2] + " " + lines[3]).lower()
            samp_vowels = re.findall(r"[aiueo]", sampiran_text)
            isi_vowels = re.findall(r"[aiueo]", isi_text)
            v_vocab = "aiueo"
            v_vec_s = [ float(samp_vowels.count(v)) for v in v_vocab ]
            v_vec_i = [ float(isi_vowels.count(v)) for v in v_vocab ]
            norm_s = math.sqrt(sum(x * x for x in v_vec_s))
            norm_i = math.sqrt(sum(x * x for x in v_vec_i))
            denom = norm_s * norm_i
            phon_sim = (
                float(sum(a * b for a, b in zip(v_vec_s, v_vec_i)) / denom)
                if denom > 0
                else 0.0
            )

            samp_toks = set(re.findall(r"[a-z]+", sampiran_text))
            isi_toks = set(re.findall(r"[a-z]+", isi_text))
            lex_jacc = float(len(samp_toks & isi_toks)) / max(len(samp_toks | isi_toks), 1)
        else:
            abab_end = 0.0
            aaaa_end = (
                _rhyme_match_score(end_sigs[0], end_sigs[1]) if n_lines == 2 else 0.0
            )
            abab_mid = 0.0
            phon_sim = 0.0
            lex_jacc = 0.0

        is_pantun = float(n_lines == 4 and scheme == "a-b-a-b" and compliance_8_12 >= 0.75)
        is_syair = float(n_lines == 4 and scheme == "a-a-a-a" and compliance_8_12 >= 0.75)

        raw_list = [
            float(n_lines),
            mean_syl,
            std_syl,
            compliance_8_12,
            abab_end,
            aaaa_end,
            abab_mid,
            phon_sim,
            lex_jacc,
            is_pantun,
            is_syair,
        ]
        feat_vec = np.array(raw_list, dtype=np.float32) if np is not None else raw_list
        return {
            "lines": lines,
            "syllable_counts": syl_counts,
            "end_signatures": end_sigs,
            "mid_signatures": mid_sigs,
            "scheme": scheme,
            "features": feat_vec,
        }

    def extract_features(self, stanzas: Sequence[str]) -> object:
        out = (
            np.zeros((len(stanzas), len(self.FEATURE_NAMES)), dtype=np.float32)
            if np is not None
            else _FallbackMatrix(len(stanzas), len(self.FEATURE_NAMES))
        )
        for i, st in enumerate(stanzas):
            out[i] = self.analyze_stanza(st)["features"]  # type: ignore[assignment,index]
        return out


# =============================================================================
# 3. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    analyzer = IndonesianProsodyAnalyzer(cfg)

    # 1. Verify syllabification on tricky digraphs & diphthongs
    assert syllabify_indonesian_word("menyanyi") == ["me", "nya", "nyi"]
    assert syllabify_indonesian_word("bangsa") == ["bang", "sa"]
    assert syllabify_indonesian_word("pulau") == ["pu", "lau"]

    # 2. Classic Indonesian Pantun (a-b-a-b, 8-12 syllables per line)
    pantun = (
        "Berburu ke padang datar\n"
        "Dapat rusa belang kaki\n"
        "Berguru kepalang ajar\n"
        "Bagai bunga kembang tak jadi"
    )
    res_pantun = analyzer.analyze_stanza(pantun)
    assert res_pantun["scheme"] == "a-b-a-b"
    assert res_pantun["features"][9] == 1.0  # is_valid_pantun

    # 3. Classic Indonesian Syair (a-a-a-a, 8-12 syllables per line)
    syair = (
        "Dengarkan tuan suatu riwayat\n"
        "Raja di desa negeri hikayat\n"
        "Guru berkata penuh nasihat\n"
        "Supaya hidup mendapat rahmat"
    )
    res_syair = analyzer.analyze_stanza(syair)
    assert res_syair["scheme"] == "a-a-a-a"
    assert res_syair["features"][10] == 1.0  # is_valid_syair

    feats = analyzer.extract_features([pantun, syair])
    assert feats.shape == (2, len(IndonesianProsodyAnalyzer.FEATURE_NAMES))

    print("[SMOKE TEST] Indonesian Prosody, Syllabifier & Pantun/Syair Analyzer")
    print(f"  Syllabify 'menyanyi' : {syllabify_indonesian_word('menyanyi')}")
    print(f"  Pantun syllable cnts : {res_pantun['syllable_counts']}, scheme={res_pantun['scheme']}")
    print(f"  Syair syllable cnts  : {res_syair['syllable_counts']}, scheme={res_syair['scheme']}")
    print("[SUCCESS] All prosody, syllabification, and rhyme checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indonesian Prosody & Pantun/Puisi Analyzer")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
