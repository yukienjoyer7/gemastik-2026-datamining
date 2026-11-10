#!/usr/bin/env python3
"""
Nusantara Regional Dialect & Code-Switching Detector
====================================================
Lexical marker and morphological/phonotactic scorer for 10 Nusantara linguistic
varieties plus English code-switching:
  1. `formal_indonesian` (Bahasa Indonesia Baku)
  2. `colloquial_jakartan` (Bahasa Gaul / Betawi-Jakarta)
  3. `javanese` (Basa Jawa *Ngoko* & *Krama*)
  4. `sundanese` (Basa Sunda)
  5. `minangkabau` (Baso Minang)
  6. `balinese` (Basa Bali)
  7. `bugis_makassar` (Basa Ugi / Mangkasara / South Sulawesi particles)
  8. `banjarese` (Bahasa Banjar)
  9. `madurese` (Bhasa Madhura)
  10. `regional_malay` (Melayu Sumatera / Kalimantan / Timur)

Outputs:
  - Per-dialect probability distribution (`10` regional/Indonesian classes)
  - Dominant dialect label
  - **Code-Switching Entropy** (Shannon entropy over active language/dialect families,
    measuring how heavily a sentence mixes multiple regional/Indonesian/English vocabularies)
  - Code-switching index & English token ratio.

Usage:
    python3 00_core/indonesian_nlp/nusantara_dialect_detector.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
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
    """Configuration for Nusantara Dialect & Code-Switching Detector."""

    smoothing_prior: float = 0.15
    temperature: float = 0.7
    smoke_test: bool = False


# =============================================================================
# 1. Curated Nusantara Dialect Lexical & Phonotactic Markers
# =============================================================================

DIALECT_NAMES: Tuple[str, ...] = (
    "formal_indonesian",
    "colloquial_jakartan",
    "javanese",
    "sundanese",
    "minangkabau",
    "balinese",
    "bugis_makassar",
    "banjarese",
    "madurese",
    "regional_malay",
)

DIALECT_MARKERS: Dict[str, Dict[str, float]] = {
    "formal_indonesian": {
        "tidak": 1.5,
        "adalah": 1.8,
        "merupakan": 2.0,
        "tersebut": 1.8,
        "kepada": 1.5,
        "terhadap": 1.8,
        "sebagai": 1.5,
        "melalui": 1.5,
        "berdasarkan": 2.0,
        "pemerintah": 1.5,
        "masyarakat": 1.5,
        "bahwa": 1.6,
        "namun": 1.5,
        "oleh": 1.3,
        "karena": 1.2,
        "dengan": 1.0,
        "untuk": 1.0,
        "dalam": 1.0,
    },
    "colloquial_jakartan": {
        "gue": 2.5,
        "gw": 2.5,
        "gua": 2.2,
        "lu": 2.2,
        "lo": 2.2,
        "elo": 2.5,
        "banget": 2.0,
        "bgt": 2.0,
        "dong": 1.8,
        "deh": 1.8,
        "sih": 1.6,
        "nih": 1.6,
        "tuh": 1.5,
        "kagak": 2.5,
        "enggak": 1.8,
        "gak": 1.6,
        "udah": 1.5,
        "doang": 2.2,
        "nyokap": 2.8,
        "bokap": 2.8,
        "buset": 2.5,
        "anjir": 2.0,
        "gimana": 1.6,
        "kayak": 1.5,
        "ngapain": 2.0,
    },
    "javanese": {
        # Ngoko & Krama high-frequency markers
        "aku": 0.6,
        "kowe": 3.0,
        "sampeyan": 2.8,
        "panjenengan": 3.2,
        "kulo": 3.0,
        "kula": 2.8,
        "ora": 2.8,
        "mboten": 3.2,
        "wis": 2.5,
        "uwis": 2.8,
        "sampun": 3.0,
        "durung": 2.8,
        "dereng": 3.0,
        "piye": 3.0,
        "pripun": 3.0,
        "opo": 2.5,
        "menapa": 2.8,
        "nopo": 2.6,
        "wingi": 2.5,
        "sesuk": 2.5,
        "kabeh": 2.5,
        "sedanten": 3.0,
        "bengi": 2.2,
        "dalu": 2.2,
        "tresna": 2.8,
        "tresno": 2.8,
        "lunga": 2.5,
        "lungo": 2.5,
        "tindak": 2.2,
        "mangan": 2.5,
        "dhahar": 3.0,
        "nedha": 2.8,
        "ngombe": 2.8,
        "unjuk": 2.2,
        "apik": 2.5,
        "sae": 2.2,
        "elek": 2.5,
        "tenan": 2.5,
        "banget": 0.5,
        "monggo": 2.8,
        "mangga": 1.5,
        "matur": 2.5,
        "nuwun": 2.8,
        "suwun": 2.8,
        "nggih": 3.0,
        "inggih": 3.0,
        "wae": 2.6,
        "ae": 2.0,
        "kok": 1.2,
        "lho": 1.5,
    },
    "sundanese": {
        "abdi": 2.8,
        "urang": 2.2,
        "maneh": 3.0,
        "anjeun": 3.2,
        "sia": 2.5,
        "teu": 3.0,
        "henteu": 3.2,
        "mah": 2.5,
        "teh": 2.5,
        "tea": 2.5,
        "atuh": 3.0,
        "euy": 3.2,
        "geuning": 3.2,
        "kumaha": 3.2,
        "naon": 3.0,
        "kunaon": 3.2,
        "kamana": 2.8,
        "dimana": 1.0,
        "iraha": 3.0,
        "saha": 3.0,
        "sababaraha": 3.0,
        "ayeuna": 3.0,
        "kamari": 2.8,
        "isukan": 2.8,
        "endah": 2.2,
        "hade": 2.5,
        "goreng": 1.2,
        "dahar": 2.2,
        "tuang": 2.5,
        "leleson": 3.0,
        "hatur": 3.0,
        "nuhun": 3.2,
        "punten": 3.0,
        "wilujeng": 3.2,
        "pisan": 2.8,
        "bae": 2.0,
        "we": 2.0,
        "wae": 1.2,
    },
    "minangkabau": {
        "ambo": 3.0,
        "awak": 2.2,
        "aden": 3.0,
        "den": 2.2,
        "ang": 2.5,
        "kau": 1.5,
        "uda": 2.2,
        "uni": 2.2,
        "indak": 3.2,
        "ndak": 2.5,
        "jan": 2.5,
        "baa": 3.2,
        "apo": 2.5,
        "kama": 2.8,
        "dima": 2.8,
        "bara": 2.8,
        "sia": 1.8,
        "bilo": 2.8,
        "kini": 1.8,
        "patang": 2.5,
        "isuak": 3.0,
        "rancak": 3.2,
        "elok": 2.2,
        "buruak": 3.0,
        "bana": 3.0,
        "sangat": 0.5,
        "pai": 2.8,
        "pulang": 1.0,
        "pitih": 3.2,
        "urang": 2.0,
        "nagari": 3.0,
        "basamo": 3.0,
        "sadonyo": 3.2,
        "lai": 2.5,
        "do": 2.2,
        "mah": 1.0,
    },
    "balinese": {
        "tiang": 3.0,
        "icang": 3.2,
        "cang": 2.5,
        "ragane": 3.2,
        "cai": 3.0,
        "ci": 2.2,
        "tusing": 3.2,
        "sing": 2.5,
        "ten": 2.5,
        "nenten": 3.2,
        "sampun": 1.8,
        "suba": 3.0,
        "durung": 1.5,
        "konden": 3.2,
        "kenken": 3.2,
        "punapi": 3.2,
        "apa": 0.8,
        "napi": 3.0,
        "kija": 3.2,
        "dija": 3.2,
        "pidan": 3.0,
        "nyen": 3.0,
        "sira": 2.8,
        "jani": 3.0,
        "mangkin": 3.0,
        "mani": 2.5,
        "benjang": 3.0,
        "melah": 3.0,
        "becik": 3.0,
        "jelek": 1.2,
        "jaen": 3.2,
        "ngajeng": 3.2,
        "madaar": 3.2,
        "matur": 1.8,
        "suksma": 3.5,
        "rahajeng": 3.5,
        "semeng": 3.0,
        "bli": 2.8,
        "mbok": 2.5,
    },
    "bugis_makassar": {
        "iyya": 3.0,
        "nakke": 3.2,
        "idi": 2.5,
        "ikatte": 3.2,
        "iko": 2.8,
        "kau": 1.5,
        "de": 2.0,
        "dek": 2.2,
        "tena": 3.2,
        "tea": 1.8,
        "aga": 2.8,
        "apa": 0.8,
        "magai": 3.2,
        "angngapa": 3.2,
        "kega": 3.0,
        "kemae": 3.2,
        "siaga": 3.0,
        "ringgi": 2.5,
        "makanja": 3.2,
        "bajik": 3.0,
        "baji": 2.8,
        "ja": 1.8,
        "ji": 2.5,
        "mi": 2.5,
        "pi": 2.2,
        "ki": 2.2,
        "ko": 2.0,
        "tonji": 3.2,
        "kapang": 3.0,
        "bede": 3.0,
        "tawwa": 3.2,
        "rong": 3.0,
        "pale": 2.8,
        "daeng": 3.0,
        "andi": 2.2,
        "kurru": 3.0,
        "sumange": 3.2,
    },
    "banjarese": {
        "ulun": 3.5,
        "unda": 3.0,
        "pian": 3.5,
        "ikam": 3.5,
        "kada": 3.5,
        "kadeda": 3.5,
        "sudah": 0.8,
        "balum": 3.0,
        "aur": 2.5,
        "kayapa": 3.5,
        "kanapa": 3.0,
        "napa": 2.8,
        "kamana": 2.0,
        "dimana": 1.0,
        "bila": 2.0,
        "siapa": 0.8,
        "wahini": 3.5,
        "isuk": 2.8,
        "samalam": 3.0,
        "bungas": 3.5,
        "baik": 0.8,
        "rigat": 3.2,
        "banar": 3.2,
        "pang": 3.0,
        "am": 2.2,
        "gin": 3.0,
        "hajalah": 3.0,
        "buhan": 3.5,
        "bubuhan": 3.5,
        "tulak": 3.0,
        "bulik": 3.2,
        "guring": 3.5,
        "banyu": 2.8,
    },
    "madurese": {
        "sengko": 3.5,
        "engko": 3.5,
        "kaule": 3.5,
        "bada": 2.5,
        "ba'na": 3.5,
        "heda": 3.2,
        "sampiyan": 3.2,
        "panjhennengngan": 3.5,
        "enja": 3.2,
        "bunten": 3.5,
        "ta": 1.8,
        "tak": 1.8,
        "la": 1.8,
        "empun": 3.0,
        "gita": 3.2,
        "berra": 3.0,
        "baramma": 3.5,
        "apa": 0.8,
        "anapa": 3.0,
        "dhimma": 3.5,
        "bila": 1.8,
        "sapa": 2.0,
        "satia": 3.2,
        "laggu": 3.2,
        "bari": 3.0,
        "bagus": 1.2,
        "becce": 3.5,
        "jhubak": 3.5,
        "rong": 1.5,
        "mator": 3.2,
        "sakalangkong": 3.8,
        "trethan": 3.5,
        "tretan": 3.5,
        "kanca": 2.5,
        "tedung": 3.2,
    },
    "regional_malay": {
        "beta": 3.0,
        "kamek": 3.5,
        "awak": 2.0,
        "kitak": 3.5,
        "hang": 3.2,
        "depa": 3.2,
        "tak": 2.0,
        "indak": 1.5,
        "sik": 3.2,
        "ndak": 1.8,
        "macam": 2.5,
        "mana": 1.2,
        "bila": 1.5,
        "kenapa": 0.8,
        "pasal": 2.5,
        "budak": 2.8,
        "encik": 3.0,
        "cik": 2.5,
        "puan": 2.8,
        "seronok": 3.2,
        "best": 1.5,
        "lah": 1.8,
        "pun": 1.2,
        "kat": 2.5,
        "dekat": 1.5,
        "cakap": 2.8,
        "bising": 2.2,
        "kedai": 2.5,
        "duit": 1.5,
        "kereta": 2.2,
        "sekejap": 3.0,
        "kejap": 3.0,
    },
}

ENGLISH_CODE_SWITCH_MARKERS = {
    "which",
    "literally",
    "actually",
    "basically",
    "prefer",
    "even",
    "though",
    "because",
    "at",
    "least",
    "by",
    "the",
    "way",
    "btw",
    "imo",
    "tbh",
    "honestly",
    "meeting",
    "follow",
    "up",
    "update",
    "progress",
    "deadline",
    "issue",
    "concern",
    "point",
    "make",
    "sense",
    "worth",
    "it",
    "overthinking",
    "healing",
    "burnout",
    "toxic",
    "red",
    "flag",
}


# =============================================================================
# 2. Nusantara Dialect & Code-Switching Detector Class
# =============================================================================


class NusantaraDialectDetector:
    """
    Scores text across 10 Nusantara linguistic varieties and computes Code-Switching Entropy.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        *tuple(f"prob_{d}" for d in DIALECT_NAMES),
        "code_switching_entropy",
        "code_switching_index",
        "english_token_ratio",
    )

    def __init__(self, cfg: CFG | None = None) -> None:
        self.cfg = cfg or CFG()

    def score_text(self, text: str) -> Dict[str, object]:
        s = str(text).lower()
        tokens = re.findall(r"[a-z']+", s)
        n_toks = max(len(tokens), 1)

        raw_scores = [float(self.cfg.smoothing_prior)] * len(DIALECT_NAMES)

        for idx, d_name in enumerate(DIALECT_NAMES):
            lex = DIALECT_MARKERS[d_name]
            for tok in tokens:
                if tok in lex:
                    raw_scores[idx] += lex[tok]

            # Phonotactic / morphological bonuses
            if d_name == "formal_indonesian":
                # Formal me-...-kan / pe-...-an affixes
                affix_hits = len(
                    re.findall(r"\b(?:me|pe|ber|di|ter)[a-z]{4,}(?:kan|an|i)\b", s)
                )
                raw_scores[idx] += 0.8 * affix_hits
            elif d_name == "sundanese":
                # Characteristic Sundanese 'eu' vowel cluster (geuning, henteu, anjeun, peuyeum)
                eu_hits = len(re.findall(r"eu", s))
                raw_scores[idx] += 1.5 * eu_hits
            elif d_name == "minangkabau":
                # Characteristic Minang diphthongs 'ua', 'ia', 'ang' word endings (baa, rancak, buruak, isuak)
                minang_phon = len(re.findall(r"(?:uak|iak)\b", s))
                raw_scores[idx] += 1.5 * minang_phon
            elif d_name == "bugis_makassar":
                # Makassar/Bugis enclitics (-mi, -ji, -ki, -ko, -pi) attached to common words
                enclitic_hits = len(
                    re.findall(r"\b(?:sudah|ada|bisa|mau|itu|ini|enak|baik)(?:mi|ji|ki|pi)\b", s)
                )
                raw_scores[idx] += 2.2 * enclitic_hits

        # Temperature-scaled softmax over the 10 dialect scores
        temp = max(self.cfg.temperature, 1e-3)
        scaled = [r / temp for r in raw_scores]
        max_sc = max(scaled)
        exp_s = [math.exp(x - max_sc) for x in scaled]
        sum_exp = max(sum(exp_s), 1e-12)
        probs_list = [float(x / sum_exp) for x in exp_s]
        probs = np.array(probs_list, dtype=np.float32) if np is not None else probs_list

        dominant_idx = max(range(len(probs_list)), key=lambda i: probs_list[i])
        dominant_dialect = DIALECT_NAMES[dominant_idx]

        # Compute Code-Switching Entropy across active linguistic families + English
        family_hits = [0.0] * (len(DIALECT_NAMES) + 1)
        for idx, d_name in enumerate(DIALECT_NAMES):
            lex = DIALECT_MARKERS[d_name]
            family_hits[idx] = sum(1.0 for tok in tokens if tok in lex)
        eng_hits = sum(1.0 for tok in tokens if tok in ENGLISH_CODE_SWITCH_MARKERS)
        family_hits[-1] = eng_hits

        total_hits = sum(family_hits)
        if total_hits > 0:
            fam_dist = [h / total_hits for h in family_hits]
            cs_entropy = float(
                -sum(p * math.log(p + 1e-12) for p in fam_dist if p > 0)
            )
            max_fam_share = float(max(fam_dist))
            cs_index = 1.0 - max_fam_share
        else:
            cs_entropy = 0.0
            cs_index = 0.0

        return {
            "probabilities": {d: float(probs_list[i]) for i, d in enumerate(DIALECT_NAMES)},
            "prob_vector": probs,
            "dominant_dialect": dominant_dialect,
            "code_switching_entropy": cs_entropy,
            "code_switching_index": cs_index,
            "english_token_ratio": float(eng_hits) / n_toks,
        }

    def extract_features(self, texts: Sequence[str]) -> object:
        feats = (
            np.zeros((len(texts), len(self.FEATURE_NAMES)), dtype=np.float32)
            if np is not None
            else _FallbackMatrix(len(texts), len(self.FEATURE_NAMES))
        )
        for i, t in enumerate(texts):
            res = self.score_text(t)
            pv = res["prob_vector"]
            for j in range(len(DIALECT_NAMES)):
                feats[i, j] = float(pv[j])  # type: ignore[index]
            feats[i, len(DIALECT_NAMES)] = float(res["code_switching_entropy"])  # type: ignore[arg-type]
            feats[i, len(DIALECT_NAMES) + 1] = float(res["code_switching_index"])  # type: ignore[arg-type]
            feats[i, len(DIALECT_NAMES) + 2] = float(res["english_token_ratio"])  # type: ignore[arg-type]
        return feats


# =============================================================================
# 3. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    detector = NusantaraDialectDetector(cfg)

    samples = {
        "formal_indonesian": "Pemerintah menetapkan kebijakan tersebut berdasarkan peraturan terhadap masyarakat.",
        "colloquial_jakartan": "Gue gak tau deh lo ngapain aja dari tadi, parah banget sih.",
        "javanese": "Kowe wis mangan durung? Aku ora ngerti piye carane matur nuwun.",
        "sundanese": "Abdi teu terang kumaha ayeuna, hatur nuhun pisan atuh euy.",
        "minangkabau": "Ambo indak tau baa kini, rancak bana nagari awak basamo.",
        "balinese": "Tiang tusing nawang kenken jani, matur suksma bli rahajeng semeng.",
        "bugis_makassar": "Tena kuisseng daeng, makanja tonji kapang sudahmi pale.",
        "banjarese": "Ulun kada tahu kayapa wahini, bungas banar pian guring.",
        "madurese": "Sengko tak tao baramma satia, mator sakalangkong tretan.",
        "regional_malay": "Kamek sik tahu macam mana kitak cakap dekat kedai sekejap.",
    }

    for expected_dialect, text in samples.items():
        res = detector.score_text(text)
        assert (
            res["dominant_dialect"] == expected_dialect
        ), f"Expected {expected_dialect}, got {res['dominant_dialect']}"

    # Code-switching test (Jakartan + Javanese + English)
    mixed_text = "Gue literally bingung banget piye carane follow up meeting ini, ora make sense."
    pure_text = samples["formal_indonesian"]
    res_mixed = detector.score_text(mixed_text)
    res_pure = detector.score_text(pure_text)
    assert (
        res_mixed["code_switching_entropy"] > res_pure["code_switching_entropy"]  # type: ignore[operator]
    ), "Code-switched text must have higher code-switching entropy than monolingual text!"

    feats = detector.extract_features(list(samples.values()) + [mixed_text])
    assert feats.shape == (11, len(NusantaraDialectDetector.FEATURE_NAMES))

    print("[SMOKE TEST] Nusantara Regional Dialect & Code-Switching Detector")
    print(f"  Verified all 10 Nusantara dialects : {list(samples.keys())}")
    print(
        f"  Code-switching entropy (monolingual vs mixed): "
        f"{res_pure['code_switching_entropy']:.3f} vs {res_mixed['code_switching_entropy']:.3f}"
    )
    print(f"  Feature matrix shape               : {feats.shape}")
    print("[SUCCESS] All dialect detection and code-switching checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nusantara Dialect & Code-Switching Detector")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
