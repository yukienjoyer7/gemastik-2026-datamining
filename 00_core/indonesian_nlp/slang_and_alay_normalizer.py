#!/usr/bin/env python3
"""
Indonesian Slang, Alay, Old Spelling & Morphological Normalizer
===============================================================
Standalone Indonesian text normalizer and morphological feature extractor designed
for Gemastik Data Mining / Indonesian NLP competitions:
  1. Comprehensive Indonesian colloquial/singkatan dictionary (`yg`, `gak`/`gk`/`tdk`/
     `enggak`/`ora`/`mboten`/`teu`, `dgn`, `krn`, `bgt`, `knp`, `tp`, `sdh`/`udh`,
     `blm`, `dr`, `utk`, `pd`, `klo`/`kl`, `jd`, `bisa`/`bs`, `org`, `dlm`, `spt`,
     `tsb`, `dll`, `dsb`, `sy`/`gw`/`gue`/`aku`/`ane`, `lu`/`lo`/`kamu`/`ente`, etc.).
  2. Indonesian numeric & quote reduplication expander (`kata2` -> `kata-kata`,
     `bapak2` -> `bapak-bapak`, `anak"` -> `anak-anak`, `jalan2` -> `jalan-jalan`,
     including affixed forms like `berjalan2` -> `berjalan-jalan`, `kata2nya` -> `kata-katanya`).
  3. Old Indonesian Spelling converter (*Ejaan Van Ophuijsen & Soewandi*:
     `oe` -> `u`, `dj` -> `j`, `tj` -> `c`, `nj` -> `ny`, `sj` -> `sy`, `ch` -> `kh`,
     handling edge cases like `koerang` -> `kurang`, `djalan` -> `jalan`, `tjakap` -> `cakap`).
  4. Character elongation normalizer (`bangeeettt` -> `banget`, `mantaaaap` -> `mantap`).
  5. Rule-based Indonesian morphological affix stripper & analyzer (extracts prefixes
     `me-/mem-/men-/meng-/meny-/ber-/di-/ter-/pe-/ke-/se-` and suffixes
     `-kan/-an/-i/-nya/-lah/-kah`, returning both root stem and affix count features).

Usage:
    python3 00_core/indonesian_nlp/slang_and_alay_normalizer.py --smoke-test
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

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class _FallbackMatrix(list):
    """Lightweight 2D matrix fallback when numpy is unavailable in bare system Python."""

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
    """Configuration for Indonesian Slang, Alay & Morphological Normalizer."""

    expand_reduplication: bool = True
    convert_old_spelling: bool = True
    normalize_elongation: bool = True
    normalize_slang: bool = True
    lowercase: bool = True
    max_char_repeat: int = 2
    smoke_test: bool = False


# =============================================================================
# 1. Comprehensive Indonesian Colloquial / Singkatan Lexicon
# =============================================================================

INDONESIAN_SLANG_MAP: Dict[str, str] = {
    # Negations (Colloquial, Javanese, Sundanese, Betawi)
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "g": "tidak",
    "tdk": "tidak",
    "enggak": "tidak",
    "engga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "kagak": "tidak",
    "kaga": "tidak",
    "ora": "tidak",
    "mboten": "tidak",
    "teu": "tidak",
    "henteu": "tidak",
    "ndak": "tidak",
    "indak": "tidak",
    "tak": "tidak",
    "bkn": "bukan",
    "jgn": "jangan",
    "blm": "belum",
    "blom": "belum",
    "belom": "belum",
    # Pronouns
    "sy": "saya",
    "sya": "saya",
    "aku": "saya",
    "ak": "saya",
    "aq": "saya",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "ane": "saya",
    "beta": "saya",
    "ambo": "saya",
    "abdi": "saya",
    "kulo": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "elo": "kamu",
    "elu": "kamu",
    "km": "kamu",
    "kmu": "kamu",
    "ente": "kamu",
    "agan": "kamu",
    "sis": "kamu",
    "bro": "kamu",
    "dy": "dia",
    "doi": "dia",
    "mrk": "mereka",
    "kita": "kita",
    "kt": "kita",
    "kmi": "kami",
    # Conjunctions & Prepositions
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "sm": "sama",
    "ama": "sama",
    "krn": "karena",
    "krna": "karena",
    "karna": "karena",
    "tp": "tetapi",
    "tpi": "tetapi",
    "tapi": "tetapi",
    "utk": "untuk",
    "untk": "untuk",
    "buat": "untuk",
    "dr": "dari",
    "dri": "dari",
    "pd": "pada",
    "pda": "pada",
    "dlm": "dalam",
    "dlam": "dalam",
    "ke": "ke",
    "k": "ke",
    "di": "di",
    "d": "di",
    "klo": "kalau",
    "kl": "kalau",
    "kalo": "kalau",
    "klu": "kalau",
    "klw": "kalau",
    "jk": "jika",
    "jikalau": "jika",
    "spt": "seperti",
    "sprti": "seperti",
    "kyk": "seperti",
    "kayak": "seperti",
    "kaya": "seperti",
    "kek": "seperti",
    "spy": "supaya",
    "biar": "agar",
    "shg": "sehingga",
    "sdg": "sedang",
    "sdgkan": "sedangkan",
    "pdhl": "padahal",
    "walaupun": "meskipun",
    "walau": "meskipun",
    "meski": "meskipun",
    # Adverbs, Modals & Particles
    "bgt": "banget",
    "bngt": "banget",
    "banget": "sangat",
    "sgt": "sangat",
    "bener": "benar",
    "bnr": "benar",
    "btul": "betul",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "uda": "sudah",
    "dah": "sudah",
    "lg": "lagi",
    "lgi": "lagi",
    "bs": "bisa",
    "bsa": "bisa",
    "dpt": "dapat",
    "jd": "jadi",
    "jdi": "jadi",
    " hrs": "harus",
    "hrs": "harus",
    "kudu": "harus",
    "mesti": "harus",
    "mw": "mau",
    "mo": "mau",
    "pengen": "ingin",
    "pingin": "ingin",
    "pgn": "ingin",
    "cm": "hanya",
    "cma": "hanya",
    "cuma": "hanya",
    "cuman": "hanya",
    "doang": "saja",
    "aja": "saja",
    "aj": "saja",
    "sj": "saja",
    "emg": "memang",
    "emang": "memang",
    "mmg": "memang",
    "pasti": "pasti",
    "msh": "masih",
    "msih": "masih",
    "skrg": "sekarang",
    "skg": "sekarang",
    "bsk": "besok",
    "kmrn": "kemarin",
    "kemaren": "kemarin",
    "td": "tadi",
    "ntar": "nanti",
    "ntr": "nanti",
    "nnti": "nanti",
    "sll": "selalu",
    "slalu": "selalu",
    "srg": "sering",
    "kdg": "kadang",
    "mkn": "mungkin",
    "mgkn": "mungkin",
    # Question Words
    "knp": "kenapa",
    "knpa": "kenapa",
    "ngapain": "mengapa",
    "gmn": "bagaimana",
    "gmna": "bagaimana",
    "gimana": "bagaimana",
    "bgmn": "bagaimana",
    "kpn": "kapan",
    "dmn": "dimana",
    "dmna": "dimana",
    "brp": "berapa",
    "brpa": "berapa",
    "sp": "siapa",
    "sapa": "siapa",
    # Common Nouns, Verbs & Abbreviations
    "org": "orang",
    "orng": "orang",
    "tsb": "tersebut",
    "dll": "dan lain-lain",
    "dsb": "dan sebagainya",
    "dst": "dan seterusnya",
    "dkk": "dan kawan-kawan",
    "yth": "yang terhormat",
    "ttg": "tentang",
    "masalah": "masalah",
    "mslh": "masalah",
    "brg": "barang",
    "hrg": "harga",
    "uang": "uang",
    "duit": "uang",
    "thn": "tahun",
    "th": "tahun",
    "bln": "bulan",
    "hr": "hari",
    "hri": "hari",
    "jm": "jam",
    "mnt": "menit",
    "dtk": "detik",
    "tmn": "teman",
    "temen": "teman",
    "ank": "anak",
    "bpk": "bapak",
    "ibu": "ibu",
    "bu": "ibu",
    "pak": "bapak",
    "kak": "kakak",
    "dek": "adik",
    "negara": "negara",
    "pemerintah": "pemerintah",
    "presiden": "presiden",
    "menteri": "menteri",
    "rakyat": "rakyat",
    "masyarakat": "masyarakat",
    "tau": "tahu",
    "tw": "tahu",
    "liat": "lihat",
    "lht": "lihat",
    "dgr": "dengar",
    "denger": "dengar",
    "blg": "bilang",
    "bilang": "berkata",
    "ngomong": "berbicara",
    "kasih": "beri",
    "ksh": "beri",
    "minta": "minta",
    "mnta": "minta",
    "bikin": "buat",
    "pake": "pakai",
    "pk": "pakai",
    "pke": "pakai",
    "beli": "beli",
    "jual": "jual",
    "kirim": "kirim",
    "krm": "kirim",
    "trm": "terima",
    "mksh": "terima kasih",
    "makasih": "terima kasih",
    "trimakasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "mantul": "mantap betul",
    "baper": "bawa perasaan",
    "gabut": "gaji buta",
    "gercep": "gerak cepat",
    "mager": "malas gerak",
    "japri": "jalur pribadi",
    "komodos": "teman",
}


# =============================================================================
# 2. Reduplication, Old Spelling, Elongation & Morphological Analyzers
# =============================================================================

_REDUPLICATION_2_RE = re.compile(
    r"\b([a-zA-Z]{2,})2(nya|an|kan|lah|kah|ku|mu)?(?![a-zA-Z0-9])"
)
_REDUPLICATION_QUOTE_RE = re.compile(
    r'\b([a-zA-Z]{2,})["\u201d\u201c](nya|an|kan|lah|kah|ku|mu)?(?![a-zA-Z0-9])'
)
_PREFIX_REDUPLICATION_RE = re.compile(
    r"^(ber|ter|me|di|pe|se)([a-zA-Z]{3,})$", re.IGNORECASE
)

# Protected words where 'oe', 'dj', 'tj', 'ch' are valid modern loanwords
_OLD_SPELLING_PROTECTED = {
    "koefisien",
    "koeksistensi",
    "koersi",
    "poetri",  # handled if needed, wait: poetri IS old spelling for putri!
    "auto",
    "audio",
    "adj",
    "adjektiva",
    "chat",
    "channel",
    "champion",
    "check",
    "chef",
    "china",
    "chip",
}
_OLD_SPELLING_PROTECTED.discard("poetri")


def expand_indonesian_reduplication(text: str) -> str:
    """
    Expands Indonesian numeric (`2`) and quote (`"`) reduplication shorthand:
      - `kata2` -> `kata-kata`
      - `bapak2` -> `bapak-bapak`
      - `anak"` -> `anak-anak`
      - `jalan2` -> `jalan-jalan`
      - `berjalan2` -> `berjalan-jalan`
      - `kata2nya` -> `kata-katanya`
    """

    def _repl(match: re.Match) -> str:
        word = match.group(1)
        suffix = match.group(2) or ""
        pref_match = _PREFIX_REDUPLICATION_RE.match(word)
        if pref_match and len(pref_match.group(2)) >= 4:
            prefix, stem = pref_match.group(1), pref_match.group(2)
            return f"{prefix}{stem}-{stem}{suffix}"
        return f"{word}-{word}{suffix}"

    out = _REDUPLICATION_2_RE.sub(_repl, text)
    out = _REDUPLICATION_QUOTE_RE.sub(_repl, out)
    return out


def convert_old_indonesian_spelling(text: str) -> str:
    """
    Converts Ejaan Van Ophuijsen & Ejaan Soewandi tokens to EYD / EBI:
      - `dj` -> `j` (`djalan` -> `jalan`, `djangkar` -> `jangkar`)
      - `tj` -> `c` (`tjakap` -> `cakap`, `tjinta` -> `cinta`)
      - `nj` -> `ny` (`njata` -> `nyata`, `boenji` -> `bunyi`)
      - `sj` -> `sy` (`sjarat` -> `syarat`, `masjarakat` -> `masyarakat`)
      - `ch` -> `kh` (`chusus` -> `khusus`, `achir` -> `akhir`)
      - `oe` -> `u` (`koerang` -> `kurang`, `soedah` -> `sudah`, `boeroeng` -> `burung`)
    Preserves capitalization of the first letter.
    """

    def _convert_token(match: re.Match) -> str:
        tok = match.group(0)
        low = tok.lower()
        if low in _OLD_SPELLING_PROTECTED:
            return tok

        is_title = tok[0].isupper() and (len(tok) == 1 or tok[1:].islower())
        is_upper = tok.isupper() and len(tok) > 1

        w = low
        w = re.sub(r"dj", "j", w)
        w = re.sub(r"tj", "c", w)
        w = re.sub(r"nj", "ny", w)
        w = re.sub(r"sj", "sy", w)
        w = re.sub(r"ch(?=[aiueo]|r|l|$)", "kh", w)
        if not w.startswith("koe") or w in {"koerang", "koening", "koelit", "koetoe", "koersi"}:
            w = re.sub(r"oe", "u", w)

        if is_upper:
            return w.upper()
        if is_title and w:
            return w[0].upper() + w[1:]
        return w

    return re.sub(r"\b[a-zA-Z]{2,}\b", _convert_token, text)


def normalize_char_elongation(text: str, max_repeat: int = 1) -> str:
    """
    Collapses elongated characters common in social media Indonesian:
      `bangeeettt` -> `banget`, `mantaaaap` -> `mantap`, `haaaalllooo` -> `halo`.
    Preserves valid Indonesian double vowels/consonants (`maaf`, `saat`, `taat`, `cemooh`, `tunggu`).
    """
    valid_double = {"maaf", "saat", "taat", "manfaat", "syafaat", "koor", "koordinasi", "kooperatif"}

    def _collapse_token(match: re.Match) -> str:
        tok = match.group(0)
        low = tok.lower()
        if low in valid_double:
            return tok
        # First collapse 3+ identical chars to 1 char
        collapsed = re.sub(r"(.)\1{2,}", r"\1", tok, flags=re.IGNORECASE)
        # Also collapse trailing double vowel/consonant if not standard
        collapsed = re.sub(r"([aiueoAIUEO])\1+$", r"\1", collapsed)
        return collapsed

    return re.sub(r"\b[a-zA-Z]{3,}\b", _collapse_token, text)


def normalize_slang_tokens(text: str, slang_map: Dict[str, str] = INDONESIAN_SLANG_MAP) -> str:
    """Replaces Indonesian slang/singkatan tokens with formal standard equivalents."""

    def _replace(match: re.Match) -> str:
        tok = match.group(0)
        low = tok.lower()
        if low in slang_map:
            repl = slang_map[low]
            if tok[0].isupper() and tok[1:].islower():
                return repl.capitalize()
            return repl
        return tok

    return re.sub(r"\b[a-zA-Z]{1,15}\b", _replace, text)


# =============================================================================
# 3. Rule-Based Indonesian Morphological Affix Stripper & Feature Analyzer
# =============================================================================


class IndonesianMorphologicalAnalyzer:
    """
    Rule-based Indonesian (Sastrawi/Nazief-Adriani inspired) affix stripper and
    morphological feature extractor.
    Extracts:
      - `stem`: approximate root word
      - `prefixes`: list of stripped prefixes (`me-`, `mem-`, `men-`, `meng-`, `meny-`,
        `ber-`, `di-`, `ter-`, `pe-`, `ke-`, `se-`)
      - `suffixes`: list of stripped suffixes/particles (`-lah`, `-kah`, `-nya`, `-ku`, `-mu`,
        `-kan`, `-an`, `-i`)
    """

    PARTICLES = ("lah", "kah", "tah", "pun")
    POSSESSIVES = ("nya", "ku", "mu")
    DERIVATIONAL_SUFFIXES = ("kan", "an", "i")
    PREFIX_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
        ("meny", r"^meny([aiueo].*)$", "s"),
        ("meng", r"^meng([aiueo].*)$", ""),
        ("meng", r"^meng([ghk].*)$", ""),
        ("men", r"^men([cdjz].*)$", ""),
        ("men", r"^men([aiueo].*)$", "t"),
        ("mem", r"^mem([bfv].*)$", ""),
        ("mem", r"^mem([aiueo].*)$", "p"),
        ("me", r"^me([lmnrwy].*)$", ""),
        ("peny", r"^peny([aiueo].*)$", "s"),
        ("peng", r"^peng([aiueoghk].*)$", ""),
        ("pen", r"^pen([cdjz].*)$", ""),
        ("pem", r"^pem([bfv].*)$", ""),
        ("pe", r"^pe([lmnrwy].*)$", ""),
        ("ber", r"^ber(.*)$", ""),
        ("ter", r"^ter(.*)$", ""),
        ("di", r"^di(.*)$", ""),
        ("ke", r"^ke(.*)$", ""),
        ("se", r"^se(.*)$", ""),
    )

    def analyze_word(self, word: str) -> Dict[str, object]:
        w = re.sub(r"[^a-zA-Z]", "", word.lower())
        prefixes: List[str] = []
        suffixes: List[str] = []

        if len(w) <= 4:
            return {"word": w, "stem": w, "prefixes": prefixes, "suffixes": suffixes}

        # 1. Strip inflectional particles (-lah, -kah, -tah, -pun)
        for part in self.PARTICLES:
            if w.endswith(part) and len(w) - len(part) >= 3:
                w = w[: -len(part)]
                suffixes.append(f"-{part}")
                break

        # 2. Strip possessive pronouns (-nya, -ku, -mu)
        for poss in self.POSSESSIVES:
            if w.endswith(poss) and len(w) - len(poss) >= 3:
                w = w[: -len(poss)]
                suffixes.append(f"-{poss}")
                break

        # 3. Strip prefixes (up to 2 nested prefixes, e.g., member-, diper-)
        for _ in range(2):
            matched_pref = False
            for pref_name, pattern, restore_char in self.PREFIX_PATTERNS:
                m = re.match(pattern, w)
                if m and len(restore_char + m.group(1)) >= 3:
                    w = restore_char + m.group(1)
                    prefixes.append(f"{pref_name}-")
                    matched_pref = True
                    break
            if not matched_pref:
                break

        # 4. Strip derivational suffixes (-kan, -an, -i)
        for suf in self.DERIVATIONAL_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                suffixes.append(f"-{suf}")
                break

        return {
            "word": word,
            "stem": w,
            "prefixes": prefixes,
            "suffixes": suffixes,
        }

    def stem_sentence(self, text: str) -> str:
        tokens = text.split()
        out: List[str] = []
        for tok in tokens:
            if "-" in tok:
                parts = [str(self.analyze_word(p)["stem"]) for p in tok.split("-") if p]
                out.append("-".join(parts))
            else:
                out.append(str(self.analyze_word(tok)["stem"]))
        return " ".join(out)


# =============================================================================
# 4. End-to-End Pipeline & Numeric Feature Extractor
# =============================================================================


class IndonesianTextNormalizer:
    """
    Full normalization & morphological feature pipeline for Indonesian text.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "slang_token_ratio",
        "reduplication_count",
        "elongated_token_count",
        "old_spelling_token_count",
        "prefix_count",
        "suffix_count",
        "affix_density",
    )

    def __init__(self, cfg: CFG | None = None) -> None:
        self.cfg = cfg or CFG()
        self.morph = IndonesianMorphologicalAnalyzer()

    def normalize(self, text: str) -> str:
        s = str(text)
        if self.cfg.expand_reduplication:
            s = expand_indonesian_reduplication(s)
        if self.cfg.convert_old_spelling:
            s = convert_old_indonesian_spelling(s)
        if self.cfg.normalize_elongation:
            s = normalize_char_elongation(s)
        if self.cfg.normalize_slang:
            s = normalize_slang_tokens(s)
        if self.cfg.lowercase:
            s = s.lower()
        return re.sub(r"\s+", " ", s).strip()

    def extract_features(self, texts: Sequence[str]) -> object:
        feats = (
            np.zeros((len(texts), len(self.FEATURE_NAMES)), dtype=np.float32)
            if np is not None
            else _FallbackMatrix(len(texts), len(self.FEATURE_NAMES))
        )
        for i, raw in enumerate(texts):
            s = str(raw)
            tokens = re.findall(r"\b\w+\b", s.lower())
            n_toks = max(len(tokens), 1)

            slang_hits = sum(1 for t in tokens if t in INDONESIAN_SLANG_MAP)
            redup_hits = len(_REDUPLICATION_2_RE.findall(s)) + len(
                _REDUPLICATION_QUOTE_RE.findall(s)
            )
            elong_hits = len(re.findall(r"(.)\1{2,}", s.lower()))
            old_hits = len(re.findall(r"(?:oe|dj|tj|nj|sj)", s.lower()))

            norm_s = self.normalize(s)
            norm_tokens = re.findall(r"\b[a-z]{3,}\b", norm_s.lower())
            pref_cnt = 0
            suf_cnt = 0
            for nt in norm_tokens:
                info = self.morph.analyze_word(nt)
                pref_cnt += len(info["prefixes"])  # type: ignore[arg-type]
                suf_cnt += len(info["suffixes"])  # type: ignore[arg-type]

            feats[i, 0] = slang_hits / n_toks
            feats[i, 1] = float(redup_hits)
            feats[i, 2] = float(elong_hits)
            feats[i, 3] = float(old_hits)
            feats[i, 4] = float(pref_cnt)
            feats[i, 5] = float(suf_cnt)
            feats[i, 6] = (pref_cnt + suf_cnt) / max(len(norm_tokens), 1)
        return feats


# =============================================================================
# 5. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    normalizer = IndonesianTextNormalizer(cfg)

    # 1. Test reduplication expansion
    r_out = expand_indonesian_reduplication('bapak2 dan anak" sedang jalan2 melihat kata2nya')
    assert "bapak-bapak" in r_out and "anak-anak" in r_out and "jalan-jalan" in r_out
    assert "kata-katanya" in r_out

    # 2. Test old spelling conversion
    old_out = convert_old_indonesian_spelling("Djalan itu soedah koerang tjakap bagi masjarakat")
    assert old_out.lower() == "jalan itu sudah kurang cakap bagi masyarakat"

    # 3. Test elongation + slang normalization
    norm_out = normalizer.normalize("Gw gak bsa jalan2 krn ujan bangeeettt dgn tmn2")
    assert "saya" in norm_out and "tidak" in norm_out and "bisa" in norm_out
    assert "jalan-jalan" in norm_out and "karena" in norm_out and "teman-teman" in norm_out

    # 4. Test morphological analyzer & feature extraction
    morph_info = normalizer.morph.analyze_word("mempertanggungjawabkannya")
    assert len(morph_info["prefixes"]) >= 1 and len(morph_info["suffixes"]) >= 1
    feats = normalizer.extract_features(
        [
            "Gw gak bsa jalan2 krn bangeeettt capek",
            "Pemerintah menetapkan kebijakan pembangunan nasional secara berkelanjutan.",
        ]
    )
    assert feats.shape == (2, len(IndonesianTextNormalizer.FEATURE_NAMES))
    assert feats[0, 0] > feats[1, 0], "Colloquial sentence must have higher slang ratio!"
    assert feats[1, 6] > feats[0, 6], "Formal sentence must have higher affix density!"

    row0_preview = [round(float(feats[0, j]), 3) for j in range(len(IndonesianTextNormalizer.FEATURE_NAMES))]
    print("[SMOKE TEST] Indonesian Slang, Alay & Morphological Normalizer")
    print(f"  Reduplication expansion : {r_out}")
    print(f"  Old spelling conversion : {old_out}")
    print(f"  Full normalized output  : {norm_out}")
    print(f"  Morphological features  : {feats.shape} -> row0={row0_preview}")
    print("[SUCCESS] All slang, alay, old spelling, and morphology checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indonesian Slang & Alay Normalizer")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
