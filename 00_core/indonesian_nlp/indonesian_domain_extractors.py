#!/usr/bin/env python3
"""
Indonesian Domain Regex & Rule Extractors (Addresses, Rupiah, Dates, Legal)
===========================================================================
High-precision deterministic extractors and numeric feature builders for
Indonesian domain entities:
  1. **Indonesian Administrative Hierarchy (`extract_indonesian_addresses`)**:
     - `Provinsi`, `Kabupaten`/`Kab.`, `Kota`, `Kecamatan`/`Kec.`, `Kelurahan`/`Kel.`, `Desa`,
       `RT/RW` (`RT 003 / RW 005`), street prefixes (`Jl.`, `Jln.`, `Jalan`, `Gg.`, `Komp.`),
       and 5-digit Indonesian postal codes (*Kode Pos*).
  2. **Indonesian Currency (`parse_rupiah_amounts`)**:
     - Parses `Rp 15.000,-`, `Rp1.500.000,00`, `1,5 juta`, `250 rb`, `2,4 miliar`, `3 triliun`
       into normalized float Rupiah values (`15000.0`, `1500000.0`, `2400000000.0`, etc.).
  3. **Indonesian Dates & Timezones (`extract_indonesian_dates_and_times`)**:
     - Parses Indonesian month names (`Januari`..`Desember`, `Jan`..`Des`), days (`Senin`..`Minggu`),
       and timezones (`WIB`, `WITA`, `WIT`) into normalized `(YYYY, MM, DD)` tuples.
  4. **Indonesian Legal (*Putusan MA / Peraturan*) Markers (`extract_indonesian_legal_markers`)**:
     - Extracts `Pasal`, `Ayat`, `UU No. ... Tahun ...`, `PP`, `Perpres`, `Permen`, and splits
       court decisions (*Putusan Pengadilan / MA*) into `Menimbang`, `Mengingat`,
       `Memperhatikan`, and `Mengadili` sections.

Usage:
    python3 00_core/indonesian_nlp/indonesian_domain_extractors.py --smoke-test
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
    """Configuration for Indonesian Domain Extractors."""

    extract_addresses: bool = True
    extract_currency: bool = True
    extract_dates: bool = True
    extract_legal: bool = True
    smoke_test: bool = False


# =============================================================================
# 1. Indonesian Administrative Hierarchy & Address Extractor
# =============================================================================

_PROVINSI_RE = re.compile(
    r"\b(?:Provinsi|Prov\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",
    re. IGNORECASE,
)
_KABUPATEN_RE = re.compile(
    r"\b(?:Kabupaten|Kab\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)
_KOTA_RE = re.compile(
    r"\bKota(?:\s+Administrasi)?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)
_KECAMATAN_RE = re.compile(
    r"\b(?:Kecamatan|Kec\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)
_KELURAHAN_DESA_RE = re.compile(
    r"\b(?:Kelurahan|Kel\.?|Desa|Ds\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
)
_RT_RW_RE = re.compile(
    r"\bRT\.?\s*(\d{1,3})\s*(?:[/,\-\s]+\s*RW\.?\s*(\d{1,3}))?",
    re.IGNORECASE,
)
_STREET_RE = re.compile(
    r"\b(?:Jl\.|Jln\.|Jalan|Gg\.|Gang|Komp\.|Komplek|Perum\.|Perumahan)\s+[A-Za-z0-9\.\-/]+(?:\s+[A-Za-z0-9\.\-/]+){0,4}",
    re.IGNORECASE,
)
_POSTAL_CODE_RE = re.compile(r"\b([1-9]\d{4})\b")


def extract_indonesian_addresses(text: str) -> Dict[str, object]:
    """Extracts structured Indonesian administrative hierarchy and street markers."""
    s = str(text)
    rt_rw_matches = [(m.group(1), m.group(2) or "") for m in _RT_RW_RE.finditer(s)]
    return {
        "provinsi": [m.group(1).strip() for m in _PROVINSI_RE.finditer(s)],
        "kabupaten": [m.group(1).strip() for m in _KABUPATEN_RE.finditer(s)],
        "kota": [m.group(1).strip() for m in _KOTA_RE.finditer(s)],
        "kecamatan": [m.group(1).strip() for m in _KECAMATAN_RE.finditer(s)],
        "kelurahan_desa": [m.group(1).strip() for m in _KELURAHAN_DESA_RE.finditer(s)],
        "rt_rw": rt_rw_matches,
        "streets": [m.group(0).strip() for m in _STREET_RE.finditer(s)],
        "postal_codes": [m.group(1) for m in _POSTAL_CODE_RE.finditer(s)],
    }


# =============================================================================
# 2. Indonesian Currency (Rupiah) Parser
# =============================================================================

_UNIT_MULTIPLIERS: Dict[str, float] = {
    "rb": 1e3,
    "ribu": 1e3,
    "k": 1e3,
    "jt": 1e6,
    "juta": 1e6,
    "jutaan": 1e6,
    "m": 1e9,
    "miliar": 1e9,
    "milyar": 1e9,
    "t": 1e12,
    "triliun": 1e12,
    "trilyun": 1e12,
}

# Pattern A: Explicit/implicit number with Indonesian multiplier word (e.g., "Rp 1,5 juta", "250 rb", "2,4 miliar")
_RUPIAH_WITH_UNIT_RE = re.compile(
    r"(?:Rp\.?\s*|IDR\s*)?(\d+(?:[.,]\d+)?)\s*(ribu|rb|juta|jutaan|jt|miliar|milyar|triliun|trilyun)\b(?:\s*rupiah)?",
    re.IGNORECASE,
)

# Pattern B: Formatted Rupiah with 'Rp' prefix and thousand-separator dots (e.g., "Rp 15.000,-", "Rp1.500.000,00")
_RUPIAH_FORMATTED_RE = re.compile(
    r"(?:Rp\.?|IDR)\s*(\d{1,3}(?:\.\d{3})+(?:,\d{1,2}|,-)?|\d+(?:,\d{1,2}|,-)?)",
    re.IGNORECASE,
)


def parse_rupiah_amounts(text: str) -> List[float]:
    """
    Parses Indonesian currency strings into normalized float Rupiah values:
      - `'Rp 15.000,-'` -> `15000.0`
      - `'Rp1.500.000,00'` -> `1500000.0`
      - `'1,5 juta'` -> `1500000.0`
      - `'250 rb'` -> `250000.0`
      - `'2,4 miliar'` -> `2400000000.0`
      - `'3 triliun'` -> `3000000000000.0`
    """
    s = str(text)
    amounts: List[Tuple[int, float]] = []
    used_spans: List[Tuple[int, int]] = []

    # 1. Match amounts with explicit Indonesian multiplier units (rb, juta, miliar, triliun)
    for m in _RUPIAH_WITH_UNIT_RE.finditer(s):
        raw_num = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        try:
            val = float(raw_num) * _UNIT_MULTIPLIERS[unit]
            amounts.append((m.start(), val))
            used_spans.append((m.start(), m.end()))
        except ValueError:
            continue

    # 2. Match formatted Rp amounts not already covered by a unit match
    for m in _RUPIAH_FORMATTED_RE.finditer(s):
        start, end = m.start(), m.end()
        if any(not (end <= u_s or start >= u_e) for u_s, u_e in used_spans):
            continue
        raw_str = m.group(1)
        # Strip trailing ',-' or ',00' decimal cents
        raw_str = re.sub(r",(?:-|\d{1,2})$", "", raw_str)
        # Remove Indonesian thousand-separator dots
        clean_digits = raw_str.replace(".", "")
        try:
            val = float(clean_digits)
            amounts.append((start, val))
        except ValueError:
            continue

    amounts.sort(key=lambda x: x[0])
    return [val for _, val in amounts]


# =============================================================================
# 3. Indonesian Dates, Days & Timezones Extractor
# =============================================================================

INDONESIAN_MONTHS: Dict[str, int] = {
    "januari": 1,
    "jan": 1,
    "februari": 2,
    "pebruari": 2,
    "feb": 2,
    "maret": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "mei": 5,
    "juni": 6,
    "jun": 6,
    "juli": 7,
    "jul": 7,
    "agustus": 8,
    "agu": 8,
    "ags": 8,
    "agst": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "oktober": 10,
    "okt": 10,
    "november": 11,
    "nopember": 11,
    "nov": 11,
    "desember": 12,
    "des": 12,
}

INDONESIAN_DAYS: Dict[str, int] = {
    "senin": 1,
    "selasa": 2,
    "rabu": 3,
    "kamis": 4,
    "jumat": 5,
    "jum'at": 5,
    "sabtu": 6,
    "minggu": 7,
    "ahad": 7,
}

_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Januari|Februari|Pebruari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Nopember|Desember|Jan|Feb|Mar|Apr|Jun|Jul|Agu|Ags|Agst|Sep|Sept|Okt|Nov|Des)\.?\s+(\d{4})\b",
    re.IGNORECASE,
)
_DAY_RE = re.compile(
    r"\b(Senin|Selasa|Rabu|Kamis|Jumat|Jum'at|Sabtu|Minggu|Ahad)\b",
    re.IGNORECASE,
)
_TIMEZONE_RE = re.compile(r"\b(WIB|WITA|WIT)\b")
_TIME_CLOCK_RE = re.compile(
    r"\b(?:pukul|pkl\.?|jam)?\s*(\d{1,2})[:.](\d{2})\s*(WIB|WITA|WIT)\b",
    re.IGNORECASE,
)


def extract_indonesian_dates_and_times(text: str) -> Dict[str, object]:
    """Extracts normalized Indonesian dates `(YYYY, MM, DD)`, day names, and timezones (`WIB`/`WITA`/`WIT`)."""
    s = str(text)
    dates: List[Tuple[int, int, int]] = []
    for m in _DATE_RE.finditer(s):
        day = int(m.group(1))
        month = INDONESIAN_MONTHS.get(m.group(2).lower(), 0)
        year = int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            dates.append((year, month, day))

    days = [m.group(1).lower().replace("'", "") for m in _DAY_RE.finditer(s)]
    timezones = [m.group(1).upper() for m in _TIMEZONE_RE.finditer(s)]
    timestamps = [
        f"{int(m.group(1)):02d}:{m.group(2)} {m.group(3).upper()}"
        for m in _TIME_CLOCK_RE.finditer(s)
    ]
    return {
        "dates": dates,
        "days": days,
        "timezones": timezones,
        "timestamps": timestamps,
    }


# =============================================================================
# 4. Indonesian Legal (Putusan MA / Peraturan) Marker & Section Extractor
# =============================================================================

_PASAL_RE = re.compile(r"\bPasal\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
_AYAT_RE = re.compile(r"\bAyat\s*\((\d+[a-z]?)\)", re.IGNORECASE)
_UU_RE = re.compile(
    r"\b(?:UU|Undang-Undang|PP|Perpres|Permen|Perda|Perpu)\s+(?:RI\s+)?(?:Nomor|No\.?)\s*(\d+)\s+Tahun\s+(\d{4})\b",
    re.IGNORECASE,
)
_PUTUSAN_SECTIONS_RE = re.compile(
    r"\b(Menimbang|Mengingat|Memperhatikan|Mengadili|Menetapkan|Memutuskan)\s*:",
    re.IGNORECASE,
)


def extract_indonesian_legal_markers(text: str) -> Dict[str, object]:
    """
    Extracts `Pasal`, `Ayat`, `UU No. ... Tahun ...` citations and segments legal documents
    into `Menimbang`, `Mengingat`, `Memperhatikan`, and `Mengadili` sections.
    """
    s = str(text)
    pasal_list = [m.group(1) for m in _PASAL_RE.finditer(s)]
    ayat_list = [m.group(1) for m in _AYAT_RE.finditer(s)]
    uu_list = [(m.group(1), m.group(2)) for m in _UU_RE.finditer(s)]

    sections: Dict[str, str] = {}
    matches = list(_PUTUSAN_SECTIONS_RE.finditer(s))
    for i, m in enumerate(matches):
        sec_name = m.group(1).capitalize()
        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        sec_body = s[start_pos:end_pos].strip()
        sections[sec_name] = (
            (sections[sec_name] + " " + sec_body).strip()
            if sec_name in sections
            else sec_body
        )

    return {
        "pasal": pasal_list,
        "ayat": ayat_list,
        "undang_undang": uu_list,
        "sections": sections,
    }


# =============================================================================
# 5. Unified Tabular Feature Builder
# =============================================================================


class IndonesianDomainFeatureExtractor:
    """
    Extracts a 14-dimensional numeric feature matrix from Indonesian texts capturing
    administrative addresses, Rupiah log-amounts, dates/timezones, and legal citations.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "address_entity_count",
        "has_rt_rw",
        "has_street_prefix",
        "rupiah_mention_count",
        "rupiah_max_log1p",
        "rupiah_sum_log1p",
        "date_mention_count",
        "day_mention_count",
        "has_wib",
        "has_wita",
        "has_wit",
        "pasal_count",
        "uu_citation_count",
        "legal_section_count",
    )

    def __init__(self, cfg: CFG | None = None) -> None:
        self.cfg = cfg or CFG()

    def extract_features(self, texts: Sequence[str]) -> object:
        feats = (
            np.zeros((len(texts), len(self.FEATURE_NAMES)), dtype=np.float32)
            if np is not None
            else _FallbackMatrix(len(texts), len(self.FEATURE_NAMES))
        )
        for i, raw in enumerate(texts):
            s = str(raw)
            addr = extract_indonesian_addresses(s)
            rp_vals = parse_rupiah_amounts(s)
            dt = extract_indonesian_dates_and_times(s)
            leg = extract_indonesian_legal_markers(s)

            addr_cnt = (
                len(addr["provinsi"])  # type: ignore[arg-type]
                + len(addr["kabupaten"])  # type: ignore[arg-type]
                + len(addr["kota"])  # type: ignore[arg-type]
                + len(addr["kecamatan"])  # type: ignore[arg-type]
                + len(addr["kelurahan_desa"])  # type: ignore[arg-type]
            )
            feats[i, 0] = float(addr_cnt)
            feats[i, 1] = float(len(addr["rt_rw"]) > 0)  # type: ignore[arg-type]
            feats[i, 2] = float(len(addr["streets"]) > 0)  # type: ignore[arg-type]
            feats[i, 3] = float(len(rp_vals))
            feats[i, 4] = math.log1p(max(rp_vals)) if rp_vals else 0.0
            feats[i, 5] = math.log1p(sum(rp_vals)) if rp_vals else 0.0
            feats[i, 6] = float(len(dt["dates"]))  # type: ignore[arg-type]
            feats[i, 7] = float(len(dt["days"]))  # type: ignore[arg-type]
            tzs = dt["timezones"]  # type: ignore[assignment]
            feats[i, 8] = float("WIB" in tzs)  # type: ignore[operator]
            feats[i, 9] = float("WITA" in tzs)  # type: ignore[operator]
            feats[i, 10] = float("WIT" in tzs)  # type: ignore[operator]
            feats[i, 11] = float(len(leg["pasal"]))  # type: ignore[arg-type]
            feats[i, 12] = float(len(leg["undang_undang"]))  # type: ignore[arg-type]
            feats[i, 13] = float(len(leg["sections"]))  # type: ignore[arg-type]
        return feats


# =============================================================================
# 6. Smoke Test
# =============================================================================


def run_smoke_test(cfg: CFG) -> None:
    # 1. Test Indonesian Administrative Address Extractor
    sample_addr = (
        "Alamat pengiriman: Jl. Merdeka Raya No. 45, RT 003 / RW 005, "
        "Kel. Gambir, Kec. Gambir, Kota Jakarta Pusat, Provinsi DKI Jakarta 10110."
    )
    addr_info = extract_indonesian_addresses(sample_addr)
    assert len(addr_info["rt_rw"]) == 1 and addr_info["rt_rw"][0] == ("003", "005")  # type: ignore[index]
    assert len(addr_info["streets"]) >= 1  # type: ignore[arg-type]
    assert len(addr_info["kecamatan"]) == 1 and len(addr_info["kelurahan_desa"]) == 1  # type: ignore[arg-type]

    # 2. Test Rupiah Currency Parser
    sample_rp = (
        "Biaya admin Rp 15.000,- dengan modal Rp1.500.000,00, pinjaman 1,5 juta, "
        "diskon 250 rb, investasi 2,4 miliar hingga 3 triliun."
    )
    rp_vals = parse_rupiah_amounts(sample_rp)
    expected_rp = [15_000.0, 1_500_000.0, 1_500_000.0, 250_000.0, 2_400_000_000.0, 3_000_000_000_000.0]
    assert rp_vals == expected_rp, f"Expected {expected_rp}, got {rp_vals}"

    # 3. Test Indonesian Dates & Timezones
    sample_dt = "Sidang digelar pada hari Senin, 17 Agustus 2026 pukul 09.30 WIB dan 10:30 WITA."
    dt_info = extract_indonesian_dates_and_times(sample_dt)
    assert dt_info["dates"] == [(2026, 8, 17)]
    assert "senin" in dt_info["days"]  # type: ignore[operator]
    assert "WIB" in dt_info["timezones"] and "WITA" in dt_info["timezones"]  # type: ignore[operator]

    # 4. Test Indonesian Legal Putusan Extractor
    sample_legal = (
        "Menimbang: bahwa terdakwa didakwa melanggar Pasal 27 Ayat (3). "
        "Mengingat: UU No. 11 Tahun 2008 dan PP Nomor 71 Tahun 2019. "
        "Memperhatikan: bukti persidangan. "
        "Mengadili: menyatakan terdakwa bersalah."
    )
    leg_info = extract_indonesian_legal_markers(sample_legal)
    assert leg_info["pasal"] == ["27"]
    assert leg_info["ayat"] == ["3"]
    assert ("11", "2008") in leg_info["undang_undang"]  # type: ignore[operator]
    assert set(leg_info["sections"].keys()) == {"Menimbang", "Mengingat", "Memperhatikan", "Mengadili"}  # type: ignore[union-attr]

    # 5. Test Unified Feature Extractor
    feat_extractor = IndonesianDomainFeatureExtractor(cfg)
    feats = feat_extractor.extract_features([sample_addr, sample_rp, sample_dt, sample_legal])
    assert feats.shape == (4, len(IndonesianDomainFeatureExtractor.FEATURE_NAMES))

    print("[SMOKE TEST] Indonesian Domain Extractors (Addresses, Rupiah, Dates, Legal)")
    print(f"  Parsed Rupiah values : {rp_vals}")
    print(f"  Parsed Dates & TZ    : dates={dt_info['dates']}, tz={dt_info['timezones']}")
    print(f"  Parsed Legal sections: {list(leg_info['sections'].keys())}")  # type: ignore[union-attr]
    print(f"  Domain feature matrix: {feats.shape}")
    print("[SUCCESS] All Indonesian domain extraction checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indonesian Domain Extractors")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    run_smoke_test(CFG(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
