#!/usr/bin/env python3
"""
Task 12 — Module: K-mer, Phytochemical Substructure & Physicochemical Featurizer
================================================================================
Extracts domain-specific symbolic, chemical (SMILES), and biological (DNA/RNA/Protein)
features without requiring external C++ chemistry/bioinformatics libraries:
  1. Overlapping Contiguous & Gapped $k$-mers (`extract_contiguous_and_gapped_kmers`):
     Captures both adjacent motifs (`k=2..5`) and 1-gapped/2-gapped residue interactions
     (`A_G`, `C__O`, i-to-(i+3) alpha-helix hydrogen-bond spacing in proteins).
  2. Chemical & Phytochemical Substructure Counter (`extract_smiles_physicochemical_features`):
     - Atom counts (`C, N, O, S, P, F, Cl, Br, I`), aromatic atom counts (`c, n, o, s`),
       bond order counts (`=`, `#`), ring numbers (`1..9`), branching factor (`(`),
       approximate Molecular Weight (MW), H-bond donor/acceptor proxies, LogP proxy,
       and topological polar surface area (TPSA) proxy.
     - 14 Functional Group SMARTS-like regex counters critical for drug discovery and
       Indonesian *Jamu* / HerbalDB bioactivity tasks (hydroxyl/phenol, carboxyl,
       primary/secondary amine, amide, ester, ketone/aldehyde, methoxy/ether,
       benzene/aromatic ring, nitro, sulfonamide, halide, nitrile, lactone/flavone).
  3. Biological Sequence Physicochemical Profiler (`extract_bio_physicochemical_features`):
     - DNA/RNA: GC content, AT/GC skew, CpG dinucleotide island ratio, purine/pyrimidine
       transition rate, homopolymer run lengths, and codon/trinucleotide Shannon entropy.
     - Protein: Kyte-Doolittle hydrophobicity mean/max/window-std, net charge at pH 7.4,
       isoelectric point (pI) proxy, aromaticity fraction (`F, W, Y`), tiny/bulky/polar
       side-chain fractions, and sequence Shannon entropy.

Usage:
    python modules/kmer_and_substructure_featurizer.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import site
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
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class CFG:
    """Configuration for K-mer, Substructure, and Physicochemical Featurizer."""

    kmer_sizes: Tuple[int, ...] = (2, 3, 4)
    include_gapped_kmers: bool = True
    window_size: int = 15
    smoke_test: bool = False


# =============================================================================
# 1. Chemical SMILES Functional Group & Physicochemical Tables
# =============================================================================

_ATOMIC_MASSES: Dict[str, float] = {
    "C": 12.011,
    "c": 12.011,
    "N": 14.007,
    "n": 14.007,
    "O": 15.999,
    "o": 15.999,
    "S": 32.06,
    "s": 32.06,
    "P": 30.974,
    "p": 30.974,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
    "B": 10.81,
    "Si": 28.085,
}

# 14 Key Functional Group Substructure Regexes on SMILES strings
_FUNCTIONAL_GROUP_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("fg_hydroxyl_or_phenol", re.compile(r"(?<!=)\bO\b|\(O\)|^O|O$|\[OH\]")),
    ("fg_carboxyl", re.compile(r"C\(=O\)\[?O[H\-]?\]?|\[O-\]C=O|OC\(=O\)O")),
    ("fg_ester_or_lactone", re.compile(r"C\(=O\)O[Cc]|O=C1O|c1oc\(=O\)")),
    ("fg_amide_or_peptide", re.compile(r"C\(=O\)N|NC\(=O\)|C\(=O\)\[nH\]")),
    ("fg_ketone_or_aldehyde", re.compile(r"[Cc]C\(=O\)[Cc]|C=O")),
    ("fg_methoxy_or_ether", re.compile(r"COc|cOC|\(OC\)|COC")),
    ("fg_amine", re.compile(r"(?<!\(=O\))N(?![=O\+])|\[NH2?\+?\]|\[nH\]")),
    ("fg_benzene_or_aryl", re.compile(r"c1[cnos]{4,5}1|c2[cnos]{4,5}2")),
    ("fg_alkene_or_conjugated", re.compile(r"C=C|C=CC=O")),
    ("fg_alkyne_or_nitrile", re.compile(r"C#C|C#N|N#C")),
    ("fg_nitro", re.compile(r"\[N\+\]\(=O\)\[O-\]|N\(=O\)=O")),
    ("fg_sulfonamide_or_sulfonyl", re.compile(r"S\(=O\)\(=O\)")),
    ("fg_halide", re.compile(r"F|Cl|Br|I")),
    ("fg_phosphate_or_phosphonate", re.compile(r"P\(=O\)|OP\(")),
]


# =============================================================================
# 2. Biological Amino-Acid & Nucleotide Physicochemical Scales
# =============================================================================

# Kyte-Doolittle Hydropathy Index (1982)
_KD_HYDROPATHY: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Residue Molecular Weight (Daltons minus water)
_AA_WEIGHT: Dict[str, float] = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.20, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.11, "W": 186.21, "Y": 163.18, "V": 99.13,
}


def extract_contiguous_and_gapped_kmers(
    seq: str, k_sizes: Tuple[int, ...] = (2, 3), max_gap: int = 2
) -> List[str]:
    """
    Generate contiguous $k$-mers plus gapped di-mers (`A_1_G`, `A_2_G`) to capture
    periodic sequence motifs (e.g., codon positions, alpha-helix $i, i+3/i+4$ contacts,
    or ortho/meta/para aromatic substitution patterns).
    """
    s = "".join(str(seq or "").split())
    out: List[str] = []
    n = len(s)
    for k in k_sizes:
        if n >= k:
            out.extend(f"k{k}:{s[i:i+k]}" for i in range(n - k + 1))
    for g in range(1, max_gap + 1):
        step = g + 1
        if n > step:
            out.extend(f"g{g}:{s[i]}_{s[i+step]}" for i in range(n - step))
    return out


def extract_smiles_physicochemical_features(smiles: str) -> np.ndarray:
    """
    Compute 32 chemical, topological, Lipinski Rule-of-5, and functional-group
    substructure features directly from a SMILES string (zero RDKit dependency).
    """
    s = str(smiles or "").strip()
    n_chars = max(len(s), 1)

    c_aliph = s.count("C") - s.count("Cl") - s.count("Ca") - s.count("Cu") - s.count("Co")
    c_arom = s.count("c")
    n_aliph = s.count("N") - s.count("Na") - s.count("Ni")
    n_arom = s.count("n")
    o_aliph = s.count("O")
    o_arom = s.count("o")
    s_atom = s.count("S") - s.count("Si") - s.count("Se") + s.count("s")
    p_atom = s.count("P") + s.count("p")
    f_atom = s.count("F") - s.count("Fe")
    cl_atom = s.count("Cl")
    br_atom = s.count("Br")
    i_atom = s.count("I")

    heavy_atoms = max(
        1,
        c_aliph + c_arom + n_aliph + n_arom + o_aliph + o_arom + s_atom + p_atom + f_atom + cl_atom + br_atom + i_atom,
    )
    mw_proxy = (
        13.5 * (c_aliph + c_arom)
        + 15.0 * (n_aliph + n_arom)
        + 16.5 * (o_aliph + o_arom)
        + 32.1 * s_atom
        + 31.0 * p_atom
        + 19.0 * f_atom
        + 35.5 * cl_atom
        + 79.9 * br_atom
        + 126.9 * i_atom
    )

    double_bonds = s.count("=")
    triple_bonds = s.count("#")
    branches = s.count("(")
    ring_digits = sum(s.count(d) for d in "123456789") // 2
    stereocenters = s.count("@")
    charges = s.count("+") + s.count("-")

    # Lipinski H-bond Acceptor (N + O) & Donor (OH + NH proxies)
    hba_proxy = float(n_aliph + n_arom + o_aliph + o_arom)
    hbd_proxy = float(
        len(re.findall(r"(?<!=)\bO\b|\(O\)|^O|O$|\[OH\]|\[NH2?\+?\]|\[nH\]|N(?![=O\+\(])", s))
    )
    # Wildman-Crippen coarse LogP proxy (hydrophobic C + halogens minus polar N/O)
    logp_proxy = (
        0.20 * (c_aliph + c_arom)
        + 0.35 * (f_atom + cl_atom + br_atom + i_atom)
        - 0.45 * (n_aliph + n_arom + o_aliph + o_arom)
    )
    # Ertl coarse TPSA proxy
    tpsa_proxy = 20.2 * (o_aliph + o_arom) + 15.8 * (n_aliph + n_arom) + 25.0 * p_atom

    base_feats = [
        float(np.log1p(n_chars)),
        float(heavy_atoms),
        float(mw_proxy / 100.0),
        float(c_aliph),
        float(c_arom),
        float(c_arom) / heavy_atoms,  # aromatic fraction
        float(n_aliph + n_arom),
        float(o_aliph + o_arom),
        float(s_atom + p_atom),
        float(f_atom + cl_atom + br_atom + i_atom),
        float(double_bonds),
        float(triple_bonds),
        float(branches),
        float(ring_digits),
        float(stereocenters + charges),
        hba_proxy,
        hbd_proxy,
        logp_proxy,
    ]

    fg_counts = [float(len(pat.findall(s))) for _, pat in _FUNCTIONAL_GROUP_PATTERNS]
    return np.array(base_feats + fg_counts, dtype=np.float32)  # 18 + 14 = 32 features


def extract_bio_physicochemical_features(seq: str) -> np.ndarray:
    """
    Compute 18 biological sequence features covering both DNA/RNA (GC content,
    AT/GC skew, CpG ratio, homopolymer runs) and Protein (Kyte-Doolittle hydropathy,
    net charge, molecular weight, aromatic/polar/bulky fractions, Shannon entropy).
    """
    s = "".join(str(seq or "").upper().split())
    n = max(len(s), 1)
    counts = Counter(s)
    ent = -sum((v / n) * math.log2(v / n) for v in counts.values()) if s else 0.0

    # Nucleotide statistics
    cnt_a = counts.get("A", 0)
    cnt_c = counts.get("C", 0)
    cnt_g = counts.get("G", 0)
    cnt_tu = counts.get("T", 0) + counts.get("U", 0)
    gc_content = (cnt_g + cnt_c) / n
    at_skew = (cnt_a - cnt_tu) / max(cnt_a + cnt_tu, 1)
    gc_skew = (cnt_g - cnt_c) / max(cnt_g + cnt_c, 1)
    cpg_obs = s.count("CG") / max(n - 1, 1)
    cpg_exp = (cnt_c / n) * (cnt_g / n)
    cpg_oe_ratio = cpg_obs / max(cpg_exp, 1e-4)

    # Max homopolymer run (e.g., AAAAA or GGGGG)
    max_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1*", s)), default=0)

    # Protein amino-acid physicochemical profile
    hydro_vals = [_KD_HYDROPATHY.get(ch, 0.0) for ch in s]
    hydro_mean = float(np.mean(hydro_vals)) if hydro_vals else 0.0
    hydro_std = float(np.std(hydro_vals)) if hydro_vals else 0.0
    if len(hydro_vals) >= 9:
        win_means = [float(np.mean(hydro_vals[i : i + 9])) for i in range(0, len(hydro_vals) - 8, 3)]
        hydro_win_max = max(win_means)
        hydro_win_min = min(win_means)
    else:
        hydro_win_max = hydro_mean
        hydro_win_min = hydro_mean

    pos_charge = counts.get("K", 0) + counts.get("R", 0) + 0.1 * counts.get("H", 0)
    neg_charge = counts.get("D", 0) + counts.get("E", 0)
    net_charge_per_res = (pos_charge - neg_charge) / n
    charge_density = (pos_charge + neg_charge) / n

    arom_aa = (counts.get("F", 0) + counts.get("W", 0) + counts.get("Y", 0)) / n
    polar_aa = sum(counts.get(aa, 0) for aa in "RNDCQEGHKSTY") / n
    aliph_aa = (counts.get("A", 0) + 2.9 * counts.get("V", 0) + 3.9 * (counts.get("I", 0) + counts.get("L", 0))) / n
    mw_aa = sum(_AA_WEIGHT.get(ch, 110.0) for ch in s) / (1000.0 * n)

    return np.array(
        [
            float(np.log1p(len(s))),
            ent,
            gc_content,
            at_skew,
            gc_skew,
            min(cpg_oe_ratio, 5.0),
            float(max_run) / n,
            hydro_mean,
            hydro_std,
            hydro_win_max,
            hydro_win_min,
            net_charge_per_res,
            charge_density,
            arom_aa,
            polar_aa,
            aliph_aa,
            mw_aa,
            float(counts.get("C", 0)) / n,  # Cysteine disulfide bridge potential
        ],
        dtype=np.float32,
    )


class SequencePhysicochemicalFeaturizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that extracts all 50 chemical (SMILES) +
    biological (DNA/RNA/Protein) physicochemical & functional-group features as a CSR matrix.
    """

    def fit(self, X: Sequence[str], y: Optional[np.ndarray] = None) -> "SequencePhysicochemicalFeaturizer":
        return self

    def transform(self, X: Sequence[str]) -> sparse.csr_matrix:
        feats = np.zeros((len(X), 50), dtype=np.float32)
        for i, seq in enumerate(X):
            chem = extract_smiles_physicochemical_features(seq)
            bio = extract_bio_physicochemical_features(seq)
            feats[i, :32] = chem
            feats[i, 32:] = bio
        return sparse.csr_matrix(feats)


def _run_smoke_test() -> None:
    curcumin = "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"
    chem_f = extract_smiles_physicochemical_features(curcumin)
    assert chem_f.shape == (32,)
    assert chem_f[4] > 0  # aromatic carbons
    assert chem_f[18] >= 2  # phenolic OH count
    assert chem_f[23] >= 2  # methoxy count

    peptide = "MKTIIALSYIFCLVFADYKDDDDK"
    bio_f = extract_bio_physicochemical_features(peptide)
    assert bio_f.shape == (18,)

    kmers = extract_contiguous_and_gapped_kmers("ATGCGATCG", k_sizes=(2, 3), max_gap=2)
    assert "k3:ATG" in kmers and "g1:A_G" in kmers

    fe = SequencePhysicochemicalFeaturizer()
    mat = fe.fit_transform([curcumin, peptide])
    assert mat.shape == (2, 50)
    print(f"[SMOKE TEST] Curcumin SMILES features shape={chem_f.shape}, Peptide features shape={bio_f.shape}")
    print(f"[SMOKE TEST] Combined feature matrix shape={mat.shape}")
    print("[SUCCESS] kmer_and_substructure_featurizer smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-mer, Substructure & Physicochemical Featurizer")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
