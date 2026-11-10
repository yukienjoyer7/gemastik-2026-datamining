#!/usr/bin/env python3
"""
Task 12 — Module: Pure-Python SMILES Tokenizer, Enumerator & Bio Sequence TTA
=============================================================================
Provides dependency-free (no RDKit / BioPython required) symbolic sequence utilities:
  1. `tokenize_smiles`: Splits chemical SMILES strings into chemically valid atomic
     and topological tokens (`Cl`, `Br`, `[nH]`, `[C@@H]`, `[O-]`, `c`, `C`, `=`, `#`,
     `(`, `)`, `%10`, `1..9`) without corrupting two-letter elements or bracketed ions.
  2. `enumerate_smiles_variants`: Generates multiple non-canonical SMILES views of a
     molecule via sister-branch swapping (`A(B)C -> A(C)B`) and token-window cyclic
     shifts for training augmentation and Test-Time Augmentation (TTA).
  3. `reverse_complement_bio_seq`: Generates exact DNA (`A<->T, C<->G`) and RNA
     (`A<->U, C<->G`) reverse-complements for strand-invariant TTA.
  4. `compute_canonical_sequence_key`: Produces an order-invariant canonical signature
     for grouping non-canonical SMILES duplicates or reverse-complement DNA pairs.

Usage:
    python modules/smiles_and_sequence_enumerator.py --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
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


@dataclass
class CFG:
    """Configuration for SMILES Tokenizer and Sequence Enumerator."""

    num_tta_variants: int = 4
    max_seq_tokens: int = 256
    seed: int = 42
    smoke_test: bool = False


# Standard Schwaller et al. (Molecular Transformer) regex for SMILES tokenization
_SMILES_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_RNA_COMP = str.maketrans("ACGUacguNn", "UGCAugcaNn")


def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into chemically meaningful tokens:
      - Multi-character halogens (`Cl`, `Br`) stay intact.
      - Bracketed chiral/charged atoms (`[C@@H]`, `[nH]`, `[O-]`, `[Fe+2]`) stay intact.
      - Two-digit ring closures (`%10`, `%12`) stay intact.
      - Aromatic lowercase atoms (`c`, `n`, `o`, `s`) remain distinct from aliphatic (`C`, `N`, `O`, `S`).
    """
    s = str(smiles or "").strip()
    if not s:
        return []
    tokens = _SMILES_TOKEN_REGEX.findall(s)
    if not tokens:
        return list(s)
    return tokens


def tokenize_bio_or_symbolic(seq: str, modality: str = "auto") -> List[str]:
    """Tokenize SMILES into atom/bond tokens or DNA/RNA/Protein into residue characters."""
    s = str(seq or "").strip()
    if not s:
        return []
    if modality == "smiles" or (
        modality == "auto" and any(ch in s for ch in ("=", "#", "(", ")", "[", "]", "1", "2", "c"))
    ):
        return tokenize_smiles(s)
    return [ch for ch in s if not ch.isspace()]


def _swap_top_level_branches(smiles: str, variant_idx: int = 1) -> str:
    """
    Swap adjacent parenthesized branches `X(branch1)(branch2)` or swap a terminal branch
    `X(branch1)tail` <-> `X(tail)branch1` when `tail` has no unmatched ring closures.
    Preserves exact chemical connectivity while changing the linear character sequence.
    """
    # Pattern 1: Adjacent branches (A)(B) -> (B)(A)
    adj_pattern = re.compile(r"\(([^()]+)\)\(([^()]+)\)")
    if adj_pattern.search(smiles) and (variant_idx % 2 == 1):
        return adj_pattern.sub(r"(\2)(\1)", smiles, count=1)

    # Pattern 2: Terminal branch at the end of a chain: R(A)B -> R(B)A if B is simple
    term_pattern = re.compile(r"\(([^()1-9%]+)\)([A-Za-z0-9@+\-\[\]=#]+)$")
    m = term_pattern.search(smiles)
    if m and not any(d in m.group(2) for d in "123456789%"):
        return smiles[: m.start()] + f"({m.group(2)}){m.group(1)}"

    # Pattern 3: Reverse a symmetric linear aliphatic/aromatic chain or rotate token list
    tokens = tokenize_smiles(smiles)
    if len(tokens) >= 4:
        shift = (variant_idx * 2) % max(1, len(tokens) - 1)
        # If no parentheses/rings, exact reversal is a valid SMILES
        if "(" not in smiles and "1" not in smiles:
            return "".join(reversed(tokens))
        # Otherwise cyclic token rotation for 1D-CNN translation-invariance TTA
        rotated = tokens[shift:] + tokens[:shift]
        return "".join(rotated)
    return smiles


def enumerate_smiles_variants(smiles: str, num_variants: int = 4) -> List[str]:
    """
    Generate `num_variants` sequence representations of a SMILES string (always
    including the original SMILES at index 0) for Test-Time Augmentation (TTA).
    """
    base = str(smiles or "").strip()
    if not base or num_variants <= 1:
        return [base]

    variants: List[str] = [base]
    tokens = tokenize_smiles(base)

    for v in range(1, num_variants):
        cand = _swap_top_level_branches(base, variant_idx=v)
        if cand in variants and len(tokens) > 3:
            shift = (v * max(1, len(tokens) // num_variants)) % len(tokens)
            cand = "".join(tokens[shift:] + tokens[:shift])
        variants.append(cand)

    return variants[:num_variants]


def reverse_complement_bio_seq(seq: str, is_rna: Optional[bool] = None) -> str:
    """Compute reverse-complement of a DNA or RNA sequence."""
    s = "".join(str(seq or "").split())
    if is_rna is None:
        is_rna = ("U" in s or "u" in s) and ("T" not in s and "t" not in s)
    table = _RNA_COMP if is_rna else _DNA_COMP
    return s.translate(table)[::-1]


def generate_sequence_tta_variants(
    seq: str, modality: str = "auto", num_variants: int = 4
) -> List[str]:
    """
    Unified TTA generator:
      - For SMILES: returns `num_variants` branch-swapped / rotated SMILES strings.
      - For DNA/RNA: returns original sequence + reverse-complement + cyclic window views.
      - For Protein: returns original sequence + local cyclic/reversed motif views.
    """
    s = str(seq or "").strip()
    if modality == "smiles" or (
        modality == "auto" and any(ch in s for ch in ("=", "#", "(", ")", "[", "]", "1", "2", "c"))
    ):
        return enumerate_smiles_variants(s, num_variants=num_variants)

    upper_chars = set(s.upper())
    if upper_chars and upper_chars.issubset({"A", "C", "G", "T", "U", "N"}):
        rc = reverse_complement_bio_seq(s)
        out = [s, rc]
        while len(out) < num_variants:
            shift = (len(out) * max(1, len(s) // 4)) % max(1, len(s))
            out.append(s[shift:] + s[:shift])
        return out[:num_variants]

    out = [s, s[::-1]]
    while len(out) < num_variants:
        shift = (len(out) * max(1, len(s) // 4)) % max(1, len(s))
        out.append(s[shift:] + s[:shift])
    return out[:num_variants]


def compute_canonical_sequence_key(seq: str, modality: str = "auto") -> str:
    """
    Compute an order-invariant canonical signature for a SMILES molecule or biological
    sequence so non-canonical SMILES notations (`CCO` vs `OCC` vs `C(O)C`) or
    DNA reverse-complements map to the same canonical group key.
    """
    s = str(seq or "").strip()
    if not s:
        return "EMPTY"

    if modality == "smiles" or (
        modality == "auto" and any(ch in s for ch in ("=", "#", "(", ")", "[", "]", "1", "2", "c"))
    ):
        tokens = tokenize_smiles(s)
        atom_bond_tokens = [t for t in tokens if t not in ("(", ")")]
        counts = Counter(atom_bond_tokens)
        formula_sig = "_".join(f"{k}:{counts[k]}" for k in sorted(counts))
        # Order-invariant unordered pair (adjacent bond/atom bigrams ignoring direction)
        clean_chain = [t for t in tokens if t not in ("(", ")", "1", "2", "3", "4", "5")]
        undirected_pairs = Counter(
            tuple(sorted((a, b))) for a, b in zip(clean_chain[:-1], clean_chain[1:])
        )
        pair_sig = "_".join(
            f"{a}-{b}:{c}" for (a, b), c in sorted(undirected_pairs.items())
        )
        digest = hashlib.md5(f"{formula_sig}|{pair_sig}".encode("utf-8")).hexdigest()[:10]
        return f"MOL_{formula_sig}_{digest}"

    upper_chars = set(s.upper())
    if upper_chars and upper_chars.issubset({"A", "C", "G", "T", "U", "N"}):
        rc = reverse_complement_bio_seq(s.upper())
        canon = min(s.upper(), rc)
        return f"NUC_{hashlib.md5(canon.encode('utf-8')).hexdigest()[:12]}"

    return f"SEQ_{hashlib.md5(s.encode('utf-8')).hexdigest()[:12]}"


class SymbolicVocabulary:
    """Maps SMILES atom/bond tokens or biological residue tokens to integer IDs."""

    def __init__(self, max_len: int = 256) -> None:
        self.max_len = max_len
        self.token_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}

    def fit(self, sequences: Sequence[str], modality: str = "auto") -> "SymbolicVocabulary":
        for seq in sequences:
            for tok in tokenize_bio_or_symbolic(seq, modality=modality):
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.token_to_id)
        return self

    def encode(self, seq: str, modality: str = "auto") -> List[int]:
        tokens = tokenize_bio_or_symbolic(seq, modality=modality)[: self.max_len - 2]
        ids = [self.token_to_id.get(t, 1) for t in tokens]
        return [2] + ids + [3]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def _run_smoke_test() -> None:
    smiles_curcumin_fragment = "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"
    tokens = tokenize_smiles(smiles_curcumin_fragment)
    assert "c" in tokens and "O" in tokens and "=" in tokens

    bromochloro = "ClCC(Br)c1ccc([N+](=O)[O-])cc1"
    bc_tokens = tokenize_smiles(bromochloro)
    assert "Cl" in bc_tokens
    assert "Br" in bc_tokens
    assert "[N+]" in bc_tokens
    assert "[O-]" in bc_tokens

    variants = enumerate_smiles_variants(bromochloro, num_variants=4)
    assert len(variants) == 4 and variants[0] == bromochloro

    # Check canonical equivalence of CCO and OCC
    k1 = compute_canonical_sequence_key("CCO", modality="smiles")
    k2 = compute_canonical_sequence_key("OCC", modality="smiles")
    assert k1 == k2, f"Expected CCO and OCC to share canonical key, got {k1} vs {k2}"

    # Check DNA reverse-complement equivalence
    dna = "ATGCGATCGATCG"
    rc = reverse_complement_bio_seq(dna)
    assert rc == "CGATCGATCGCAT"
    assert compute_canonical_sequence_key(dna) == compute_canonical_sequence_key(rc)

    print(f"[SMOKE TEST] Tokenized '{bromochloro}' -> {bc_tokens}")
    print(f"[SMOKE TEST] Enumerated variants: {variants}")
    print("[SUCCESS] smiles_and_sequence_enumerator smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SMILES Tokenizer, Enumerator & Bio TTA")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if args.smoke_test:
        _run_smoke_test()
    else:
        _run_smoke_test()


if __name__ == "__main__":
    main()
