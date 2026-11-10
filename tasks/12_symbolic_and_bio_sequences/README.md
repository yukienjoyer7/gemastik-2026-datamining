# Task 12: Symbolic, Chemical (SMILES) & Biological (DNA/RNA/Protein) Sequences

## 1. Formulation & Why Symbolic Sequences Differ from Prose NLP
- **Input:** Symbolic character strings where every character, case distinction (`C` = aliphatic carbon vs. `c` = aromatic carbon vs. `Cl` = chlorine), digit (`1..9` = ring-closure index), or parenthesis (`(` `)` = molecular branch) encodes exact physical, topological, or biological structure:
  1. **Chemical SMILES / SELFIES Strings:** Bioactivity, toxicity, solubility, blood-brain barrier permeability, and **Indonesian *Jamu* / Herbal Phytochemical (HerbalDB / IJAH)** compound-target interaction prediction.
  2. **Nucleotide Sequences (DNA / RNA FASTA):** Promoter/enhancer classification, viral variant typing, splice-site detection, mRNA degradation stability (OpenVaccine / Ribonanza).
  3. **Protein Amino-Acid Sequences (20-Letter Alphabet):** Enzyme thermostability ($\Delta T_m$, Novozymes), subcellular localization, antimicrobial peptide classification, B-cell epitope prediction.
  4. **Phonetic / Transliterated Aksara Sequences:** Fine-grained character-level sequence mapping and phonetic classification.
- **Critical Rule:** **NEVER lowercase SMILES strings or apply standard word tokenizers!** Lowercasing turns `CCN` (ethylamine) into `ccn` (an invalid aromatic fragment) and merges `Co` (Cobalt) with `cO` (aromatic carbon + oxygen).

## 2. Validation Strategy & Leakage Traps
- **Scaffold / Core-Skeleton `GroupKFold` (SMILES):** Random splits leak congeneric series (molecules sharing the exact same ring scaffold differing only by a methyl group). Always group by **Murcko-style Ring-System Skeleton** (`eda_and_data.py`) so all analogs of a core scaffold stay in the same fold.
- **Sequence-Identity / $k$-mer Cluster `GroupKFold` (DNA/RNA/Protein):** Homologous sequences or point mutants (`>80%` $k$-mer Jaccard similarity) must be grouped into the same fold to prevent memorizing sequence homologs.
- **Non-Canonical SMILES Duplicate Trap:** The same molecule can be written in dozens of valid SMILES strings (`CCO`, `OCC`, `C(O)C`). Detect and group equivalent topological formulas/skeletons before splitting, and average their predictions in post-processing (`postprocess.py`).

## 3. Escalation Playbook
- **Tier 1 (CPU, Zero-RDKit Dependency, <2 mins):**
  - **Case-Sensitive Multi-Scale Character & Atom-Token $k$-mer `(1, 6)` TF-IDF** (proven top baseline in Kaggle Leash Bio BELKA & Novozymes that runs fast without requiring RDKit!) + **Handcrafted Physicochemical & Topological Features** (for SMILES: atom counts `C, N, O, S, P, F, Cl, Br, I`, aromatic lowercase counts, double/triple bonds `=`, `#`, branch count `(`, ring closure digits `1..9`, MW proxy, H-bond donor/acceptor proxies, Lipinski Rule-of-5 descriptors, and functional group SMARTS/regex counts; for Bio: GC content, dinucleotide/trinucleotide frequencies, Kyte-Doolittle hydrophobicity, net charge, instability/aliphatic proxies, sequence Shannon entropy) $\rightarrow$ Out-Of-Fold **GBDT (`HistGradientBoostingClassifier`) + Calibrated Logistic/Ridge** ensemble (`baselines/tier1_fast_cpu.py`).
- **Tier 2 (Fast 1D Deep Learning + Sequence TTA):**
  - **1D Dilated Multi-Scale ResNet (`k=3, 5, 7, 11`) + Squeeze-and-Excitation (SE) + Self-Attention Pooling** operating directly on integer-encoded SMILES/Bio tokens (`baselines/tier2_strong_dl.py` & `modules/dilated_1d_resnet_encoder.py`).
  - **Test-Time Augmentation (TTA):** Average predictions across **Non-Canonical SMILES Enumerations** (cyclic root rotations & branch swaps) or **DNA/RNA Reverse-Complements** (`modules/smiles_and_sequence_enumerator.py`).

## 4. Directory Contents
- `eda_and_data.py` — Auto-detects sequence alphabet (`SMILES`, `DNA/RNA`, `PROTEIN`), checks SMILES parenthesis & ring-closure syntax validity, computes GC-content / amino-acid / atom compositions, and generates leak-free Scaffold / $k$-mer similarity `GroupKFold` clusters.
- `baselines/tier1_fast_cpu.py` — Standalone OOF pipeline combining Case-Sensitive Multi-Scale $k$-mer `(1, 6)` TF-IDF + Physicochemical/Topological/Substructure features into a GBDT + Ridge/Logistic blend (`oof_tier1_seq.npy`, `test_tier1_seq.npy`, `sub_tier1_seq.csv`).
- `baselines/tier2_strong_dl.py` — Standalone PyTorch OOF trainer using `DilatedResNet1DSequenceModel` + SE attention + SMILES Enumeration / Reverse-Complement Test-Time Augmentation (`oof_tier2_seq_dl.npy`, `test_tier2_seq_dl.npy`, `sub_tier2_seq_dl.csv`).
- `modules/smiles_and_sequence_enumerator.py` — Pure-Python regex SMILES tokenizer (`Cl`, `Br`, `[nH]`, `[C@@H]`), non-canonical SMILES cyclic/branch enumerator, and DNA/RNA reverse-complement generator for training augmentation & TTA.
- `modules/kmer_and_substructure_featurizer.py` — Overlapping contiguous & gapped $k$-mers, 18 phytochemical/drug functional group substructure counters (hydroxyl, carboxyl, amine, amide, ester, ketone, methoxy, phenol,flavone/aromatic rings, sulfonamide), and sliding-window amino-acid physicochemical profiles.
- `modules/dilated_1d_resnet_encoder.py` — Reusable PyTorch `DilatedResNet1DSequenceModel` with multi-kernel (`k=3, 5, 7, 11`) dilated residual blocks, Squeeze-and-Excitation channel attention, and Mean+Max+Attention pooling.
- `postprocess.py` — Canonical/scaffold duplicate prediction averaging (ensures identical molecules/sequences in different SMILES or reverse-complement notations get consistent predictions), rank/temperature calibration, and threshold/multiplier optimization.
- `winners/SUMMARY.md` — Curated references on Kaggle Leash Bio – BELKA, Novozymes Enzyme Stability, OpenVaccine / Ribonanza RNA, and Indonesian *Jamu* (IJAH / HerbalDB) bioactivity prediction.
