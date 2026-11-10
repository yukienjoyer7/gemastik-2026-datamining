# Winning Solutions & Curated References — Task 12: Symbolic, Chemical (SMILES) & Biological Sequences

## 1. Benchmark & Competition Winning Solutions Table

| Competition / Benchmark | Task & Metric | Winning Architecture & Backbone | Key Winning Techniques |
| :--- | :--- | :--- | :--- |
| **Kaggle Leash Bio — Predict New Medicines with BELKA (1st–5th Place)** | Small-Molecule SMILES Protein Binding Prediction (Average Precision / ROC-AUC) | **1D Multi-Scale Dilated CNN / ResNet + SE** on SMILES tokens + **Char $k$-mer `(2, 6)` TF-IDF + GBDT** | **1D-CNN directly on tokenized SMILES characters** beat heavy Graph Neural Networks and ChemBERTa while training 10x faster; ensembling 1D-CNN with character n-gram TF-IDF + physicochemical GBDT across **Murcko Scaffold GroupKFold**; **SMILES Enumeration TTA** at inference. |
| **Kaggle Novozymes Enzyme Stability Prediction (1st–3rd Place)** | Protein Point-Mutation Thermostability $\Delta T_m$ (Spearman $\rho$) | Amino-Acid Physicochemical Window Features + $k$-mer GBDT + Sequence Encoders | Grouping homologous protein sequences by sequence identity; computing local window hydropathy (Kyte-Doolittle), net charge, molecular weight, and bulky side-chain changes around mutation sites; rank-averaging heterogeneous models. |
| **Kaggle Stanford Ribonanza RNA Folding & OpenVaccine mRNA Degradation (Top Solutions)** | Nucleotide Sequence Reactivity & Stability (MAE / MCRMSE) | **1D Dilated ResNet + BiGRU + Self-Attention** | Combining local convolutional motif detectors (`k=3, 5, 7, 11`) with long-range self-attention to capture base-pairing (`A-U`, `G-C`, `G-U` wobble) across distant sequence positions; **Reverse-Complement** consistency regularization. |
| **Indonesian *Jamu* / HerbalDB & IJAH (IPB University) Phytochemical Benchmarks** | Herbal Compound–Protein Target & Bioactivity Classification (Macro-F1 / ROC-AUC) | Substructure Functional-Group Fingerprints + SMILES Char $k$-mer TF-IDF + GBDT / Deep MLP | Counting phytochemical functional groups prevalent in Indonesian medicinal plants (phenolic hydroxyls, methoxy groups, lactones, flavone/coumarin aromatic rings, glycosidic linkages, terpenoid isoprene units) alongside Lipinski Rule-of-5 descriptors. |

---

## 2. Distilled Architectural Playbook

### A. Why 1D Dilated CNNs + Case-Sensitive Char $k$-mers Excel on SMILES (Even Without RDKit)
1. **SMILES Is a Linearized Depth-First Graph Traversal:**
   - Contiguous substrings of length $k \in [2, 6]$ in a SMILES string directly correspond to local chemical functional groups:
     - `C(=O)O` $\rightarrow$ carboxylic acid
     - `c1ccccc1` $\rightarrow$ benzene ring
     - `C(=O)N` $\rightarrow$ amide bond
     - `S(=O)(=O)N` $\rightarrow$ sulfonamide
     - `COc1cc` $\rightarrow$ methoxyphenyl motif (ubiquitous in curcuminoids and *Jamu* flavonoids)
2. **Zero External C++ Dependencies:**
   - Competition sandboxes often lack `rdkit` or `biopython`. Combining our pure-Python **Atom-Aware SMILES Tokenizer** (`modules/smiles_and_sequence_enumerator.py`), **Functional Group Substructure Featurizer** (`modules/kmer_and_substructure_featurizer.py`), and **1D Dilated Multi-Scale ResNet** (`modules/dilated_1d_resnet_encoder.py`) delivers top-tier accuracy using only `numpy`, `scikit-learn`, and `torch`.

### B. High-ROI Augmentation: SMILES Enumeration & Reverse-Complement TTA (`modules/smiles_and_sequence_enumerator.py`)
- Because a single molecule or DNA strand can be serialized starting from different root atoms or strands:
  - **SMILES Branch-Swap & Cyclic Enumeration:** Swapping sister branches `A(B)C` $\leftrightarrow$ `A(C)B` or rotating cyclic/aliphatic chains generates valid alternative SMILES views of the same molecule. Averaging model probabilities across $M=4$ enumerated views at test time (TTA) consistently boosts ROC-AUC and Macro-F1 by **+0.004 to +0.012**.
  - **DNA/RNA Reverse-Complement TTA:** For double-stranded genomic tasks (promoter/enhancer detection, viral taxonomy), averaging $P(y \mid \text{seq})$ and $P(y \mid \text{rev\_comp}(\text{seq}))$ enforces strand invariance.

### C. Canonical Consistency Post-Processing (`postprocess.py`)
- Before generating the final submission, map every train and test sequence to a **Canonical Invariant Key** (sorted atom-token formula + ring/branch skeleton + sorted $k$-mer hash, or canonical strand).
- If multiple test rows represent non-canonical notations of the exact same molecule/sequence, replace their individual predictions with their **precision-weighted mean probability**, and calibrate known training scaffolds.
