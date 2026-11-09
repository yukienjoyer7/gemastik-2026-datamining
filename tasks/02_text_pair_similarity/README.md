# Task 02: Text Pair Classification & Semantic Similarity

## 1. Formulation & Common Competition Metrics
- **Input:** Pair of texts `(text_a, text_b)` plus optional `context` (e.g., question duplicates, paraphrase detection, NLI entailment, product/entity matching, STS similarity score).
- **Output:** Binary/multiclass label or continuous similarity score $[0, 1]$ / $[0, 5]$.
- **Metrics:** Pearson $r$, Spearman $\rho$, LogLoss, F1, ROC-AUC.

## 2. Validation Strategy & Leakage Traps
- **Critical Leakage Trap (Transitive Graph Leakage):** If `(A, B)` is in train and `(B, C)` is in validation, the model memorizes `B`. Build an undirected graph of all paired items (`eda_and_data.py`) and apply `GroupKFold` / `StratifiedGroupKFold` on **connected components**.
- **Symmetry:** Check whether `(A, B)` and `(B, A)` have identical semantics; use symmetric test-time augmentation (TTA) by averaging predictions of `(A, B)` and `(B, A)`.

## 3. Escalation Playbook
- **Tier 1 (CPU, <5 mins):** Word/Char TF-IDF cosine similarity, Jaccard/Dice n-gram overlap, edit similarity, length ratios + **Magic Pair-Graph Features** (node degree, shared neighbors, Adamic-Adar, component size, PageRank) $\rightarrow$ LightGBM / HistGradientBoosting.
- **Tier 2 (GPU Cross-Encoder):** `[CLS] context [SEP] text_a [SEP] text_b [SEP]` with **Context-Grouping Augmentation**, **Symmetric TTA**, and `PearsonCorrelationLoss` / `BCEWithLogitsLoss`.
- **Tier 3 (Bi-Encoder + Transitive Graph Post-Processing):** Sentence-Transformer Bi-Encoder embeddings (`concat([u, v, |u - v|, u * v])`, `CoSENTLoss`) combined with Cross-Encoder OOFs and transitive triangle propagation.

## 4. Directory Contents
- `eda_and_data.py` — Union-Find pair graph analyzer (connected components, degree distribution, hubs, cross-fold leakage check), length ratio/symmetry diagnostics, and `TextPairCrossEncoderDataset` with balanced truncation.
- `baselines/tier1_fast_cpu.py` — Standalone lexical similarity + Magic Pair-Graph features + OOF GBDT (`LightGBM` / `HistGradientBoosting`) pipeline.
- `baselines/tier2_strong_dl.py` — Standalone Cross-Encoder Transformer OOF pipeline with **Context-Grouping Augmentation**, **Symmetric TTA**, and Pearson/BCE/CE losses.
- `modules/pair_features_and_biencoder.py` — `BiEncoderInteractionHead`, `CoSENTLoss`, `PearsonCorrelationLoss`, and `PairGraphFeatureExtractor`.
- `postprocess.py` — Transitive triangle consistency propagation across pair graph wedges, MinMax/rank scaling, and F1 threshold optimization.
- `winners/SUMMARY.md` — Detailed breakdown of U.S. Patent 1st Place, Quora Question Pairs 1st Place, and Shopee Product Matching winning solutions.
