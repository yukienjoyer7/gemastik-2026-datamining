# Task 02: Text Pair Classification & Semantic Similarity

## 1. Formulation & Common Competition Metrics
- **Input:** Pair of texts `(text_a, text_b)` (e.g., question duplicates, paraphrase detection, NLI entailment, product/entity matching, STS similarity score).
- **Output:** Binary/multiclass label or continuous similarity score $[0, 1]$ / $[0, 5]$.
- **Metrics:** Pearson $r$, Spearman $\rho$, LogLoss, F1, ROC-AUC.

## 2. Validation Strategy & Leakage Traps
- **Critical Leakage Trap (Transitive Graph Leakage):** If `(A, B)` is in train and `(B, C)` is in validation, the model memorizes `B`. Build an undirected graph of all paired items and apply `GroupKFold` / `StratifiedGroupKFold` on **connected components**.
- **Symmetry:** Check whether `(A, B)` and `(B, A)` have identical semantics; use test-time augmentation (TTA) by averaging predictions of `(A, B)` and `(B, A)`.

## 3. Escalation Playbook
- **Tier 1 (CPU, <5 mins):** TF-IDF cosine similarity, BM25 overlap, Jaccard/char n-gram overlap, Levenshtein/fuzzy ratios, length ratios + Graph features (node degree, PageRank, shared neighbors in pair graph) $\rightarrow$ LightGBM / CatBoost.
- **Tier 2 (GPU Cross-Encoder):** `[CLS] context [SEP] text_a [SEP] text_b [SEP]` with DeBERTa-v3 / XLM-R / IndoBERT. Use `PearsonCorrelationLoss` / `BCEWithLogitsLoss` / `CrossEntropyLoss`.
- **Tier 3 (Bi-Encoder + Reranker Fusion):** Sentence-Transformer Bi-Encoder embeddings (`u`, `v`, `|u - v|`, `u * v`, cosine) combined with Cross-Encoder OOF logits and graph statistics inside a 2nd-stage GBDT.

## 4. Directory Contents
- `baselines/` — Standalone pair feature + GBDT baseline and Cross-Encoder OOF script.
- `modules/` — Connected-component graph feature extractor, Pearson/Spearman losses, Bi-Encoder interaction heads.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions (e.g., U.S. Patent Phrase to Phrase Matching, Quora Question Pairs).
