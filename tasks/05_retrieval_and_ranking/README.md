# Task 05: Retrieval & Ranking

## 1. Formulation & Common Competition Metrics
- **Input:** Queries $Q$ and candidate corpus $D$ (or query-item pairs with relevance grades).
- **Output:** Top-$K$ ordered candidate IDs per query (or variable-length candidate sets).
- **Metrics:** MAP@K, NDCG@K, MRR@K, Recall@K, $F_2$-score / $F_1$-score on retrieved set.

## 2. Validation Strategy & Leakage Traps
- **Query/Corpus Split:** Use `GroupKFold` on `query_id` (or topic/user cluster). If candidate items also appear as queries or share tight clusters, check whether unseen corpus items exist in test.
- **Candidate Generation Recall Ceiling:** Your Stage-2 reranker cannot recover positives missed by Stage-1 retrieval. Always track **Stage-1 Recall@K** ($K \in \{5, 10, 50, 100\}$) via `evaluate_candidate_recall` in [`eda_and_data.py`](eda_and_data.py) before tuning the reranker.

## 3. Escalation Playbook
- **Stage 1 (Candidate Generation — High Recall):**
  - Sparse: Built-in vectorized Okapi BM25 + Word/Char TF-IDF cosine top-$K$ (`tier1_fast_cpu.py`).
  - Dense Bi-Encoder: Contrastive fine-tuning with **MultipleNegativesRankingLoss (MNRL)** + iterative **Hard Negative Mining** with false-negative margin filtering + **$\alpha$-Query Expansion (AQE)** (`tier2_strong_dl.py`).
  - Hybrid Fusion: Reciprocal Rank Fusion (RRF) of Sparse + Dense candidates (`modules/retrieval_and_reranking.py`).
- **Stage 2 (Reranking — High Precision):**
  - Cross-Encoder reranker on top-$K$ candidates (`tier2_strong_dl.py`).
  - GBDT Pointwise/Listwise (`LightGBM` / `HistGradientBoostingClassifier`) combining BM25 score, TF-IDF cosine, Jaccard overlap, reciprocal rank, and query-relative z-score / gap-to-top1 features (`tier1_fast_cpu.py`).
- **Stage 3 (Post-Processing):**
  - Mutual $k$-NN reciprocal score boosting + dynamic $(\tau_{\text{abs}}, \tau_{\text{rel}})$ candidate cutoff optimization for $F_\beta$ (`postprocess.py`).

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Retrieval & ranking diagnostics, unseen corpus overlap check, `evaluate_candidate_recall`, and leak-free `GroupKFold` builder.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Two-Stage CPU Retrieval (Okapi BM25 + Word/Char TF-IDF + RRF) + GBDT Reranker with query-context features.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Bi-Encoder Contrastive Fine-Tuning (MNRL + Hard Negative Mining + $\alpha$-QE) + Cross-Encoder Reranker.
- [`modules/retrieval_and_reranking.py`](modules/retrieval_and_reranking.py) — `MultipleNegativesRankingLoss`, `HardNegativeMiner`, `reciprocal_rank_fusion`, `alpha_query_expansion`, and `build_query_context_features`.
- [`postprocess.py`](postprocess.py) — Dynamic candidate cutoff selector ($\tau_{\text{abs}}, \tau_{\text{rel}}, \text{min\_k}, \text{max\_k}$) + Mutual $k$-NN score boosting.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Detailed breakdown of winning solutions (Learning Equality 1st, Shopee 1st, H&M 1st).
