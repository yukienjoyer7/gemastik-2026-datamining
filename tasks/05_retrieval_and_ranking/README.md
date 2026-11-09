# Task 05: Retrieval & Ranking

## 1. Formulation & Common Competition Metrics
- **Input:** Queries $Q$ and candidate corpus $D$ (or query-item pairs with relevance grades).
- **Output:** Top-$K$ ordered candidate IDs per query.
- **Metrics:** MAP@K, NDCG@K, MRR@K, Recall@K, F2-score on retrieved set.

## 2. Validation Strategy & Leakage Traps
- **Query/Corpus Split:** Use `GroupKFold` on `query_id` (or topic/user cluster). If candidate items also appear as queries or share tight clusters, check whether unseen corpus items exist in test.
- **Candidate Generation Recall Ceiling:** Your Stage-2 reranker cannot recover positives missed by Stage-1 retrieval. Always track **Stage-1 Recall@K** ($K \in \{50, 100, 200\}$) before tuning the reranker.

## 3. Escalation Playbook
- **Stage 1 (Candidate Generation — High Recall):**
  - Sparse: BM25 (`rank_bm25` / `bm25s`) + Char/Word TF-IDF cosine top-$K$.
  - Dense Bi-Encoder: `bge-m3` / `multilingual-e5` / fine-tuned Sentence-Transformer with **MultipleNegativesRankingLoss (MNRL)** + iterative **Hard Negative Mining** $\rightarrow$ FAISS / exact cosine top-$K$.
  - Hybrid Fusion: Reciprocal Rank Fusion (RRF) of Sparse + Dense candidates.
- **Stage 2 (Reranking — High Precision):**
  - Cross-Encoder reranker (`DeBERTa` / `bge-reranker-v2-m3`) on top-$K$ candidates.
  - GBDT Pointwise/Listwise (`LightGBM lambdarank` / `CatBoost YetiRank`) combining BM25 score, Bi-Encoder cosine, Cross-Encoder logit, rank positions, and query-neighborhood statistics.

## 4. Directory Contents
- `baselines/` — Standalone BM25 + Bi-Encoder retrieval and GBDT/Cross-Encoder reranking scripts.
- `modules/` — Hard negative mining loop, Reciprocal Rank Fusion (RRF), MNRL / InfoNCE loss.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions (e.g., Learning Equality, Shopee, H&M, WANDS).
