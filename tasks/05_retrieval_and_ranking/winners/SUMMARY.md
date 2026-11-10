# Winning Solutions & Curated References — Retrieval & Ranking

## 1. Benchmark Competition Hall of Fame

| Competition | Metric | 1st Place Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **Kaggle Learning Equality – Curriculum Recommendations** | Macro $F_2$ | Stage-1 Multilingual Bi-Encoder (`xlm-roberta-base`, `paraphrase-multilingual-mpnet`) $\rightarrow$ Stage-2 Cross-Encoder + GBDT | Iterative hard-negative mining cycles (3 rounds), hierarchical topic-tree path context concatenation (`root > parent > node`), dynamic relative/absolute probability cutoffs. |
| **Kaggle Shopee – Price Match Guarantee** | Mean $F_1$ | Multimodal Bi-Encoder (ArcFace text + image embeddings) + Iterative Neighborhood Blending | $\alpha$-Query Expansion (AQE), Mutual $k$-NN graph smoothing, minimum-size cluster guarantees (top-2 fallback when cluster size = 1). |
| **Kaggle H&M Personalized Fashion Recommendations** | MAP@12 | Multi-Source Candidate Generation (10+ strategies) $\rightarrow$ LightGBM `lambdarank` | Decay-weighted repurchase candidates, co-visitation item-to-item transitions, BPR collaborative embeddings, and query-relative z-score features. |
| **WANDS / Amazon ESCI Product Search** | NDCG@10 | Hybrid BM25 + Dense Bi-Encoder RRF $\rightarrow$ DeBERTa-v3 Cross-Encoder + CatBoost `YetiRank` | False-negative margin filtering during hard negative mining (`sim > 0.95 * pos_sim` excluded), out-of-fold Cross-Encoder logit stacking. |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Kaggle Learning Equality – Curriculum Recommendations (1st Place)
- **Problem Formulation:** Match educational topic nodes ($Q$) against a corpus of $>150\text{k}$ content items ($D$) across 20+ languages, evaluated by Macro $F_2$ (which weights Recall $4\times$ higher than Precision).
- **Stage 1 — Context-Enriched Bi-Encoder with Iterative Hard-Negative Mining:**
  1. **Tree Context Representation:** A topic title alone (e.g., `"Introduction"`) is ambiguous. Winners serialized the full ancestor breadcrumb path (`"Science > Biology > Cell Structure > Introduction"`) plus sibling node titles into the query representation.
  2. **MultipleNegativesRankingLoss (MNRL):** Trained with large effective batch sizes ($B \ge 256$ via gradient caching / GradCache) and temperature $\tau = 0.05$.
  3. **Iterative Mining Loop:**
     - *Round 0:* Train on random + in-batch negatives.
     - *Round 1:* Retrieve top-50 candidates per topic using Round-0 embeddings; select negatives ranked 1..30 and retrain.
     - *Round 2:* Filter suspected **false negatives** (unlabeled positives where cosine similarity exceeds $0.95 \times \text{sim}(q, d^+)$) so the model does not unlearn valid matches.
- **Stage 2 — Cross-Encoder & GBDT Context Reranking:**
  - Top-50 candidates per query are scored by a Cross-Encoder (`xlm-roberta-large`) and fed into a LightGBM classifier alongside **query-level relative features**:
    - `score_diff_to_top1 = score - max(score_for_query)`
    - `score_zscore = (score - mean(score_for_query)) / std(score_for_query)`
    - Neighbor topic correlation rate (fraction of parent/sibling topics in the tree that also retrieved candidate $d$).
- **Post-Processing ($F_2$ Dynamic Cutoff):**
  - Instead of a fixed threshold $\tau$, accept candidate $i$ if $p_i \ge \tau_{\text{abs}}$ **and** $p_i \ge \tau_{\text{rel}} \cdot \max_j p_j$, while guaranteeing at least `min_k = 1` prediction per query.

---

### B. Shopee – Price Match Guarantee (1st Place)
- **Problem Formulation:** Given query products (title + image), retrieve all duplicate products in the test set (evaluated by item-level Macro $F_1$).
- **Key Winning Techniques:**
  1. **$\alpha$-Query Expansion (AQE):**
     $$\tilde{\mathbf{q}}_i = \text{normalize}\!\left(\mathbf{q}_i + \sum_{j \in \mathcal{N}_k(i)} \left(\mathbf{q}_i^\top \mathbf{d}_j\right)^\alpha \mathbf{d}_j\right)$$
     Smooths noisy single-item embeddings using high-confidence neighbors before final similarity computation.
  2. **Mutual $k$-NN Reciprocal Verification:**
     If item $B$ is in the top-$K$ of item $A$, **and** item $A$ is in the top-$K$ of item $B$, their similarity score receives a reciprocal boost $+\delta$. Non-mutual edges have their similarity attenuated.
  3. **Rank-1 Fallback Guarantee:**
     Since every query in Shopee was guaranteed to have at least 2 items in its duplicate group (itself + 1 match), any query with only 1 accepted item had its 2nd-best neighbor automatically appended if its similarity exceeded a lower safety threshold.

---

### C. H&M Personalized Fashion Recommendations (1st Place)
- **Problem Formulation:** Recommend top-12 articles per customer (`MAP@12`) from $>100\text{k}$ fashion items.
- **Key Winning Techniques:**
  1. **Multi-Source Candidate Union (Target ~100–150 Candidates/Query):**
     - Source 1: Customer's own recent purchases (exponential time-decay weighted).
     - Source 2: Item-to-item co-purchased/co-viewed transition matrix.
     - Source 3: Demographic/age-bucket popularity within the last 7 days.
     - Source 4: Dense Bi-Encoder / BPR user-to-item cosine top-$K$.
  2. **Stage-1 Recall Auditing:**
     - Every new candidate source was evaluated strictly by whether it increased **Stage-1 Recall@100** without bloating candidate count.
  3. **Stage-2 Listwise GBDT (`lambdarank`):**
     - Combined raw retrieval scores from each candidate source, reciprocal ranks, user-item attribute affinity (brand/color/department match ratio), and within-query normalized scores.

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Stage-1 `evaluate_candidate_recall` auditor, corpus cold-start overlap check, and leak-free `GroupKFold` splitter.
- [`modules/contrastive_mnrl_loss.py`](../modules/contrastive_mnrl_loss.py): `MultipleNegativesRankingLoss` (in-batch + hard negatives with temperature $\tau$ and symmetric loss).
- [`modules/hard_negative_miner.py`](../modules/hard_negative_miner.py): `HardNegativeMiner` (with false-negative margin filtering `sim <= margin * pos_sim`).
- [`modules/rank_fusion_and_qe.py`](../modules/rank_fusion_and_qe.py): `reciprocal_rank_fusion` and `alpha_query_expansion`.
- [`modules/query_context_features.py`](../modules/query_context_features.py): `build_query_context_features` for Stage-2 GBDT reranking.
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Vectorized Okapi BM25 + Word/Char TF-IDF $\rightarrow$ RRF Stage-1 + GBDT Stage-2 reranker.
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Contrastive Bi-Encoder + Hard Negative Mining + $\alpha$-QE + Cross-Encoder reranker.
- [`postprocess.py`](../postprocess.py): Mutual $k$-NN score boosting and dynamic $(\tau_{\text{abs}}, \tau_{\text{rel}})$ candidate cutoff optimization for $F_\beta$.
