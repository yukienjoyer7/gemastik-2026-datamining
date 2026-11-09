# Task 03: Clustering & Topic Modeling

## 1. Formulation & Common Competition Metrics
- **Input:** Unlabeled (or partially labeled) corpus of documents/short texts.
- **Output:** Hard cluster assignments, soft topic distributions, or interpretable topic representations.
- **Metrics:** Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), V-Measure, Silhouette Score, Topic Coherence ($C_v$ / NPMI).

## 2. Validation Strategy & Pitfalls
- **Hyperparameter Sensitivity:** UMAP `n_neighbors`/`min_dist` and HDBSCAN `min_cluster_size` drastically change cluster count and noise (`-1`) ratio.
- **HDBSCAN Noise (`-1`) Trap:** Competitions requiring every sample to have a valid cluster ID cannot leave `-1` unassigned—always re-assign `-1` outliers via $k$-NN / cosine similarity to nearest cluster centroids or `approximate_predict`.

## 3. Escalation Playbook
- **Tier 1 (CPU Fast):** TF-IDF (L2 normalized) + TruncatedSVD (LSA) + Spherical K-Means (`MiniBatchKMeans` on unit vectors) or NMF/LDA.
- **Tier 2 (Dense Embedding + Manifold Clustering):** Multilingual/Indonesian Sentence Embeddings (`bge-m3`, `multilingual-e5-base`, `paraphrase-multilingual-mpnet-base-v2`) $\rightarrow$ L2 normalize $\rightarrow$ UMAP (`metric='cosine'`) $\rightarrow$ K-Means / HDBSCAN + outlier reassignment + c-TF-IDF (BERTopic pipeline).
- **Tier 3 (Ensemble / Consensus Clustering):** Co-association matrix / consensus clustering across multiple embedding models and $K$ values; SimCSE / domain-adapted contrastive fine-tuning before clustering.

## 4. Directory Contents
- `baselines/` — Standalone TF-IDF+SVD+KMeans and Embedding+UMAP+HDBSCAN/BERTopic pipelines.
- `modules/` — HDBSCAN `-1` outlier reassignment, cluster consensus/ensembling, coherence & intrinsic evaluators.
- `winners/SUMMARY.md` — Curated references & implementations.
