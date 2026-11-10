# Task 03: Clustering & Topic Modeling

## 1. Formulation & Common Competition Metrics
- **Input:** Unlabeled (or partially labeled) corpus of documents/short texts.
- **Output:** Hard cluster assignments ($0 \dots K-1$), soft topic distributions, or interpretable topic keyword representations.
- **Metrics:** Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), V-Measure, Silhouette Score, Topic Coherence ($C_v$ / NPMI).

## 2. Validation Strategy & Pitfalls
- **Hyperparameter Sensitivity:** UMAP `n_neighbors`/`min_dist` and HDBSCAN `min_cluster_size` drastically change cluster count and noise (`-1`) ratio.
- **HDBSCAN Noise (`-1`) Trap:** Competitions requiring every sample to have a valid cluster ID cannot leave `-1` unassigned—always re-assign `-1` outliers via cosine similarity to nearest cluster centroids or $k$-NN (`reassign_hdbscan_outliers`).

## 3. Escalation Playbook
- **Tier 1 (CPU Fast):** Word `(1, 2)` + Char-wb `(3, 5)` TF-IDF $\rightarrow$ `TruncatedSVD` (LSA) + L2 `Normalizer` $\rightarrow$ **Spherical K-Means** + **NMF** topic modeling + automatic $K$ selection scan + c-TF-IDF keyword extraction.
- **Tier 2 (Dense Embedding + Manifold Clustering):** Multilingual/Indonesian Sentence Embeddings (`bge-m3`, `multilingual-e5-base`, `paraphrase-multilingual-mpnet-base-v2`) $\rightarrow$ L2 normalize $\rightarrow$ UMAP / Spectral manifold reduction $\rightarrow$ HDBSCAN / K-Means + **Mandatory `-1` Outlier Reassignment** + c-TF-IDF with Maximal Marginal Relevance (MMR).
- **Tier 3 (Ensemble / Consensus & Capacity Post-Processing):** `ConsensusClusterer` (Hungarian label alignment & co-association matrix clustering) + Sinkhorn sharpening and exact capacity-constrained assignment (`linear_sum_assignment`).

## 4. Directory Contents
- `eda_and_data.py` — Unsupervised corpus diagnostics: document length quantiles, exact duplicate & boilerplate header/footer detector, Indonesian+English stopword ratio, Zipf power-law slope/$R^2$, and Hopkins statistic clusterability check.
- `baselines/tier1_fast_cpu.py` — Standalone Word+Char TF-IDF + LSA + Spherical K-Means + NMF pipeline with automatic $K$ scan (Silhouette / Calinski-Harabasz / Davies-Bouldin) and top-10 c-TF-IDF topic extraction.
- `baselines/tier2_strong_dl.py` — Standalone BERTopic-style Dense Embedding + UMAP/Spectral + HDBSCAN/K-Means + mandatory `-1` outlier reassignment + c-TF-IDF/MMR topic representation.
- `modules/hdbscan_outlier_reassignment.py` — `reassign_hdbscan_outliers()` (soft cosine centroid similarity & $k$-NN noise `-1` reassignment with confidence scores).
- `modules/consensus_clustering.py` — `ConsensusClusterer` (Hungarian algorithm `linear_sum_assignment` label alignment + co-association matrix hierarchical ensemble).
- `modules/c_tfidf_and_mmr_topics.py` — `compute_c_tfidf_topics()` + Maximal Marginal Relevance (MMR) diverse keyword selection per cluster.
- `postprocess.py` — 1-step DEC/Sinkhorn self-training centroid refinement and balanced cluster size assignment via `linear_sum_assignment`.
- `winners/SUMMARY.md` — Curated guide covering BERTopic architecture, SimCSE unsupervised contrastive adaptation, consensus clustering, and handling HDBSCAN noise.
