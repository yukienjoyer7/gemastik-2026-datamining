# Winning Solutions & Curated References — Task 03: Clustering & Topic Modeling

## 1. Benchmark & Competition Winning Solutions Table

| Competition / Benchmark | Task & Metric | Winning Pipeline & Backbone | Key Winning Techniques |
| :--- | :--- | :--- | :--- |
| **BERTopic Architecture (Grootendorst, 2022) & MTEB Clustering** | Unsupervised Topic Discovery & Document Clustering (NMI / ARI / Topic Coherence $C_v$) | `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `paraphrase-multilingual-mpnet-base-v2` | **Decoupled 4-Stage Pipeline:** Dense Sentence Embeddings $\rightarrow$ L2 Normalize $\rightarrow$ **UMAP** (`metric='cosine', min_dist=0.0`) $\rightarrow$ **HDBSCAN** + **c-TF-IDF** with **Maximal Marginal Relevance (MMR)** keyword diversification. |
| **SimCSE Unsupervised Domain Adaptation (Gao et al., EMNLP 2021)** | Domain-Specific Short Text & Scientific Clustering (V-Measure / NMI) | `indobenchmark/indobert-base-p1` / `RoBERTa-base` with SimCSE InfoNCE pretraining | Before clustering a specialized corpus (legal, medical, Indonesian social media), run 1 epoch of **unsupervised SimCSE** (passing each sentence twice with different dropout masks as positive pairs). Fixes anisotropic embedding space collapse and boosts clustering NMI by **+4% to +9%**. |
| **Consensus / Co-Association Clustering Ensembles (Strehl & Ghosh / Kaggle Unsupervised Challenges)** | Hard Partitioning under Seed & Hyperparameter Variance (ARI / NMI) | Multi-view ensemble combining Spherical K-Means (LSA), Spectral/UMAP K-Means, and NMF | Aligns cluster IDs across $M$ diverse runs via **Hungarian Bipartite Matching** (`scipy.optimize.linear_sum_assignment`) or builds the $N \times N$ **Co-Association Matrix** $C_{ij}$ followed by average-linkage Agglomerative Clustering. |
| **Constrained / Balanced Cluster Post-Processing (Sinkhorn & Hungarian Capacity Assignment)** | Competition Clustering with Uniform or Known Cluster Priors (Accuracy / Macro-F1 / NMI) | DEC Self-Training Sharpening + Minimum-Cost Bipartite Matching | Standard `argmax` over soft cluster probabilities suffers from majority-cluster collapse. Applying **Sinkhorn-Knopp scaling** + **Hungarian capacity assignment** (`postprocess.py`) enforces exact or prior-matched cluster sizes. |

---

## 2. Distilled Architectural Playbook

### A. Handling HDBSCAN `-1` Noise in Competitive Evaluations (`modules/clustering_tricks.py`)
- In real-world exploration, HDBSCAN leaving 15%–40% of documents as `-1` (noise) is helpful for purity.
- **In competition evaluations (NMI, ARI, V-Measure, Macro-F1), leaving `-1` untouched is fatal:** every `-1` point is treated as a single artificial "noise cluster," destroying ARI and NMI.
- Always call `reassign_hdbscan_outliers(embeddings, labels, method="centroid_cosine")` (`modules/clustering_tricks.py` & `baselines/tier2_strong_dl.py`) to assign `-1` points to their closest unit-normalized cluster centroid in the original high-dimensional embedding space.

### B. Why Spherical K-Means + NMF Beats Vanilla K-Means on Text (`baselines/tier1_fast_cpu.py`)
- Raw TF-IDF vectors live on a high-dimensional sparse simplex where Euclidean distance is dominated by document length.
- Applying `TruncatedSVD` (32–128 components) followed by **L2 Normalization** projects documents onto the unit hypersphere where squared Euclidean distance $\|u - v\|_2^2 = 2(1 - \cos(u, v))$ is strictly monotonic with cosine similarity.
- Blending Spherical K-Means soft probabilities with Hungarian-aligned **NMF** topic weights combines global geometric separation with additive part-based topic interpretability.

### C. Consensus Ensembling Across Embedding Backbones (`modules/clustering_tricks.py`)
- Clustering has no validation loss to early-stop on, making single runs brittle to random seed and UMAP `n_neighbors`.
- Run 5–15 clustering partitions across different backbones (`TF-IDF LSA`, `multilingual-e5-base`, `bge-m3`) and UMAP seeds, then fuse them with `ConsensusClusterer`.
