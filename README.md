# GEMASTIK 2026 Data Mining — Technical Reference & Cheatsheet

Practical, competition-oriented repository of standalone Python scripts, validation schemes, modeling playbooks, and distilled winning solutions for **GEMASTIK 2026 (Data Mining Division)**.

---

## 1. Repository Map

### Shared Cross-Task Core (`00_core/`)
 Cross-cutting utilities designed as standalone `.py` modules used across all problem types:
- **[`00_core/README.md`](./00_core/README.md)** — Core module overview & usage workflow.
- **[`00_core/eda/`](./00_core/eda/README.md)** — Fast dataset profiling, train/test distribution drift, duplicate & target leakage detection.
- **[`00_core/validation/`](./00_core/validation/README.md)** — Leak-free CV splitters (`StratifiedKFold`, `GroupKFold`, `MultilabelIterativeStratifiedKFold`, `PurgedGroupTimeSeriesSplit`, Adversarial Validation).
- **[`00_core/metrics_and_losses/`](./00_core/metrics_and_losses/README.md)** — Custom evaluation metrics, threshold/cutpoint optimizers (Nelder-Mead / Optuna for Macro-F1, MCC, QWK), and custom PyTorch losses.
- **[`00_core/tabular_gbdt/`](./00_core/tabular_gbdt/README.md)** — Unified OOF training & inference templates for LightGBM, CatBoost, and XGBoost.
- **[`00_core/ensembling/`](./00_core/ensembling/README.md)** — Out-of-fold hill climbing, Nelder-Mead/Dirichlet blending, rank averaging, power averaging, and Ridge/MLP stacking.
- **[`00_core/tracking_and_utils/`](./00_core/tracking_and_utils/README.md)** — Deterministic seed locking, dataframe memory compression, execution timers, and offline Hugging Face loading helpers.
- **[`00_core/indonesian_nlp/`](./00_core/indonesian_nlp/README.md)** — Indonesian slang/alay/old-spelling normalizer, Judol/Pinjol/toxic de-obfuscator, Pantun/Syair prosody & syllabifier, Nusantara regional dialect detector, and Rupiah/Address/Date/Legal extractors.

### Task-Specific Playbooks & Standalone Templates (`tasks/`)
Each task folder is self-contained with its own decision playbook (`README.md`), standalone baselines (`baselines/`), reusable drop-in tricks (`modules/`), and curated competition solutions (`winners/SUMMARY.md`):

| ID | Task Family | Directory | Primary Focus |
| :--- | :--- | :--- | :--- |
| **01** | **Text Classification** | [`tasks/01_text_classification/`](./tasks/01_text_classification/README.md) | Single/multi-class classification, TF-IDF+Linear/GBDT, DeBERTa/IndoBERT, long-document pooling |
| **02** | **Text Pair & Semantic Similarity** | [`tasks/02_text_pair_similarity/`](./tasks/02_text_pair_similarity/README.md) | Cross-encoder vs. Bi-encoder, transitive graph `GroupKFold`, Cosine/Pearson/Contrastive objectives |
| **03** | **Clustering & Topic Modeling** | [`tasks/03_clustering_topic_modeling/`](./tasks/03_clustering_topic_modeling/README.md) | Sentence embeddings + UMAP/HDBSCAN, BERTopic, TF-IDF + NMF/K-Means, cluster coherence/NMI/ARI |
| **04** | **Multilabel Classification** | [`tasks/04_multilabel_classification/`](./tasks/04_multilabel_classification/README.md) | Iterative stratification, Asymmetric Loss (ASL), ZLPR, label co-occurrence, per-class thresholding |
| **05** | **Retrieval & Ranking** | [`tasks/05_retrieval_and_ranking/`](./tasks/05_retrieval_and_ranking/README.md) | Two-stage retrieval (BM25 + Bi-Encoder) $\rightarrow$ Cross-Encoder / GBDT reranking, hard negative mining, MAP/NDCG/MRR |
| **06** | **Seq2Seq & Text Transformation** | [`tasks/06_seq2seq_text_transform/`](./tasks/06_seq2seq_text_transform/README.md) | Encoder-Decoder (T5/mT5/BART), decoding strategies, GEC/summarization/normalization, BLEU/ROUGE/ChrF |
| **07** | **Token Classification & NER** | [`tasks/07_token_classification_ner/`](./tasks/07_token_classification_ner/README.md) | Subword offset mapping, sliding-window stride aggregation, CRF/Span heads, BIO post-processing, span F1 |
| **08** | **Tabular & Multimodal Fusion** | [`tasks/08_tabular_and_multimodal/`](./tasks/08_tabular_and_multimodal/README.md) | Out-of-fold target encoding, group aggregations, text+tabular fusion (OOF embeddings $\rightarrow$ GBDT) |
| **09** | **Aspect-Based Sentiment (ABSA)** | [`tasks/09_aspect_based_sentiment_absa/`](./tasks/09_aspect_based_sentiment_absa/README.md) | Multi-aspect sentiment (ACSA) & triplet extraction (ASTE), auxiliary-sentence NLI prompts, multi-aspect attention |
| **10** | **Thread & Graph-Structured Text** | [`tasks/10_thread_and_graph_text/`](./tasks/10_thread_and_graph_text/README.md) | Reply trees & citation graphs, `[ROOT]+[PARENT]+[TARGET]` context serialization, OOF graph label propagation |
| **11** | **Semi-Structured Log, Markup & Code** | [`tasks/11_semi_structured_log_and_code/`](./tasks/11_semi_structured_log_and_code/README.md) | HTML/DOM/JSON/XML, Drain log template mining, SQLi/XSS/URL payload entropy, AST/byte-aware classification |
| **12** | **Symbolic, SMILES & Bio Sequences** | [`tasks/12_symbolic_and_bio_sequences/`](./tasks/12_symbolic_and_bio_sequences/README.md) | Chemical SMILES (*Jamu*/drug bioactivity) & DNA/RNA/Protein FASTA, case-sensitive $k$-mers, 1D Dilated ResNet + TTA |

### Offline & Competition Preparation (`offline_prep/`)
- **[`offline_prep/README.md`](./offline_prep/README.md)** — Pre-competition checklist for local wheels, pretrained Hugging Face caches (English + Indonesian/multilingual backbones), and NLTK/spaCy/Sastrawi assets.

---

## 2. Competition Day Triage Protocol (First 60 Minutes)

1. **Minute 0–15 (Understand & Audit):**
   - Identify the exact **evaluation metric** and **submission format**.
   - Run `00_core/eda/` on train & test to inspect missingness, duplicates, near-duplicates, ID/timestamp columns, and train-test distribution shift.
2. **Minute 15–30 (Lock Validation):**
   - Open `tasks/<matched_task>/README.md` to check leakage traps.
   - Generate `train_folds.parquet` using `00_core/validation/` and **never change fold splits** across models you intend to ensemble.
3. **Minute 30–60 (Tier-1 Fast CPU Baseline):**
   - Run `tasks/<matched_task>/baselines/tier1_fast_cpu.py` to get a verified CV score, save `oof_<exp>.npy` + `test_<exp>.npy`, and make a sanity-check submission to confirm CV $\leftrightarrow$ LB correlation.
4. **Hour 1+ (Iterate, Upgrade & Ensemble):**
   - Launch `tier2_strong_dl.py` or feature-engineered GBDT pipelines.
   - Plug in high-ROI components from `tasks/<matched_task>/modules/`.
   - Apply `postprocess.py` (threshold / metric optimization on OOF) and blend diverse model families via `00_core/ensembling/`.

---

## 3. Standalone `.py` Script & OOF Contract

All scripts in this repository are **standalone `.py` files** (no fragile relative package imports required) so they can be executed from the terminal or pasted directly into a notebook cell:

- **Top-of-file Configuration (`class CFG`):** Paths, `seed`, `n_folds`, `model_name`, `target_col`, and hyperparameters are defined in one place at the top of each script.
- **Standardized Fold Input:** Reads `train_folds.parquet` (or `.csv`) containing an explicit integer `fold` column (`0 .. n_folds - 1`).
- **Standardized Artifact Output:**
  - `oof_<exp_name>.npy` — Out-of-fold raw logits or calibrated probabilities aligned with `train_folds` row order (shape: `(N_train, N_classes)` or `(N_train,)`).
  - `test_<exp_name>.npy` — Fold-averaged test predictions aligned with `test` row order.
  - `sub_<exp_name>.csv` — Ready-to-submit file formatted to competition specifications.
