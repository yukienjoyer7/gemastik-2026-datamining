# `offline_prep/` — Offline Competition Environment & Pre-Download Checklist

## Scope
Ensure zero dependency on live internet access, Hugging Face Hub availability, or PyPI servers during the GEMASTIK 2026 final round.

## 1. Pretrained Model Cache Checklist (Hugging Face)
### Multilingual & Indonesian Backbones
- **Classification / NER / Pair Cross-Encoder:**
  - `indobenchmark/indobert-base-p1` / `indobenchmark/indobert-large-p1`
  - `indolem/indobertweet-base-uncased` (for social media / informal Indonesian text)
  - `xlm-roberta-base` / `xlm-roberta-large`
  - `microsoft/mdeberta-v3-base`
- **English Backbones:**
  - `microsoft/deberta-v3-small` / `microsoft/deberta-v3-base` / `microsoft/deberta-v3-large`
- **Dense Retrieval / Embeddings / Clustering:**
  - `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3`
  - `intfloat/multilingual-e5-small` / `intfloat/multilingual-e5-base`
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Seq2Seq:**
  - `google/mt5-small` / `google/mt5-base`
  - `indobenchmark/indobart-v2`

## 2. Offline Python Wheels Checklist (`pip download`)
- **Core ML & GBDT:** `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `scipy`
- **Validation & Ensembling:** `iterative-stratification`
- **NLP & Retrieval:** `transformers`, `tokenizers`, `datasets`, `accelerate`, `sentence-transformers`, `rank-bm25`, `bm25s`, `faiss-cpu`, `seqeval`, `sacrebleu`, `rouge-score`, `rapidfuzz`, `PySastrawi`
- **Clustering & Topic Modeling:** `umap-learn`, `hdbscan`, `bertopic`
