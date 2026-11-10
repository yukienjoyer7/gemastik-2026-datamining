# Task 01: Text Classification (Single-Label Binary & Multiclass)

## 1. Formulation & Common Competition Metrics
- **Input:** Single text field (or concatenated metadata + text: `f"{title} [SEP] {category} [SEP] {body}"`).
- **Output:** Binary or $C$-class label / probability distribution.
- **Metrics:** Macro-F1, Weighted-F1, LogLoss, ROC-AUC, Accuracy, Quadratic Weighted Kappa (for ordinal ratings/scores).

## 2. Validation Strategy & Leakage Traps
- **Default:** `StratifiedKFold(n_splits=5)` on target class.
- **Group Trap:** If multiple texts share an author, user, product ID, or thread ID, use `StratifiedGroupKFold` to prevent memorizing author/entity style.
- **Near-Duplicate Trap:** Check exact/near duplicates before splitting; group duplicates into the same fold.

## 3. Escalation Playbook
- **Tier 1 (CPU, <3 mins):** Word TF-IDF `(1, 3)` + Char-wb TF-IDF `(3, 5)` + Lexical Stats $\rightarrow$ `LogisticRegression` + `CalibratedClassifierCV(LinearSVC)` + `RidgeClassifier`.
- **Tier 2 (GPU Backbone):** DeBERTa-v3 (`microsoft/deberta-v3-small` / `base` or `indobenchmark/indobert-base-p1` / `xlm-roberta-base`) + Mean/WeightedLayer Pooling + Multi-Sample Dropout + LLRD + AWP + EMA.
- **Tier 3 (Tricks & Post-Processing):**
  - Long documents (>512 tokens): Head+Tail truncation (`head_len=128, tail_len=382`) + Dynamic Padding.
  - Regularization: R-Drop, Adversarial Weight Perturbation (AWP), Layer-wise Learning Rate Decay (LLRD), EMA.
  - Post-processing: Temperature Scaling + Nelder-Mead class probability multipliers for Macro-F1 + High-Confidence Pseudo-Labeling.

## 4. Directory Contents
- `eda_and_data.py` — Class balance, per-class length quantiles, train-test vocabulary OOV overlap, Dirichlet-smoothed log-odds n-grams, `TextClassificationDataset` (Head+Tail truncation), and `DynamicPaddingCollator`.
- `baselines/tier1_fast_cpu.py` — Standalone Word `(1,3)` + Char-wb `(3,5)` TF-IDF + Lexical stats OOF ensemble (`LogisticRegression` + `CalibratedClassifierCV(LinearSVC)` + `RidgeClassifier`).
- `baselines/tier2_strong_dl.py` — Standalone PyTorch + HuggingFace OOF trainer with selectable pooling, Multi-Sample Dropout, LLRD, AMP, Cosine schedule, AWP, and ModelEMA.
- `modules/pooling_heads.py` — `MeanPooling`, `MaxPooling`, `MeanMaxPooling`, `AttentionPooling`, `WeightedLayerPooling`, and `GeMPooling`.
- `modules/awp.py` — `AWP` (Adversarial Weight Perturbation) with step/restore and loss threshold triggers.
- `modules/ema.py` — `ModelEMA` (Exponential Moving Average of model weights with warmup-aware decay).
- `modules/llrd_and_schedulers.py` — `get_llrd_optimizer_params` (Layer-wise Learning Rate Decay), linear/cosine warmup schedulers, `MultiSampleDropout`, and `RDropClassificationLoss`.
- `modules/long_text_chunking.py` — `head_tail_truncate_ids`, `split_into_sliding_windows`, and `SlidingWindowChunkAggregator` (mean, max, or attention chunk pooling).
- `postprocess.py` — Temperature scaling calibration, Nelder-Mead class probability multiplier optimizer for Macro-F1, and `select_pseudo_labels()`.
- `winners/SUMMARY.md` — Curated Kaggle & Indonesian NLP competition winning solutions and architectural playbook.
