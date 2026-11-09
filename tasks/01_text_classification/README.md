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
- **Tier 1 (CPU, <3 mins):** Word TF-IDF `(1, 2)` + Char-wb TF-IDF `(3, 5)` $\rightarrow$ `RidgeClassifier` / `LogisticRegression(C=...)` / `SGDClassifier(loss='log_loss')` + CatBoost with native `text_features`.
- **Tier 2 (GPU Backbone):** DeBERTa-v3 (`microsoft/deberta-v3-small` / `base` or `indobenchmark/indobert-base-p1` / `xlm-roberta-base`) + MeanPooling + Multi-Sample Dropout + Cosine Schedule.
- **Tier 3 (Tricks & Ensembling):**
  - Long documents (>512 tokens): Head+Tail truncation or Sliding Window + Attention/Mean aggregation across chunks.
  - Regularization: R-Drop, Adversarial Weight Perturbation (AWP), Layer-wise Learning Rate Decay (LLRD), EMA.
  - Post-processing: Nelder-Mead class probability multipliers / threshold tuning for Macro-F1; blending TF-IDF + GBDT + Transformer OOFs.

## 4. Directory Contents
- `baselines/` — Standalone end-to-end training scripts (`tier1_fast_cpu.py`, `tier2_deberta_oof.py`).
- `modules/` — Standalone pooling heads (Mean, Max, Attention, WeightedLayer), AWP, EMA, R-Drop.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions and repository references.
