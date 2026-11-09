# Task 08: Tabular & Multimodal (Text + Tabular) Modeling

## 1. Formulation & Common Competition Metrics
- **Input:** Structured tabular columns (numerical, categorical, timestamps, entity IDs) optionally paired with free-text fields.
- **Output:** Binary/multiclass probability, continuous regression target, or survival/ranking score.
- **Metrics:** ROC-AUC, LogLoss, RMSE, RMSLE, MAE, MAPE, QWK, Macro-F1.

## 2. Validation Strategy & Leakage Traps
- **Out-of-Fold Feature Engineering Rule:** Target encoding, weight-of-evidence, and target-based aggregations **must** be computed strictly inside each CV fold (with smoothing), never on the full dataset before splitting.
- **Group Aggregation Rule:** Non-target aggregations (`groupby(cat)[num].agg(['mean', 'std', 'min', 'max'])`) can be computed on train or concat(train, test) depending on whether test is observable at inference time.

## 3. Escalation Playbook
- **Tier 1 (Fast GBDT Baseline):** Raw numeric + frequency/ordinal encoded categoricals $\rightarrow$ LightGBM + CatBoost on locked `train_folds.parquet`.
- **Tier 2 (Systematic Feature Engineering + Multimodal Text Fusion):**
  - Numeric ratios/differences, group aggregations (`mean`, `std`, `z-score` relative to group), entity history lags/rolling stats, frequency encoding, OOF smoothed Target Encoding.
  - **Text + Tabular Fusion (Proven Winner Pattern):** Train a standalone text model (TF-IDF+Ridge and/or fine-tuned DeBERTa) on the text columns to produce **Out-Of-Fold logits/probabilities**, then feed those scalar OOF predictions + truncated SVD of embeddings directly as features into LightGBM / CatBoost / XGBoost.
- **Tier 3 (Neural Tabular Diversity + Ensembling):**
  - Add an MLP / ResNet-like tabular net / TabNet / RealMLP trained on scaled + rank-Gauss transformed features for ensemble diversity with GBDTs.

## 4. Directory Contents
- `baselines/` — Standalone GBDT + automated feature engineering & multimodal text-to-tabular fusion script.
- `modules/` — Leak-free OOF target encoder, group aggregation generator, rank-Gauss preprocessor.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions.
