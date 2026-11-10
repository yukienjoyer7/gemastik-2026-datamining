# Task 08: Tabular & Multimodal (Text + Tabular) Modeling

## 1. Formulation & Common Competition Metrics
- **Input:** Structured tabular columns (numerical, categorical, timestamps, entity IDs) optionally paired with free-text fields.
- **Output:** Binary/multiclass probability, continuous regression target, or ordinal score.
- **Metrics:** ROC-AUC, LogLoss, RMSE, RMSLE, MAE, Quadratic Weighted Kappa (QWK), Macro-$F_1$.

## 2. Validation Strategy & Leakage Traps
- **Out-of-Fold Feature Engineering Rule:** Target encoding (`OOFTargetEncoder`) and supervised text meta-features (`generate_oof_text_meta_feature`) **must** be computed strictly inside each CV fold (with smoothing), never on the full dataset before splitting.
- **Group Aggregation Rule:** Non-target aggregations (`groupby(cat)[num].agg(['mean', 'std', 'min', 'max'])` + `z-score` + `ratio-to-mean`) are computed via `GroupAggregationTransformer`.

## 3. Escalation Playbook
- **Tier 1 (Automated Feature-Engineered GBDT Baseline — `tier1_fast_cpu.py`):**
  - Row missingness count + per-column `<col>_isna` flags, datetime cyclical (`sin`/`cos`) & lag features, frequency encoding, leak-free OOF smoothed Target Encoding, group aggregations (`mean`, `std`, `min`, `max`, `diff`, `zscore`, `ratio`), and TF-IDF + `TruncatedSVD` for text columns $\rightarrow$ OOF `LightGBM` / `HistGradientBoostingClassifier`.
- **Tier 2 (Multimodal Text Fusion + Neural Tabular Diversity — `tier2_strong_dl.py`):**
  - Stage A: Leak-free OOF text model (`TF-IDF` + `Ridge`) generating scalar `text_oof_logit` meta-features.
  - Stage B: PyTorch **Tabular ResNet / Gated MLP** (`TabularResNetGatedMLP`) with learned `nn.Embedding` for categoricals + **RankGauss (`QuantileTransformer`)** for numericals + Cosine Annealing & Weight Decay.
- **Tier 3 (Ensembling & Post-Processing — `postprocess.py`):**
  - Out-of-fold Nelder-Mead / Rank-Averaging optimizer blending `oof_tier1_tabular.npy` (GBDT) and `oof_tier2_neural.npy` (Neural Tabular) + `OptimizedRounder` for ordinal/integer targets.

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Automatic tabular + multimodal schema detector, missingness correlation matrix, target correlation ranking (Pearson + Spearman + Mutual Information), and high-collinearity pair detector.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Automated Feature-Engineered GBDT Baseline exporting `oof_tier1_tabular.npy`, `test_tier1_tabular.npy`, `sub_tier1_tabular.csv`, and feature importance table.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Multimodal Text+Tabular OOF Fusion & PyTorch `TabularResNetGatedMLP` with `RankGauss` scaling exporting `oof_tier2_neural.npy`.
- [`modules/oof_target_encoder.py`](modules/oof_target_encoder.py) — `OOFTargetEncoder` (Empirical Bayes smoothed out-of-fold target encoding for binary, multiclass, and continuous targets).
- [`modules/group_aggregations.py`](modules/group_aggregations.py) — `GroupAggregationTransformer` (`mean`, `std`, `min`, `max`, `diff_from_mean`, `zscore`, `ratio_to_mean` grouped by categorical entities).
- [`modules/numeric_interactions_and_rankgauss.py`](modules/numeric_interactions_and_rankgauss.py) — `NumericInteractionGenerator` (top-correlated ratios, diffs, products) + `RankGaussScaler` (`QuantileTransformer` normal projection for neural nets).
- [`modules/datetime_features.py`](modules/datetime_features.py) — `DatetimeFeatureExtractor` (cyclical sin/cos encodings, business-day/weekend flags, elapsed deltas, entity time-gap lag features).
- [`postprocess.py`](postprocess.py) — Nelder-Mead / Rank-Averaging GBDT + Neural Tabular blender and `OptimizedRounder`.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Detailed breakdown of winning solutions (AmEx Default Prediction 1st, PetFinder.my 1st, Mercari / IEEE-CIS Fraud Detection 1st).
