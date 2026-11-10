# Winning Solutions & Curated References — Tabular & Multimodal

## 1. Benchmark Competition Hall of Fame

| Competition | Metric | 1st Place Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **Kaggle American Express – Default Prediction** | Amex Metric (Normalized Gini + Recall@4%) | LightGBM `dart` + CatBoost + Tabular ResNet/GRU Ensemble | Entity aggregation hierarchy (`last`, `first`, `mean`, `std`, `min`, `max`, `last - mean`, `last / mean`), denoising/round-noise stripping, and **RankGauss** (`QuantileTransformer`) for Neural Nets. |
| **Kaggle PetFinder.my – Pawpularity / Adoption Prediction** | RMSE / Quadratic Weighted Kappa (QWK) | Multimodal OOF Stacking (Text DeBERTa/TF-IDF + Image Swin + Tabular GBDT + SVR Head) | Feeding **leak-free OOF text/image logits + TruncatedSVD components** into LightGBM/CatBoost + `OptimizedRounder` (Nelder-Mead threshold tuning for QWK). |
| **Mercari Price Suggestion & IEEE-CIS Fraud Detection** | RMSLE / ROC-AUC | Frequency + Group Aggregations + Leak-Free OOF Target Encoding + Ridge/MLP + GBDT | Synthetic entity identification (`card1 + addr1 + D1`), `groupby(entity)[TransactionAmt]` z-scores & ratios, and Out-Of-Fold Ridge stacking of sparse text features. |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Kaggle American Express Default Prediction (1st Place)
- **Problem Formulation:** Predict credit default probability from 180+ anonymized numerical and categorical features across customer statement histories.
- **The Gold-Standard Group & Entity Aggregation Recipe (`GroupAggregationTransformer`):**
  1. For every numerical feature $x$ and grouping entity $g$ (e.g., customer, merchant, category, region), compute:
     - `mean(x)`, `std(x)`, `min(x)`, `max(x)`
     - **Difference from group mean:** `x_diff = x - mean(x)`
     - **Z-score within group:** `x_zscore = (x - mean(x)) / (std(x) + eps)`
     - **Ratio to group mean:** `x_ratio = x / (abs(mean(x)) + eps)`
  2. **Why `x - mean` and `x_zscore` Dominate Feature Importance:**
     Decision trees cannot easily compute `feature_A - feature_B` with axis-aligned splits. Giving GBDTs explicit relative features (`spend_zscore_by_segment`, `last_minus_mean`) consistently produces the single largest CV jump (+0.005 to +0.015 AUC/Gini) in tabular competitions.
- **Architectural Diversity (GBDT + RankGauss Neural Net):**
  - While two slightly different LightGBM seeds barely help when blended, combining **GBDT (`LightGBM` / `CatBoost` / `HistGradientBoosting`)** with a **PyTorch Tabular ResNet / Gated MLP** trained on **RankGauss (`QuantileTransformer(output_distribution='normal')`)** scaled inputs yields substantial ensemble gains because smooth neural linear combinations + `nn.Embedding` tables make orthogonal errors compared to step-function tree splits.

---

### B. PetFinder.my Adoption & Pawpularity (1st Place — Multimodal Fusion)
- **Problem Formulation:** Combine structured tabular attributes (breed, age, fee, vaccinations) with unstructured free-text descriptions (`desc`) and metadata.
- **How to Fuse Text + Tabular Without Overfitting:**
  - **Trap:** Concatenating 768-dim raw Transformer embeddings or 5,000 TF-IDF columns directly alongside 20 tabular columns causes GBDTs to overfit noisy high-dimensional projections while diluting split sampling on core tabular features.
  - **Winning 2-Stage Multimodal Pattern (Implemented in `tier1_fast_cpu.py` & `tier2_strong_dl.py`):**
    1. **Low-Rank Unsupervised Compression:** Fit `TfidfVectorizer(ngram_range=(1, 2))` + `TruncatedSVD(n_components=4..16)` on text columns and append those compact SVD components to the tabular matrix.
    2. **Supervised OOF Scalar Meta-Feature:** Train a dedicated text model (`TfidfVectorizer` + `RidgeClassifier` / `Ridge` or fine-tuned `DeBERTa`) strictly out-of-fold to produce a single scalar column `text_oof_logit` on train and fold-averaged `text_test_logit` on test. Feed that single high-signal scalar directly into GBDT and Neural Tabular models!
- **Ordinal Target Post-Processing (`OptimizedRounder`):**
  - When the target is ordinal (`0, 1, 2, 3, 4`) scored by Quadratic Weighted Kappa (QWK) or Macro-F1, train the model with continuous regression/BCE loss and optimize the $K-1$ class cutoffs $(\tau_1, \dots, \tau_{K-1})$ on OOF predictions using **Nelder-Mead (`scipy.optimize.minimize`)**.

---

### C. IEEE-CIS Fraud Detection & Mercari (1st Place — Leak-Free Target Encoding & Missingness)
- **Leak-Free Empirical Bayes Smoothed OOF Target Encoding (`OOFTargetEncoder`):**
  $$\text{TE}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$
  - Must be computed **strictly inside each CV fold loop** (`folds != f`) when populating validation fold `f`. Computing target encoding on the full training set before splitting leaks the target and inflates CV by 0.02+ while failing on test!
- **Missingness as Signal:**
  - In production databases, `NaN` is rarely random—it reflects which form path, device type, or credit bureau returned a record. Always add `row_na_count` (number of missing columns per row), binary `<col>_isna` indicators, and inspect the missingness correlation matrix (`compute_missingness_correlations` in `eda_and_data.py`).

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Automatic multimodal schema detector (`num_cols`, `cat_low_card`, `cat_high_card`, `text_cols`, `datetime_cols`, `id_cols`), missingness correlation matrix, Pearson/Spearman/Mutual-Information target ranking, and collinearity detector.
- [`modules/oof_target_encoder.py`](../modules/oof_target_encoder.py): `OOFTargetEncoder` (Empirical Bayes smoothed out-of-fold target encoding).
- [`modules/group_aggregations.py`](../modules/group_aggregations.py): `GroupAggregationTransformer` (`mean`, `std`, `min`, `max`, `diff_from_mean`, `zscore`, `ratio_to_mean`).
- [`modules/numeric_interactions_and_rankgauss.py`](../modules/numeric_interactions_and_rankgauss.py): `NumericInteractionGenerator` and `RankGaussScaler`.
- [`modules/datetime_features.py`](../modules/datetime_features.py): `DatetimeFeatureExtractor` (cyclical `sin/cos`, business-day flags, entity lag deltas).
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Automated Feature-Engineered GBDT baseline (Datetime cyclical/lag + Frequency + OOF Target Encoding + Group Aggregations + Text TF-IDF/SVD + Feature Importance export).
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Multimodal Text OOF Logit Stacking + PyTorch `TabularResNetGatedMLP` with learned `nn.Embedding` and `RankGauss` scaling.
- [`postprocess.py`](../postprocess.py): Nelder-Mead GBDT + Neural Tabular OOF blender (`optimize_gbdt_neural_blend`), Rank Averaging, and `OptimizedRounder` for ordinal targets.
