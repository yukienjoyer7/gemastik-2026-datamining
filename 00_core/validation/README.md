# `00_core/validation/` — Leak-Free Cross-Validation & Splitters

## Scope
Standalone fold-generation scripts that output a locked `train_folds.parquet` file (`fold` column in `0 .. K-1`) so every team member and model trains on identical splits.

## Quick Selection Rule
- **Standard Classification / Multiclass:** `StratifiedKFold`
- **Continuous Target (Regression / Similarity Score):** Binned `StratifiedKFold` (Sturges' rule bins)
- **Shared Entities (User ID, Author, Document Source, Query ID):** `GroupKFold` or `StratifiedGroupKFold`
- **Text Pairs / Entity Resolution (`(A, B)` pairs):** Connected-components `GroupKFold` (prevents transitive leakage where `(A, B)` is in train and `(B, C)` is in val)
- **Multilabel Classification:** `MultilabelStratifiedKFold` (`iterative-stratification`)
- **Temporal / Forecasting:** `PurgedGroupTimeSeriesSplit` / expanding window split

## Planned Standalone Scripts
- `make_folds.py` — Unified CLI/standalone script supporting all CV schemes above + connected-component grouping for pair tasks.
- `adversarial_validation.py` — Train vs. Test classifier (`is_test` target) with LightGBM + TF-IDF/tabular features to quantify covariate shift (ROC-AUC) and identify drifting features or validation subsets matching the test set.
