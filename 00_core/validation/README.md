# `00_core/validation/` — Leak-Free Cross-Validation & Splitters

## Scope
Standalone fold-generation and covariate-shift scripts that output a locked `train_folds.parquet` file (`fold` column in `0 .. K-1`) so every team member and model trains on identical splits.

## Scripts & CLI Usage
- `make_folds.py` — Unified fold creator supporting `stratified`, `binned_stratified` (Sturges' rule quantile binning), `group`, `stratified_group`, `pair_connected_components` (Union-Find graph connected components on `(pair_col_a, pair_col_b)`), `multilabel` (`iterative-stratification` or greedy iterative fallback), and `purged_time_series` (`PurgedGroupTimeSeriesSplit` with purge/embargo gap).
  ```bash
  python3 00_core/validation/make_folds.py --input-path train.parquet --output-path train_folds.parquet --scheme stratified --target-col target --n-splits 5
  python3 00_core/validation/make_folds.py --smoke-test
  ```
- `adversarial_validation.py` — Train (`is_test=0`) vs. Test (`is_test=1`) classifier with 5-fold GBDT + optional TF-IDF SVD on text columns. Reports OOF ROC-AUC (warns if `AUC > 0.60`), top drifting features, and density-ratio importance weights $w_i = p(\text{test}|x_i) / (1 - p(\text{test}|x_i))$.
  ```bash
  python3 00_core/validation/adversarial_validation.py --train-path train.parquet --test-path test.parquet --target-col target
  python3 00_core/validation/adversarial_validation.py --smoke-test
  ```
