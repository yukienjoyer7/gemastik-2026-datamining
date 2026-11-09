# `00_core/ensembling/` — OOF Ensembling, Blending & Stacking

## Scope
Late-stage performance multipliers that consume standardized `oof_<exp>.npy` and `test_<exp>.npy` arrays from any task or model family and output `oof_<blend>.npy`, `test_<blend>.npy`, and `sub_<blend>.csv`.

## Scripts & CLI Usage
- `hill_climbing_ensemble.py` — Greedy forward stepwise OOF Hill Climbing (`allow_negative_weights=True`, weight grid `np.linspace(-0.25, 0.50, 151)`).
  ```bash
  python3 00_core/ensembling/hill_climbing_ensemble.py --oof-paths oof_m1.npy oof_m2.npy --test-paths test_m1.npy test_m2.npy --train-path train_folds.parquet --metric roc_auc
  python3 00_core/ensembling/hill_climbing_ensemble.py --smoke-test
  ```
- `nelder_mead_blender.py` — `scipy.optimize.minimize` (`SLSQP` / `Nelder-Mead`) continuous simplex blender + post-blend temperature scaling.
  ```bash
  python3 00_core/ensembling/nelder_mead_blender.py --oof-paths oof_m1.npy oof_m2.npy --test-paths test_m1.npy test_m2.npy --train-path train_folds.parquet --objective logloss
  python3 00_core/ensembling/nelder_mead_blender.py --smoke-test
  ```
- `rank_and_power_averaging.py` — Percentile rank averaging (`scipy.stats.rankdata`) and power averaging ($\sum w_m p_m^\gamma$) with OOF $\gamma$ optimization.
  ```bash
  python3 00_core/ensembling/rank_and_power_averaging.py --oof-paths oof_m1.npy oof_m2.npy --test-paths test_m1.npy test_m2.npy --train-path train_folds.parquet --metric roc_auc
  python3 00_core/ensembling/rank_and_power_averaging.py --smoke-test
  ```
- `stacking_meta_learner.py` — Leak-free 2nd-stage OOF Stacking (`logreg`, `ridge`, `mlp`) using identical `fold` indices from `train_folds` plus nonlinear row-wise meta-features (`mean`, `std`, `min`, `max`, `spread`, `logit`, pairwise diffs).
  ```bash
  python3 00_core/ensembling/stacking_meta_learner.py --oof-paths oof_m1.npy oof_m2.npy --test-paths test_m1.npy test_m2.npy --train-path train_folds.parquet --meta-model logreg
  python3 00_core/ensembling/stacking_meta_learner.py --smoke-test
  ```
