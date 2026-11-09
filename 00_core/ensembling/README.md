# `00_core/ensembling/` — OOF Ensembling, Blending & Stacking

## Scope
Late-stage performance multipliers that consume standardized `oof_<exp>.npy` and `test_<exp>.npy` arrays from any task or model family.

## Planned Standalone Scripts
- `hill_climbing_ensemble.py` — Forward stepwise greedy hill climbing (with optional negative weights) on OOF predictions for any custom competition metric.
- `nelder_mead_blender.py` — Scipy Nelder-Mead / SLSQP simplex weight optimization + temperature calibration.
- `rank_and_power_averaging.py` — Rank averaging and power averaging for ROC-AUC, PR-AUC, and ranking metrics where raw probability calibration differs between models.
- `stacking_meta_learner.py` — Out-of-fold 2nd-stage meta-learner (Ridge / LogisticRegression / shallow GBDT / MLP) with strict nested leak prevention.
