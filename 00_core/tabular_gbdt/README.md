# `00_core/tabular_gbdt/` — Universal GBDT OOF Pipelines

## Scope
Standalone Out-Of-Fold training and inference pipelines for **LightGBM**, **CatBoost**, and **XGBoost** supporting classification, regression, ranking (`lambdarank` / `YetiRank`), and multimodal OOF feature stacking.

## Planned Standalone Scripts
- `lgbm_oof_pipeline.py` — LightGBM OOF trainer with categorical handling, early stopping, fold-averaged importance plot/table, and standardized `oof_*.npy` / `test_*.npy` export.
- `catboost_oof_pipeline.py` — CatBoost OOF trainer with native `cat_features` and `text_features` support (strong baseline for mixed text + tabular datasets).
- `xgb_oof_pipeline.py` — XGBoost (`hist` tree method) OOF trainer.
- `optuna_gbdt_tuner.py` — Leak-free CV hyperparameter tuning template with pruning.
