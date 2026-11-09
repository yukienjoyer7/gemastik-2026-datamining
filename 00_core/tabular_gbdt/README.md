# `00_core/tabular_gbdt/` — Universal GBDT OOF Pipelines

## Scope
Standalone Out-Of-Fold training, inference, and tuning pipelines for **LightGBM**, **CatBoost**, and **XGBoost** supporting `binary`, `multiclass`, `regression`, and `lambdarank`. Every trainer exports `oof_<exp>.npy`, `test_<exp>.npy`, and `sub_<exp>.csv`.

## Scripts & CLI Usage
- `lgbm_oof_pipeline.py` — LightGBM OOF trainer supporting `binary`, `multiclass`, `regression`, and `lambdarank`, fold-averaged feature importance export, and `HistGradientBoosting` fallback.
  ```bash
  python3 00_core/tabular_gbdt/lgbm_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name lgbm_v1
  python3 00_core/tabular_gbdt/lgbm_oof_pipeline.py --smoke-test
  ```
- `catboost_oof_pipeline.py` — CatBoost OOF trainer with native `cat_features` and `text_features` support.
  ```bash
  python3 00_core/tabular_gbdt/catboost_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name cb_v1
  python3 00_core/tabular_gbdt/catboost_oof_pipeline.py --smoke-test
  ```
- `xgb_oof_pipeline.py` — XGBoost (`tree_method="hist"`, `enable_categorical=True`) OOF trainer.
  ```bash
  python3 00_core/tabular_gbdt/xgb_oof_pipeline.py --train-path train_folds.parquet --test-path test.parquet --task binary --exp-name xgb_v1
  python3 00_core/tabular_gbdt/xgb_oof_pipeline.py --smoke-test
  ```
- `optuna_gbdt_tuner.py` — Leak-free K-Fold CV hyperparameter tuner for LightGBM/XGBoost/CatBoost saving best parameters to JSON.
  ```bash
  python3 00_core/tabular_gbdt/optuna_gbdt_tuner.py --train-path train_folds.parquet --model-family lgbm --task binary --n-trials 30
  python3 00_core/tabular_gbdt/optuna_gbdt_tuner.py --smoke-test
  ```
