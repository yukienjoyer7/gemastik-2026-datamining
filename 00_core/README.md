# `00_core/` — Shared Cross-Task Utilities

Standalone Python utilities shared across all tasks in `tasks/01..08`. Every script in `00_core/` is designed to be dependency-light, copy-pasteable, and CLI-runnable.

## Submodules

| Directory | Purpose |
| :--- | :--- |
| [`eda/`](./eda/README.md) | Automated tabular & text profiling, duplicate/near-duplicate detection, train-test drift check |
| [`validation/`](./validation/README.md) | Leak-free CV fold generation (`train_folds.parquet`) + Adversarial Validation |
| [`metrics_and_losses/`](./metrics_and_losses/README.md) | Competition evaluation metrics, threshold/cutpoint optimizers, custom PyTorch/GBDT losses |
| [`tabular_gbdt/`](./tabular_gbdt/README.md) | Universal LightGBM, CatBoost, and XGBoost OOF training, feature importance, and Optuna tuning |
| [`ensembling/`](./ensembling/README.md) | OOF Hill Climbing, Nelder-Mead blending, Rank Averaging, Power Averaging, Stacking |
| [`tracking_and_utils/`](./tracking_and_utils/README.md) | Deterministic seed locking, DataFrame memory reduction, timers/loggers, offline HF helpers |
