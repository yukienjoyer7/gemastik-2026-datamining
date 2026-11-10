# `00_core/` — Shared Cross-Task Utilities

Standalone Python utilities shared across all tasks in `tasks/01..08`. Every script in `00_core/` is self-contained, includes a top-of-file `@dataclass class CFG`, adheres to the standardized OOF/prediction contract (`train_folds.parquet`, `oof_<exp>.npy`, `test_<exp>.npy`, `sub_<exp>.csv`), and supports `--smoke-test`.

## Submodules

| Directory | Scripts | Purpose |
| :--- | :--- | :--- |
| [`eda/`](./eda/README.md) | `quick_profile_tabular.py`, `quick_profile_text.py`, `duplicate_and_leak_check.py` | Automated tabular & text profiling, duplicate/near-duplicate detection, train-test drift check |
| [`validation/`](./validation/README.md) | `make_folds.py`, `adversarial_validation.py` | Leak-free CV fold generation (`train_folds.parquet`) + Adversarial Validation |
| [`metrics_and_losses/`](./metrics_and_losses/README.md) | `threshold_optimizers.py`, `pytorch_losses.py`, `ranking_and_ir_metrics.py` | Competition evaluation metrics, threshold/cutpoint optimizers, custom PyTorch losses |
| [`tabular_gbdt/`](./tabular_gbdt/README.md) | `lgbm_oof_pipeline.py`, `catboost_oof_pipeline.py`, `xgb_oof_pipeline.py`, `optuna_gbdt_tuner.py` | Universal LightGBM, CatBoost, and XGBoost OOF training, feature importance, and Optuna tuning |
| [`ensembling/`](./ensembling/README.md) | `hill_climbing_ensemble.py`, `nelder_mead_blender.py`, `rank_and_power_averaging.py`, `stacking_meta_learner.py` | OOF Hill Climbing, Nelder-Mead/SLSQP blending, Rank/Power Averaging, 2nd-Stage Stacking |
| [`tracking_and_utils/`](./tracking_and_utils/README.md) | `reproducibility_and_system.py`, `memory_reduction.py`, `offline_hf_helper.py` | Deterministic seed locking, DataFrame memory reduction, timers/loggers, offline HF helpers |
| [`indonesian_nlp/`](./indonesian_nlp/README.md) | `slang_and_alay_normalizer.py`, `deobfuscator_judol_toxic.py`, `prosody_pantun_puisi.py`, `nusantara_dialect_detector.py`, `indonesian_domain_extractors.py` | Indonesian slang/alay/old-spelling normalizer, Judol/Pinjol/toxic de-obfuscator, Pantun/Syair prosody & syllabifier, Nusantara dialect detector, and Rupiah/Address/Date/Legal extractors |

## Quick Verification
```bash
python3 -m py_compile 00_core/*/*.py
for s in 00_core/*/*.py; do python3 "$s" --smoke-test; done
```
