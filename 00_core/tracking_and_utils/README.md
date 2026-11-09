# `00_core/tracking_and_utils/` — Reproducibility, Memory & Offline Helpers

## Scope
Lightweight boilerplate utilities imported or pasted into competition scripts.

## Scripts & CLI Usage
- `reproducibility_and_system.py` — `seed_everything(seed)`, `Timer` context manager, `setup_logger(log_file)`, and `free_memory()` (`gc.collect()` + `torch.cuda.empty_cache()`).
  ```bash
  python3 00_core/tracking_and_utils/reproducibility_and_system.py --smoke-test
  ```
- `memory_reduction.py` — Safe Pandas `reduce_mem_usage(df, exclude_cols=...)` downcasting int/float columns + low-cardinality object-to-category conversion report.
  ```bash
  python3 00_core/tracking_and_utils/memory_reduction.py --input-path train.parquet --output-path train_slim.parquet
  python3 00_core/tracking_and_utils/memory_reduction.py --smoke-test
  ```
- `offline_hf_helper.py` — `download_for_offline(model_name, save_dir)` and strict air-gapped `load_offline_model_and_tokenizer(save_dir, task=...)` (`HF_HUB_OFFLINE=1`, `local_files_only=True`).
  ```bash
  python3 00_core/tracking_and_utils/offline_hf_helper.py --smoke-test
  ```
