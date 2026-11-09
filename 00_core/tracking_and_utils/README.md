# `00_core/tracking_and_utils/` — Reproducibility, Memory & Offline Helpers

## Scope
Lightweight boilerplate utilities pasted or imported into standalone scripts.

## Planned Standalone Scripts
- `reproducibility_and_system.py` — `seed_everything()`, PyTorch deterministic flags, AMP (`bfloat16`/`float16`) setup, execution timer context manager, and structured console/file logger.
- `memory_reduction.py` — Safe Pandas/Polars numeric downcasting (`reduce_mem_usage`) without corrupting float precision on targets/IDs.
- `offline_hf_helper.py` — Utilities to download, save, and load Hugging Face models/tokenizers/datasets in strictly air-gapped competition environments (`local_files_only=True`).
