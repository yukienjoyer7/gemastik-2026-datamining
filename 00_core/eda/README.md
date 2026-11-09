# `00_core/eda/` — Fast Exploratory Data Analysis & Leakage Triage

## Scope
Standalone scripts to audit unfamiliar datasets within the first 15 minutes of a competition.

## Scripts & CLI Usage
- `quick_profile_tabular.py` — Shape, dtypes, missingness %, cardinality, constant/quasi-constant columns, ID/timestamp monotonicity checks, outlier quantiles (`p01, p05, p50, p95, p99`), target distribution, and train-test mismatch (KS-test for numeric, TVD + Chi2 for categorical).
  ```bash
  python3 00_core/eda/quick_profile_tabular.py --train-path train.parquet --test-path test.parquet --target-col target
  python3 00_core/eda/quick_profile_tabular.py --smoke-test
  ```
- `quick_profile_text.py` — Character/word length quantiles (`p10, p50, p90, p95, p99`), empty/whitespace-only string counts, vocabulary overlap & OOV rate between train and test, URL/HTML/emoji/digit/uppercase density, and top class-discriminative unigrams/bigrams via Dirichlet-smoothed log-odds ratio.
  ```bash
  python3 00_core/eda/quick_profile_text.py --train-path train.parquet --test-path test.parquet --text-col text --target-col target
  python3 00_core/eda/quick_profile_text.py --smoke-test
  ```
- `duplicate_and_leak_check.py` — Exact SHA-256 hash deduplication + character n-gram `(3, 5)` TF-IDF cosine similarity near-duplicate search across `train-train` (detecting conflicting labels) and `train-test` (detecting exact/near leakage).
  ```bash
  python3 00_core/eda/duplicate_and_leak_check.py --train-path train.parquet --test-path test.parquet --text-col text --target-col target
  python3 00_core/eda/duplicate_and_leak_check.py --smoke-test
  ```
