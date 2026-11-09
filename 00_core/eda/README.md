# `00_core/eda/` — Fast Exploratory Data Analysis & Leakage Triage

## Scope
Standalone scripts to audit unfamiliar datasets within the first 15 minutes of a competition.

## Planned Standalone Scripts
- `quick_profile_tabular.py` — Shape, dtypes, missingness patterns, cardinality, constant/quasi-constant columns, ID/timestamp ordering checks, target distribution.
- `quick_profile_text.py` — Character/word/token length percentiles, vocabulary overlap between train and test, language/script detection, special character/HTML/URL noise check.
- `duplicate_and_leak_check.py` — Exact and near-duplicate (MinHash / char n-gram cosine) detection across train-train (label consistency check) and train-test (leakage / lookup opportunities).
