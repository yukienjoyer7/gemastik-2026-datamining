# Task 04: Multilabel Classification

## 1. Formulation & Common Competition Metrics
- **Input:** Document / record.
- **Output:** Binary indicator vector $y \in \{0, 1\}^L$ across $L$ non-mutually-exclusive labels (often with severe positive-negative imbalance and label hierarchy/co-occurrence).
- **Metrics:** Micro-F1, Macro-F1, Samples-F1, mean Average Precision (mAP), Macro ROC-AUC, Binary Cross-Entropy.

## 2. Validation Strategy & Leakage Traps
- **Mandatory Splitter:** `MultilabelStratifiedKFold` (from `iterative-stratification`) to preserve rare label combinations across folds.
- **Threshold Overfitting Trap:** Optimizing $L$ per-class thresholds on full OOF can overfit rare classes (<20 positives). Use nested OOF thresholding or regularize rare-class thresholds toward a global prior threshold.

## 3. Escalation Playbook
- **Tier 1 (CPU, <5 mins):** TF-IDF $\rightarrow$ `OneVsRestClassifier(LogisticRegression / Ridge)` or Classifier Chains + LightGBM per label (with OOF predictions of correlated labels as meta-features).
- **Tier 2 (GPU Transformer):** DeBERTa-v3 / XLM-R / IndoBERT with $L$-output head + `BCEWithLogitsLoss` or **Asymmetric Loss (ASL)** / **ZLPR Loss** for heavy positive-negative imbalance.
- **Tier 3 (Label Co-occurrence & Post-Processing):**
  - Label-aware attention / Label Graph / appending label descriptions to input (`[CLS] label_1, label_2 ... [SEP] text`).
  - Per-class threshold optimization + temperature scaling + positive count alignment.

## 4. Directory Contents
- `baselines/` — Standalone TF-IDF OvR/Chain baseline and Transformer Multilabel OOF trainer.
- `modules/` — Asymmetric Loss (ASL), ZLPR Loss, regularized per-class threshold optimizer.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions.
