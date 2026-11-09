# Task 04: Multilabel Classification

## 1. Formulation & Common Competition Metrics
- **Input:** Document / record.
- **Output:** Binary indicator vector $y \in \{0, 1\}^L$ across $L$ non-mutually-exclusive labels (often with severe positive-negative imbalance and label hierarchy/co-occurrence).
- **Metrics:** Micro-F1, Macro-F1, Samples-F1, mean Average Precision (mAP), Macro ROC-AUC, Binary Cross-Entropy.

## 2. Validation Strategy & Leakage Traps
- **Mandatory Splitter:** `MultilabelStratifiedKFold` (from `iterative-stratification`) to preserve rare label combinations across folds.
- **Threshold Overfitting Trap:** Optimizing $L$ per-class thresholds on full OOF can overfit rare classes (<20 positives). Use nested OOF thresholding and shrink rare-class thresholds toward the global prior threshold $\tau_{\text{global}}$ based on positive support $n_c$.

## 3. Escalation Playbook
- **Tier 1 (CPU, <5 mins):** Word + Char-wb TF-IDF $\rightarrow$ **2-Stage Out-Of-Fold Classifier Chain** (Stage 1 independent OOF `LogisticRegression` + `Ridge`, Stage 2 stacking SVD features + Stage-1 OOF probabilities + conditional co-occurrence dot products into regularized linear + GBDT meta-models).
- **Tier 2 (GPU Transformer):** DeBERTa-v3 / XLM-R / IndoBERT with **Label-Aware Prompting** + **Label-Wise Attention Head** + `BCEWithLogitsLoss` / **Asymmetric Loss (ASL)** / **ZLPR Loss**.
- **Tier 3 (Hierarchy & Regularized Threshold Post-Processing):**
  - Automatic parent-child hierarchy rule discovery ($P(y_j=1 \mid y_i=1) \ge 0.98$) and `HierarchicalConstraintEnforcer`.
  - Support-regularized per-class threshold optimization + minimum-1-label argmax fallback.

## 4. Directory Contents
- `eda_and_data.py` — Multilabel diagnostics: head vs. tail label frequency distribution, label cardinality histogram, zero-label sample check, pairwise co-occurrence matrix, and conditional probability $P(y_j=1 \mid y_i=1)$ hierarchy rule detector.
- `baselines/tier1_fast_cpu.py` — Standalone Word+Char TF-IDF + SVD **2-Stage Out-Of-Fold Classifier Chain** exploiting label correlations without leakage.
- `baselines/tier2_strong_dl.py` — Standalone Multi-label Transformer OOF trainer supporting `BCEWithLogitsLoss`, `AsymmetricLossOptimized (ASL)`, and `ZLPRLoss`, plus **Label-Aware Prompting** and **Label-Wise Attention Head**.
- `modules/multilabel_losses_and_heads.py` — Reusable `AsymmetricLoss`, `ZLPRLoss`, `LabelWiseAttentionHead`, and `HierarchicalConstraintEnforcer`.
- `postprocess.py` — Empirical-Bayes support-regularized per-class threshold optimizer + minimum-1-label argmax fallback.
- `winners/SUMMARY.md` — Detailed breakdown of Kaggle Mechanisms of Action (MoA), Jigsaw Toxic Comment Multilabel, and Planet Amazon winning solutions.
