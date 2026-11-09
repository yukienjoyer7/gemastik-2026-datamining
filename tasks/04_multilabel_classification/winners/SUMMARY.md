# Winning Solutions & Curated References — Task 04: Multilabel Classification

## 1. Competition Winning Solutions Table

| Competition / Source | Task & Metric | Winning Architecture | Key Winning Techniques |
| :--- | :--- | :--- | :--- |
| **Kaggle Mechanisms of Action (MoA) Prediction (1st–3rd Place)** | 206-Target Multilabel Binary Classification with Extreme Sparsity (LogLoss) | Blend of ResNet-style MLP, 1D-CNN, and TabNet | **Transfer Learning from Auxiliary Non-Scored Targets (402 aux labels):** Pretrained the backbone on `scored + non_scored` labels before fine-tuning on the 206 scored labels; **Label Smoothing (`0.0005`–`0.001`)** to prevent extreme log-loss penalties on rare false negatives; `MultilabelStratifiedKFold` (from `iterative-stratification`). |
| **Kaggle Jigsaw Toxic Comment Classification Challenge (1st Place)** | 6-Label Toxicity Multilabel Classification (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) (Macro ROC-AUC) | Ensemble of Word+Char TF-IDF 2-Stage Stacking + BiGRU/DeBERTa with Attention & MeanMax Pooling | **2-Stage OOF Classifier Chain (`baselines/tier1_fast_cpu.py`):** Because `severe_toxic == 1` implies `toxic == 1` ($P(\text{toxic} \mid \text{severe\_toxic}) \approx 1.0$), feeding Stage-1 OOF probabilities of all 6 labels into a Stage-2 meta-model yielded a massive boost on rare correlated labels. **Rank-averaging** before blending linear and neural models. |
| **Kaggle Planet: Understanding the Amazon from Space (1st Place)** | 17-Label Multilabel Tagging with Co-Occurrence & Mutual Exclusion Rules ($F_2$ Score) | Ensemble of ConvNets + Coordinate-Ascent Threshold Optimizer + Co-Occurrence Post-Processing | **Coordinate-Ascent Per-Class Threshold Optimization:** Tuned $\tau_c \in (0.1, 0.9)^{17}$ out-of-fold to directly maximize $F_\beta$ while shrinking rare-label thresholds toward the global prior (`postprocess.py`); enforced label hierarchy & mutual-exclusion constraints. |

---

## 2. Distilled Architectural Playbook

### A. Loss Function Selection for Heavy Imbalance (`modules/multilabel_losses_and_heads.py`)
1. **Standard `BCEWithLogitsLoss`:** Strong default when evaluation metric is **LogLoss** or **ROC-AUC** (paired with slight label smoothing `1e-3` if LogLoss is used).
2. **`AsymmetricLoss (ASL)` ($\gamma_- = 4.0, \gamma_+ = 0.0, m = 0.05$):** Best when positive density is $< 5\%$ and evaluation metric is **mAP / Macro-F1 / Micro-F1**. Hard-thresholds easy negatives via probability margin $m$ and down-weights remaining negatives by $(p_m)^4$ without attenuating rare positive gradients.
3. **`ZLPRLoss` (Su et al., 2022):** Best when labels have strong intra-sample competition or variable cardinality per document. Directly enforces $s_{\text{pos}} > 0 > s_{\text{neg}}$.

### B. Exploiting Label Semantics & Hierarchy (`baselines/tier2_strong_dl.py`)
- **Label-Aware Prompting:** Prepending `"candidate labels: {label_1}, {label_2}, ... | document: {text}"` allows early Transformer self-attention layers to match document spans directly to label tokens—crucial for zero/few-shot tail classes.
- **Label-Wise Attention Head (`LabelWiseAttentionHead`):** Instead of compressing a 500-word document into a single mean-pooled vector for 50 distinct labels, each label $l$ uses its own learned query vector $q_l \in \mathbb{R}^D$ to attend to the specific sentence where label $l$ is expressed.
- **Hierarchical Implication Enforcement (`eda_and_data.py` & `HierarchicalConstraintEnforcer`):** Run `compute_cooccurrence_and_implications` during EDA. Whenever $P(y_{\text{parent}}=1 \mid y_{\text{child}}=1) = 1.0$ in train, enforce $P(\text{parent}) \leftarrow \max(P(\text{parent}), P(\text{child}))$ at inference.

### C. Avoiding the Rare-Class Threshold Overfitting Trap (`postprocess.py`)
- Optimizing independent thresholds $\tau_c^*$ on OOF for a label with only 8 positive examples will memorize those 8 OOF scores and collapse on the test set.
- Always use **Empirical-Bayes Support Shrinkage** (`optimize_regularized_thresholds` in `postprocess.py`):
  $$\hat{\tau}_c = \frac{n_c}{n_c + \lambda}\tau_c^* + \frac{\lambda}{n_c + \lambda}\tau_{\text{global}}$$
  so head classes ($n_c \gg \lambda$) use their class-specific optimal threshold while rare tail classes ($n_c \ll \lambda$) safely fall back toward $\tau_{\text{global}}$.
