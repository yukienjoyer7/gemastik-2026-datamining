# Winning Solutions & Curated References — Task 01: Text Classification

## 1. Benchmark & Competition Winning Solutions Table

| Competition / Benchmark | Task & Metric | Winning Architecture & Backbone | Key Winning Techniques |
| :--- | :--- | :--- | :--- |
| **Kaggle Feedback Prize — English Language Learning (1st–3rd Place)** | Multi-output / Multiclass Essay Scoring (MCRMSE / Macro-F1) | `microsoft/deberta-v3-large` + `deberta-v2-xlarge` | **MeanPooling** + **WeightedLayerPooling** (last 4 layers), **Layer-wise Learning Rate Decay (LLRD, 0.9)**, **AWP** starting from Epoch 2, Multi-Sample Dropout (5x), and 2nd-stage **LightGBM / Ridge** stacking on OOF embeddings + readability/lexical statistics. |
| **Kaggle Feedback Prize — Evaluating Student Writing** | Long-Document Discourse Classification (F1) | `microsoft/deberta-large` + `google/bigbird-roberta-large` | Extended context length (`max_len=1024..1536`) or **Head+Tail truncation** (`128 + 382`) when GPU-constrained; **Dynamic Padding** for 3x faster training; 2 rounds of high-confidence **Pseudo-Labeling** on external/unlabeled essays. |
| **Kaggle CommonLit Readability & Automated Essay Scoring 2.0 (1st Place)** | Ordinal / Continuous Text Rating (RMSE / QWK) | `deberta-v3-large` + Feature-Engineered GBDT Fusion | Concatenating OOF DeBERTa pooled representations with **Word `(1, 3)` + Char-wb `(3, 5)` TF-IDF Ridge OOF** and surface lexical stats (paragraph length, word length, punctuation density, OOV ratio) into LightGBM + CatBoost, followed by **Nelder-Mead** threshold/multiplier post-processing. |
| **Kaggle Jigsaw Toxic Comment & Unintended Bias (1st Place)** | Binary / Subgroup-Fair Toxicity Classification (ROC-AUC / LogLoss) | Ensemble of `DeBERTa-v3`, `RoBERTa-large`, BiGRU-Attention, and Char+Word TF-IDF Linear Models | **Subgroup-aware sample weighting** in BCE loss to prevent false-positive bias on identity terms; **Temperature Scaling** before blending heterogeneous model families (TF-IDF LogisticRegression + Neural Models). |
| **IndoNLU Benchmark (`EmoT`, `SmSA`, `HoASA`)** | Indonesian Emotion, Sentiment & Aspect Classification (Macro-F1) | `indobenchmark/indobert-base-p1`, `indobenchmark/indobert-large-p1`, `indolem/indobertweet-base-uncased` | Slang/colloquial normalization dictionary prior to Word+Char-wb TF-IDF; combining `indobertweet` (strong on Twitter/social media slang) with `indobert-base-p1` (strong on formal/news Indonesian) via Nelder-Mead class multiplier optimization. |

---

## 2. Distilled Architectural Playbook

### A. Why Char-wb `(3, 5)` + Word `(1, 3)` TF-IDF Remains Essential
- Even when `DeBERTa-v3` or `IndoBERT` achieves a higher standalone score, adding **Tier 1 TF-IDF + Lexical OOF probabilities** (`baselines/tier1_fast_cpu.py`) with weight `0.10–0.25` consistently improves ensemble CV by **+0.003 to +0.008 Macro-F1**.
- Character word-boundary n-grams (`analyzer="char_wb"`, `ngram_range=(3, 5)`) capture morphological affixes (`me-`, `di-`, `-kan`, `-nya` in Indonesian), misspellings, and obfuscated profanity that subword tokenizers split inconsistently.

### B. Pooling Selection Guide (`modules/pooling_and_regularization.py`)
1. **`MeanPooling` (Default #1 choice):** Outperforms `[CLS]` pooling on almost every modern DeBERTa-v3 / RoBERTa classification task because `[CLS]` is not pretrained with Next-Sentence Prediction in DeBERTa-v3/RoBERTa.
2. **`MeanMaxPooling`:** Best when a single trigger phrase (e.g., hate speech, sentiment spike, key legal clause) determines the document label inside a long neutral text.
3. **`WeightedLayerPooling`:** Best when fine-tuning 24-layer `-large` models where syntactic cues live in layers 16–20 and semantic cues live in layers 21–24.

### C. Stabilization Recipes for Small/Noisy Competition Datasets (`baselines/tier2_strong_dl.py`)
- **Layer-wise Learning Rate Decay (`llrd_factor=0.9`):** Protects lower lexical/syntactic layers from catastrophic forgetting while letting top task-specific layers adapt rapidly.
- **Multi-Sample Dropout (`num_dropouts=5`, `p=0.2`):** Computes 5 parallel dropout masks before the final `nn.Linear` layer and averages the logits—acts as an internal ensemble with near-zero compute overhead.
- **Adversarial Weight Perturbation (`AWP`):** Enable starting at epoch 1 or 2 (`adv_lr=1e-2, adv_eps=1e-2`) once gradients stabilize. Perturbs weights toward worst-case loss neighborhoods to find flatter minima.
- **Post-Processing (`postprocess.py`):**
  - If the evaluation metric is **LogLoss / Brier Score**, fit `TemperatureScaler` on OOF logits.
  - If the evaluation metric is **Macro-F1**, run `optimize_class_multipliers` (coordinate search + Nelder-Mead) inside cross-validation folds to boost recall on minority classes.
