# Winning Solutions & Curated References — Aspect-Based Sentiment Analysis (ABSA)

## 1. Benchmark Competition & Shared-Task Hall of Fame

| Benchmark / Shared Task | Metric | 1st Place Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **IndoNLU `HoASA` (Hotel Aspect-Based Sentiment Analysis)** | Macro $F_1$ across 10 hotel aspects | `IndoBERT-Large` Multi-Aspect Attention + Auxiliary-Sentence Cross-Encoder Ensemble | Contrastive clause segmentation (`tapi`, `namun`), per-aspect seed keyword context prompts, hierarchical `None` vs. `{Neg, Neu, Pos}` threshold tuning. |
| **IndoNLU `CASA` (Car Aspect-Based Sentiment Analysis)** | Macro $F_1$ across 6 automotive aspects | `IndoBERT` + `XLM-RoBERTa-Large` Auxiliary-Sentence NLI Pair Classification | Reshaping $(N, A)$ into $N \times A$ NLI pairs `"[CLS] Aspek: {aspect} ({keywords}) [SEP] {review} [SEP]"`, multi-sample dropout, OOF class prior calibration. |
| **SemEval ABSA (Restaurant & Laptop ACSA / ACD)** | Macro $F_1$ / Micro $F_1$ | BERT/DeBERTa-v3 Auxiliary Sentence QA (`BERT-pair-QA-M` / `BERT-pair-NLI-M`) | Formulating ABSA as natural-language question answering / entailment (Sun et al., NAACL) yields $+3.5\text{–}5.0$ Macro-$F_1$ over single-sentence classification heads. |
| **SemEval Span-ASTE (Aspect Sentiment Triplet Extraction)** | Strict Triplet $F_1$ `(aspect, opinion, polarity)` | Dual-Channel BIO Span Extractor + Span-Pair Biaffine Polarity Scorer | Distance-aware span-pair pruning (`|a_pos - o_pos| <= 12`), joint boundary confidence $\times$ polarity confidence scoring. |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Why Auxiliary-Sentence Cross-Encoders Dominate ACSA (Sun et al. / SemEval & IndoNLU Winners)
1. **The Bottleneck of Standard `[CLS]` Pooling:**
   - In a standard multi-head classifier on top of `[CLS]`, a single pooled vector must simultaneously encode `makanan=Positive`, `pelayanan=Negative`, and `harga=Neutral` from a sentence like *"Rasa makanannya juara banget, sayang pelayanannya lelet walaupun harganya standar"*.
2. **The Auxiliary-Sentence Transformation (`modules/auxiliary_sentence_prompter.py`):**
   - Instead of classifying $A$ aspects from one static `[CLS]` vector, construct an explicit aspect query prompt:
     $$\texttt{[CLS] Aspek: pelayanan (staf, resepsionis, ramah, lambat, sopan) [SEP] } \text{Review} \texttt{ [SEP]}$$
   - Now every self-attention layer in the Transformer attends directly from the aspect keywords (`staf, ramah, lambat`) to the corresponding opinion tokens (`lelet`) inside the review!
3. **Single-Pass Multi-Aspect Cross-Attention (`modules/multi_aspect_attention_head.py`):**
   - When $A$ is large ($A \ge 10$ as in `HoASA`), running $A$ separate Cross-Encoder forward passes per review costs $A\times$ compute.
   - Pairing a single-pass `MultiAspectCrossAttentionHead` (which learns $A$ dedicated query vectors attending over the token hidden states in 1 pass) with an Auxiliary-Sentence Cross-Encoder provides both high speed and strong architectural diversity for ensembling (`baselines/tier2_strong_dl.py`).

---

### B. Fast CPU Feature Engineering for ABSA (`baselines/tier1_fast_cpu.py`)
1. **Contrastive Clause Splitting & Local Keyword Windows:**
   - Indonesian reviews heavily use contrastive conjunctions (`tapi`, `tetapi`, `namun`, `sedangkan`, `padahal`, `sayangnya`).
   - Splitting reviews on contrastive markers and extracting $\pm k$ token windows around aspect seed keywords (`extract_aspect_context_window`) gives linear/GBDT models clean, aspect-isolated TF-IDF features.
   - Weighting local aspect-window TF-IDF $2.5\times$ higher than global document TF-IDF prevents strong sentiment in one clause from leaking into unrelated aspects.

---

### C. Hierarchical Presence vs. Polarity Calibration (`postprocess.py`)
1. **Decoupling Aspect Detection from Sentiment Polarity:**
   - In 4-class ACSA (`0=None, 1=Negative, 2=Neutral, 3=Positive`), class `0` (`None`) often accounts for $60\text{–}85\%$ of all `(review, aspect)` cells, while class `2` (`Neutral`) is $<5\%$.
   - Standard `argmax` under-predicts `Neutral` and borderline active aspects.
   - Decomposing the decision into:
     1. **Aspect Presence Gate:** $P(\text{present}) = 1 - P(\text{None}) \ge \tau_a$
     2. **Active Polarity Selection:** $\hat{y}_{i,a} = \arg\max_{c \in \{1,2,3\}} w_{a,c} P(y_{i,a} = c)$
     consistently boosts Macro-$F_1$ by $+1.5\text{–}3.0$ points on OOF validation.

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Audits per-aspect presence, polarity balance, mixed-polarity contrast rate, and builds leak-free multi-aspect stratified folds.
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Global Word/Char TF-IDF + Aspect-Keyword Context Window ($\pm k$ tokens & contrastive clause splitting) TF-IDF + OOF per-aspect classifier.
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Auxiliary-Sentence Cross-Encoder + Single-Pass Multi-Aspect Cross-Attention Transformer OOF ensemble.
- [`modules/auxiliary_sentence_prompter.py`](../modules/auxiliary_sentence_prompter.py): Wide-to-long auxiliary-sentence pair converter and `(N, A, C)` tensor reshaper.
- [`modules/multi_aspect_attention_head.py`](../modules/multi_aspect_attention_head.py): PyTorch `MultiAspectCrossAttentionHead` with learnable aspect query embeddings.
- [`modules/absa_triplet_decoder.py`](../modules/absa_triplet_decoder.py): Span-ASTE BIO + span-pair `(aspect_span, opinion_span, polarity)` triplet extractor and evaluator.
- [`postprocess.py`](../postprocess.py): Hierarchical presence threshold + per-aspect polarity multiplier coordinate optimizer.
