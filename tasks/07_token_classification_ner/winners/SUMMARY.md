# Winning Solutions & Curated References — Token Classification & NER

## 1. Benchmark Competition Hall of Fame

| Competition | Metric | 1st Place Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **Kaggle PII Data Detection** | Micro $F_5$ | Sliding-Window `DeBERTa-v3-large` (`max_len=1024, stride=256`) + Residual BiLSTM Head | Per-class probability thresholding ($\tau_c \in [0.15, 0.40]$ because $F_5$ penalizes FN $25\times$ more than FP), BIO transition repair, regex post-filters for emails/URLs/IDs. |
| **Kaggle NBME – Score Clinical Patient Notes** | Character-level Micro $F_1$ | Multi-Tokenizer Ensemble (`DeBERTa-v3-large` + `RoBERTa-large` + `BioLinkBERT`) projected to Character Probabilities | `token_probs_to_char_probs` alignment trick enabling seamless blending across incompatible subword tokenizers + multi-round pseudo-labeling on unlabeled clinical notes. |
| **Kaggle Feedback Prize – Evaluating Student Writing** | Span-Overlap Macro $F_1$ | Stage-1 Token Classifier (`DeBERTa-v3` + BiLSTM) $\rightarrow$ Stage-2 GBDT Span Reranker | Extracts candidate spans at multiple thresholds, pools `(start_prob, end_prob, mean_prob, span_len, discourse_position)` features, and trains a LightGBM span scorer + Span NMS. |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Kaggle PII Data Detection (1st Place — Micro $F_5$)
- **Problem Formulation:** Detect 12 categories of Personally Identifiable Information (`NAME_STUDENT`, `EMAIL`, `PHONE_NUM`, `STREET_ADDRESS`, `URL_PERSONAL`, etc.) in long student essays. Scored by **Micro $F_5$**:
  $$F_5 = \frac{(1 + 5^2) \cdot P \cdot R}{5^2 \cdot P + R} = \frac{26 \cdot \text{TP}}{26 \cdot \text{TP} + 25 \cdot \text{FN} + \text{FP}}$$
  A single missed positive (FN) costs **$25\times$** as much as a false alarm (FP)!
- **Key Winning Techniques:**
  1. **Sliding-Window Context Preservation (`max_length=1024, stride=256`):**
     Essays frequently exceed 512 tokens. Overlapping windows ensure tokens near window boundaries are scored with full left+right context; overlapping predictions are averaged via `offset_mapping`.
  2. **Residual BiLSTM Layer Before Classifier:**
     Adding a single bidirectional LSTM layer with residual connection (`x = LayerNorm(x + BiLSTM(x))`) on top of the final Transformer hidden states before the linear projection gave a consistent **+0.006 $F_5$** gain over a bare linear head.
  3. **Per-Class Probability Thresholding (The #1 $F_5$ Lever):**
     Default `argmax` (equivalent to $\tau \approx 0.50$) severely under-predicts rare PII classes under $F_5$. Winners tuned separate thresholds per class on OOF (often $\tau_{\text{O}} > 0.85$ or $\tau_{\text{entity}} \approx 0.20\text{--}0.35$), boosting CV and Private LB by **+0.015 to +0.025 $F_5$**.
  4. **BIO Sequence Repair & Whitespace Normalization:**
     Promotes orphan `I-NAME_STUDENT` tags to `B-NAME_STUDENT` and merges consecutive name spans separated by single spaces or newlines.

---

### B. NBME – Score Clinical Patient Notes (1st Place — Character Span $F_1$)
- **Problem Formulation:** Extract exact character ranges `[start_char, end_char]` matching clinical concepts in doctor notes.
- **The Gold-Standard Multi-Tokenizer Ensembling Trick (`token_probs_to_char_probs`):**
  - `DeBERTa-v3` uses SentencePiece (`▁`), `RoBERTa` uses byte-level BPE (`Ġ`), and `Electra` / `IndoBERT` use WordPiece (`##`). Their token arrays have **different lengths and different subword boundaries** for the exact same sentence!
  - **Solution:** For every model $m$, use `return_offsets_mapping=True` to project its token probabilities `(n_tokens, C)` onto a character-level array `char_probs_m` of shape `(len(text), C)`:
    ```python
    for tok_idx, (start_char, end_char) in enumerate(offsets):
        char_probs[start_char:end_char] += token_probs[tok_idx]
    ```
  - Once every model is represented in character space `(len(text), C)`, any number of heterogeneous architectures can be blended via weighted arithmetic or rank averaging, and decoded into `(start_char, end_char, label)` spans while stripping leading/trailing whitespace.

---

### C. Feedback Prize (1st & 2nd Place — Two-Stage Span Reranking + NMS)
- **Problem Formulation:** Segment long essays into discourse elements (`Lead`, `Position`, `Claim`, `Evidence`, `Counterclaim`, `Rebuttal`, `Concluding Statement`).
- **Two-Stage Span Extraction Pipeline:**
  1. **Stage 1 (Token + Boundary Probabilities):** Train sliding-window Transformer to output 3 probabilities per class at every token/character: `P(Begin)`, `P(Inside)`, `P(End)`.
  2. **Stage 2 (Candidate Span Pool + GBDT Scorer + Span NMS):**
     - Generate over-complete candidate spans `(s, e, class)` using low boundary thresholds.
     - Extract span-level features: `mean(P_inside[s:e])`, `P_begin[s]`, `P_end[e-1]`, `span_length_ratio`, `relative_position_in_doc`, and neighbor span probabilities.
     - Train an OOF LightGBM binary/regression model to predict exact span IoU with ground truth, followed by **1D Span Non-Maximum Suppression (`span_nms`)**.

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Entity density/length diagnostics, leading/trailing whitespace boundary audit, and `verify_round_trip_offset_mapping` (verifies 100% round-trip F1 ceiling before training).
- [`modules/linear_chain_crf.py`](../modules/linear_chain_crf.py): Pure-PyTorch `LinearChainCRF` (forward log-partition + Viterbi decode + BIO transition constraints).
- [`modules/char_span_aggregator.py`](../modules/char_span_aggregator.py): `token_probs_to_char_probs` and `char_probs_to_spans` for multi-tokenizer character-level ensembling.
- [`modules/span_metrics.py`](../modules/span_metrics.py): Strict entity-level `evaluate_entity_spans` (Micro-F1, Macro-F1, $F_\beta$).
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Multi-word `DictionaryTrie` gazetteer + contextual orthographic feature extractor + OOF linear classifier with **Viterbi BIO Transition Decoding**.
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Sliding-Window Character-Aligned Transformer + Residual BiLSTM NER OOF pipeline.
- [`postprocess.py`](../postprocess.py): `repair_bio_sequence`, `merge_adjacent_same_label_spans`, `span_nms`, and `optimize_per_class_thresholds_and_min_len` for $F_\beta$.
