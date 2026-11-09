# Task 07: Token Classification & Named Entity Recognition (NER / Span Extraction)

## 1. Formulation & Common Competition Metrics
- **Input:** Raw text or tokenized word sequence.
- **Output:** BIO/BILOU token tags or character/word-level entity spans `(start, end, entity_type)`.
- **Metrics:** Strict Entity-level Micro/Macro $F_1$ (`seqeval`), Partial/Overlap Span $F_1$, Token/Span-level $F_\beta$ (e.g., PII Detection $F_5$).

## 2. Validation Strategy & Leakage Traps
- **Entity / Document Grouping:** Use `MultilabelStratifiedKFold` on document-level entity type counts or `GroupKFold` on document author/source.
- **Whitespace & Offset Alignment Trap:** Leading/trailing spaces, newline characters (`\n`), and subword tokenization (`return_offsets_mapping=True`) are the #1 source of silent $F_1$ drops. Always run `verify_round_trip_offset_mapping` in [`eda_and_data.py`](eda_and_data.py) to verify 100% round-trip span reconstruction before training!

## 3. Escalation Playbook
- **Tier 1 (CPU Fast):** Multi-word `DictionaryTrie` entity gazetteer + contextual orthographic features ($w_{-2}..w_{+2}$) + OOF linear classifier with **Viterbi BIO Transition Decoding** (`tier1_fast_cpu.py`).
- **Tier 2 (Sliding-Window Character-Aligned Transformer + BiLSTM):**
  - Handles arbitrarily long documents via overlapping windows (`max_length=512, stride=128, return_offsets_mapping=True`).
  - Projects overlapping window token probabilities into **character-level probability arrays `(doc_char_len, num_classes)`** (`tier2_strong_dl.py` & `modules/ner_heads_and_alignment.py`), allowing seamless ensembling across incompatible tokenizers!
- **Tier 3 (CRF Head & Post-Processing):**
  - Pure-PyTorch `LinearChainCRF` (`modules/ner_heads_and_alignment.py`).
  - Per-class probability threshold & minimum span length coordinate optimizer for $F_\beta$ + BIO transition repair + adjacent span merging + 1D Span NMS (`postprocess.py`).

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Entity type counts, span length distributions, whitespace boundary audit, and Round-Trip `offset_mapping` Verification Function.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Multi-word `DictionaryTrie` gazetteer + contextual orthographic features + Viterbi BIO decoder.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Sliding-Window Character-Aligned Transformer + Residual BiLSTM NER OOF pipeline.
- [`modules/ner_heads_and_alignment.py`](modules/ner_heads_and_alignment.py) — Pure-PyTorch `LinearChainCRF`, `token_probs_to_char_probs`, `char_probs_to_spans`, and strict entity-level evaluator.
- [`postprocess.py`](postprocess.py) — BIO sequence repair, adjacent span merger, Per-Class Probability Threshold & Minimum Span Length Optimizer ($F_5$), and 1D Span NMS.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Detailed breakdown of winning solutions (Kaggle PII Data Detection 1st, NBME 1st, Feedback Prize 1st/2nd).
