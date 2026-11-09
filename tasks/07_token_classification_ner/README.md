# Task 07: Token Classification & Named Entity Recognition (NER / Span Extraction)

## 1. Formulation & Common Competition Metrics
- **Input:** Raw text or tokenized word sequence.
- **Output:** BIO/BILOU token tags or character/word-level entity spans `(start, end, entity_type)`.
- **Metrics:** Strict Entity-level Micro/Macro F1 (`seqeval`), Partial/Overlap Span F1, Token-level F\beta (e.g., PII Detection $F_5$).

## 2. Validation Strategy & Leakage Traps
- **Entity / Document Grouping:** Use `MultilabelStratifiedKFold` on document-level entity type counts (or `GroupKFold` on document author/source).
- **Whitespace & Offset Alignment Trap:** Leading/trailing spaces, newline characters (`\n`), and subword tokenization (`return_offsets_mapping=True`) are the #1 source of silent F1 drops. Always verify round-trip character-span reconstruction on validation before training!

## 3. Escalation Playbook
- **Tier 1 (CPU Fast):** Dictionary/Regex gazetteer lookup + `sklearn-crfsuite` (CRF with word shape, prefix/suffix, POS/context window features).
- **Tier 2 (GPU Sliding-Window Transformer):** DeBERTa-v3 / XLM-R / IndoBERT with `return_offsets_mapping=True`, sliding window (`max_length=512, stride=128`), averaging overlapping window token probabilities.
- **Tier 3 (Architecture & Post-Processing):**
  - Replace linear head with **BiLSTM / GRU + Linear** or **CRF / Span-Boundary pointer head**.
  - Class-specific probability thresholds (especially when optimizing $F_\beta$ like $F_5$ or rare entity classes) + BIO transition repair rules + regex post-filters.

## 4. Directory Contents
- `baselines/` — Standalone CRF baseline and DeBERTa sliding-window character-offset NER OOF script.
- `modules/` — Subword-to-char-span aligner, BIO repair & span NMS post-processor, CRF layer.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions (e.g., Feedback Prize, NBME, PII Data Detection).
