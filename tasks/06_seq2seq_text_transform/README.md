# Task 06: Sequence-to-Sequence & Text Transformation

## 1. Formulation & Common Competition Metrics
- **Input:** Source sequence (noisy text, document, dialogue, structured table/triples, or vernacular text).
- **Output:** Target sequence (summary, normalized/corrected text, translation, extracted structured string).
- **Metrics:** ROUGE-1/2/L, SacreBLEU, ChrF / ChrF++, Exact Match, Levenshtein / Word Error Rate (WER), BERTScore.

## 2. Validation Strategy & Pitfalls
- **Group by Source Document/Entity:** Use `GroupKFold` on source entity/template to avoid target memorization.
- **Decoding Mismatch:** Always tune generation hyperparameters (`num_beams`, `length_penalty`, `max_new_tokens`, `no_repeat_ngram_size`, `min_length`) directly against the competition metric on OOF validation.

## 3. Escalation Playbook
- **Tier 1 (Heuristic / Copy / Retrieval Baseline):** Often in text normalization or grammatical error correction (GEC), copying the input unchanged or dictionary-mapping common tokens achieves a high initial baseline—measure this first!
- **Tier 2 (Encoder-Decoder Fine-Tuning):** `google/mt5-base`, `facebook/mbart-large-50`, `ByT5` (for character-level robustness/typos), or `IndoBART` / `IndoT5` with `Seq2SeqTrainer` / custom PyTorch loop + Label Smoothing.
- **Tier 3 (Reranking & Ensembling):**
  - Generate $N$-best beam candidates across diverse models $\rightarrow$ Minimum Bayes Risk (MBR) decoding (select candidate maximizing average BLEU/ROUGE/ChrF against all other hypotheses) or Cross-Encoder reranking.

## 4. Directory Contents
- `baselines/` — Standalone Seq2Seq OOF trainer and beam-search evaluator.
- `modules/` — Minimum Bayes Risk (MBR) hypothesis selector, edit-distance post-processors, dynamic length control.
- `winners/SUMMARY.md` — Curated Kaggle & competition winning solutions.
