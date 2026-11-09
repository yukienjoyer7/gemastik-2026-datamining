# Task 06: Sequence-to-Sequence & Text Transformation

## 1. Formulation & Common Competition Metrics
- **Input:** Source sequence (noisy/slang text, document, dialogue, structured triples, or unnormalized text).
- **Output:** Target sequence (normalized/corrected text, summary, translation, extracted structured string).
- **Metrics:** ROUGE-1/2/L, ChrF / ChrF++, Exact Match, Normalized Levenshtein / Word Error Rate (WER), SacreBLEU.

## 2. Validation Strategy & Pitfalls
- **Group by Source Document/Entity:** Use `GroupKFold` on source entity/template (`group_id`) via [`eda_and_data.py`](eda_and_data.py) to prevent target memorization.
- **Copy-Ratio Audit First:** Always inspect the **Verbatim Copy Ratio** and **Levenshtein Similarity** between source and target before picking an architecture.
- **Decoding Mismatch:** Tune generation hyperparameters (`num_beams`, `length_penalty`, `min_len_ratio`, `max_len_ratio`) and apply **Minimum Bayes Risk (MBR)** directly against the competition metric on OOF validation.

## 3. Escalation Playbook
- **Tier 1 (CPU Fast — Regime-Adaptive Baseline):**
  - *Normalization / Error Correction Regime:* Contextual bigram + unigram replacement lookup table learned from aligned training pairs + identity fallback (`tier1_fast_cpu.py`).
  - *Summarization / Extraction Regime:* TF-IDF + TextRank sentence graph centrality with Lead-$K$ positional bias (`tier1_fast_cpu.py`).
- **Tier 2 (Encoder-Decoder Fine-Tuning + N-Best Beam Search):**
  - `ByT5` / `mT5` / `mBART` / `IndoT5` with Label Smoothing ($\epsilon = 0.10$) and $N$-best beam hypothesis export (`tier2_strong_dl.py`).
- **Tier 3 (MBR Decoding & Post-Processing):**
  - Minimum Bayes Risk (`mbr_select_best_hypothesis` in `modules/seq2seq_decoding_and_mbr.py`) + consecutive n-gram loop cleaner and multi-model MBR ensemble (`postprocess.py`).

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Length ratio quantiles, Verbatim Copy Ratio & Edit Distance Profile, and PyTorch `Seq2SeqDataset`.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Contextual N-gram token replacement aligner + TF-IDF TextRank sentence selector.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Encoder-Decoder OOF trainer with Label Smoothing, $N$-best beam search, and MBR reranking.
- [`modules/seq2seq_decoding_and_mbr.py`](modules/seq2seq_decoding_and_mbr.py) — `mbr_select_best_hypothesis`, `chrf_score`, `rouge_l_score`, and `normalized_levenshtein`.
- [`postprocess.py`](postprocess.py) — Repeated n-gram loop stripper, whitespace/punctuation normalizer, length ratio guardrails, and Multi-Model MBR Ensemble Combiner.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Curated competition references on Text Normalization, GEC (`ByT5`), Summarization, and MBR Ensembling.
