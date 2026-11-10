# Task 09: Aspect-Based Sentiment Analysis (ABSA) & Triplet Extraction

## 1. Formulation & Common Competition Metrics
- **Input:** Review or social media text $x$ (e.g., hotel, restaurant, automotive, e-commerce review).
- **Output Formulations:**
  1. **Aspect-Category Sentiment Analysis (ACSA — IndoNLU `HoASA` / `CASA` style):** For each of $A$ predefined aspect categories (e.g., `kebersihan`, `pelayanan`, `harga`, `lokasi` or `mesin`, `bahan_bakar`, `interior`), predict a 4-way label $y_{i,a} \in \{\text{None (0)}, \text{Negative (1)}, \text{Neutral (2)}, \text{Positive (3)}\}$.
  2. **Aspect-Term Sentiment Triplet Extraction (ASTE — SemEval style):** Extract explicit `(aspect_span, opinion_span, polarity)` triplets from raw text.
- **Metrics:** Macro-$F_1$ averaged across all $A$ aspects (or across active non-`None` aspect polarities), Micro-$F_1$, and exact triplet match $F_1$.

## 2. Validation Strategy & Leakage Traps
- **Joint `(Aspect, Polarity)` Multilabel Stratification:** Standard single-label stratification fails on multi-aspect data because each aspect has its own rare polarity classes (especially `Neutral`). Use `create_absa_multilabel_folds` in [`eda_and_data.py`](eda_and_data.py) to stratify across one-hot `(aspect, polarity)` indicators simultaneously.
- **Multi-Aspect Polarity Contrast Trap:** A global document-level classifier predicts the same sentiment for every aspect mentioned in a mixed-sentiment review (e.g., *"Makanannya enak banget tapi pelayanannya lambat dan mahal"* $\rightarrow$ `makanan=Positive`, `pelayanan=Negative`, `harga=Negative`). Always isolate **aspect-specific context windows** or **aspect-query attention**.

## 3. Escalation Playbook
- **Tier 1 (CPU Fast — Aspect-Keyword Context Window TF-IDF + Per-Aspect Linear/GBDT):**
  - Extracts global Word + Char TF-IDF alongside **Aspect-Keyword Local Context Window TF-IDF** ($\pm k$ words around seed keywords for aspect $a$), training an OOF classifier per aspect (`baselines/tier1_fast_cpu.py`).
- **Tier 2 (Auxiliary-Sentence Cross-Encoder & Single-Pass Multi-Aspect Attention Transformer):**
  - *Formulation A (Auxiliary-Sentence NLI/QA Prompting):* Converts each `(review, aspect_a)` into `"[CLS] Aspek: {aspect_name} ({aspect_keywords}) [SEP] {review_text} [SEP]"` (`modules/auxiliary_sentence_prompter.py` & `baselines/tier2_strong_dl.py`).
  - *Formulation B (Single-Pass Multi-Aspect Cross-Attention Head):* Encodes the review once and uses $A$ learnable aspect query embeddings to cross-attend over token hidden states, outputting `(batch, A, C)` logits (`modules/multi_aspect_attention_head.py` & `baselines/tier2_strong_dl.py`).
  - *Formulation C (Span-ASTE Triplet Extraction):* Decodes `(aspect_span, opinion_span, polarity)` via BIO tagging + span-pair scorer (`modules/absa_triplet_decoder.py`).
- **Tier 3 (Post-Processing & Hierarchical Calibration):**
  - Per-aspect probability temperature/prior calibration + hierarchical `P(aspect_present)` vs. `P(polarity | present)` threshold optimization (`postprocess.py`).

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Audits aspect presence rates, polarity distributions, mixed-polarity co-occurrence within single reviews, and builds leak-free multi-aspect stratified folds.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Global Word/Char TF-IDF + Aspect-Keyword Context Window ($\pm k$ tokens) TF-IDF + OOF per-aspect `LogisticRegression`/`LightGBM` exporting `oof_tier1_absa.npy`, `test_tier1_absa.npy`, `sub_tier1_absa.csv`.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Auxiliary-Sentence Cross-Encoder & Single-Pass Multi-Aspect Attention Transformer OOF pipeline.
- [`modules/auxiliary_sentence_prompter.py`](modules/auxiliary_sentence_prompter.py) — Reshapes wide multi-aspect tables (`id, text, aspect_1..aspect_A`) into long auxiliary-sentence pair prompts and restores `(N, A, C)` probability tensors.
- [`modules/multi_aspect_attention_head.py`](modules/multi_aspect_attention_head.py) — PyTorch `MultiAspectCrossAttentionHead` learning $A$ aspect query vectors attending over token hidden states.
- [`modules/absa_triplet_decoder.py`](modules/absa_triplet_decoder.py) — Extracts `(aspect_span, opinion_span, polarity)` triplets from BIO tags and span-pair interaction logits.
- [`postprocess.py`](postprocess.py) — Per-aspect class prior multiplier search + hierarchical presence/polarity consistency calibration maximizing Macro-$F_1$.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Curated references on IndoNLU `HoASA` & `CASA`, SemEval 1st-place Auxiliary-Sentence BERT/DeBERTa, and Span-ASTE triplet extraction.
