# Winning Solutions & Curated References — Seq2Seq & Text Transformation

## 1. Benchmark Competition Hall of Fame

| Competition / Benchmark | Task & Metric | Winning Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **Google Text Normalization Challenge** | Spoken-to-Written Normalization (Exact Match Accuracy) | Contextual N-gram Lookup Table + Gradient Boosted Token Classifier + Seq2Seq Fallback | 98.5%+ of tokens are verbatim copies or deterministic semiotic class expansions; neural Seq2Seq is invoked *only* on ambiguous OOV spans. |
| **BEA / CoNLL Grammatical Error Correction (GEC) & Colloquial Normalization** | GEC / Slang Normalization ($F_{0.5}$, ChrF, Levenshtein) | `google/byt5-base` + Edit-Tagging (`GECToR`) + Minimum Bayes Risk (MBR) Ensembling | Byte-level `ByT5` eliminates subword OOV corruption on typos/slang; MBR decoding combines edit-based and autoregressive models. |
| **Abstractive Summarization & Dialogue State / Text-to-Text Benchmarks** | ROUGE-1/2/L, BLEU, BERTScore | `mT5-large` / `mBART-large-50` / `IndoBART` + Diverse Beam Search + MBR Reranking | Dynamic target length bounds (`min_length`, `max_new_tokens`, `length_penalty`) calibrated to OOF length ratio quantiles + trailing n-gram loop stripping. |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Copy-Ratio Gate: The First Diagnostic to Run
Before training any Encoder-Decoder model, always measure the **Verbatim Copy Ratio** and **Normalized Levenshtein Similarity** between `source_text` and `target_text` (implemented in [`eda_and_data.py`](../eda_and_data.py)):
1. **High-Copy Regime ($\text{Token Copy Ratio} \ge 0.70$, e.g., Text Normalization, GEC, Slang-to-Formal, OCR Post-Correction):**
   - Pure free-form generation risks hallucinating or dropping rare named entities and numbers.
   - **Tier-1 Winner Pattern:** Contextual `(prev_token, token, next_token)` replacement lookup table built strictly inside each CV fold + identity fallback (`tier1_fast_cpu.py`).
   - **Tier-2 Winner Pattern:** Character/Byte-level Seq2Seq (`google/byt5-small` / `byt5-base`) with tight length bounds (`0.8 * len(src)` to `1.4 * len(src)`). Because `ByT5` operates directly on UTF-8 bytes, it never suffers from subword tokenizer fragmentation on typos, morphological affixes, or Indonesian slang abbreviations (`yg`, `blm`, `sdh`, `kmrn`).
2. **Low-Copy / Abstractive Regime ($\text{Token Copy Ratio} < 0.70$, e.g., Summarization, Translation, Paraphrasing, Data-to-Text):**
   - **Tier-1 Winner Pattern:** TF-IDF + TextRank sentence graph centrality with Lead-$K$ positional bias (`tier1_fast_cpu.py`).
   - **Tier-2 Winner Pattern:** Multilingual Encoder-Decoder (`google/mt5-base`, `facebook/mbart-large-50`, `IndoT5`) trained with Label Smoothing ($\epsilon = 0.10$) and decoded with $N$-best Beam Search (`num_beams=4..8`).

---

### B. Minimum Bayes Risk (MBR) Decoding & Multi-Model Ensembling
Token-level logit averaging across different Seq2Seq architectures (e.g., `ByT5` + `mT5` + `mBART`) is impossible when their tokenizers differ. Winning teams ensemble heterogeneous generative models at the **hypothesis level** via **Minimum Bayes Risk (MBR)**:
1. Collect a candidate pool $\mathcal{H} = \{h_1, h_2, \dots, h_M\}$ from $N$-best beam outputs across all folds and model families.
2. Given the exact competition evaluation metric $U(h_i, h_j)$ (`ChrF`, `ROUGE-L`, or `Normalized Levenshtein`), select:
   $$h^* = \arg\max_{h_i \in \mathcal{H}} \sum_{j=1}^{M} w_j \, U(h_i, h_j)$$
3. **Why MBR Works:** Beam search maximizes sequence likelihood (which favors short or overly generic strings), whereas MBR directly maximizes expected competition utility against the posterior distribution of hypotheses. Consistently yields **+0.5 to +1.8 ROUGE / ChrF / BLEU** over single-best beam output.

---

### C. Decoding & Post-Processing Guardrails
1. **Degenerate Repetition Loop Stripping (`strip_repeated_ngrams`):**
   - Autoregressive decoders occasionally get stuck repeating trailing 1–4 token n-grams (`"ke surabaya ke surabaya ke surabaya"`). Stripping consecutive repeated n-grams in [`postprocess.py`](../postprocess.py) cleanly repairs these failures.
2. **Length-Ratio Clamping:**
   - Compute 5th and 95th percentiles of `len(target) / len(source)` on OOF training data. Enforce these bounds during generation and post-processing, falling back to the cleaned source or extractive baseline if a beam collapses to an empty string.

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Source/target length ratio quantiles, Verbatim Copy Ratio, Levenshtein profile, and PyTorch `Seq2SeqDataset`.
- [`modules/mbr_decoding.py`](../modules/mbr_decoding.py): `mbr_select_best_hypothesis` (Minimum Bayes Risk decoding over $N$-best beam or multi-model hypotheses).
- [`modules/seq2seq_metrics.py`](../modules/seq2seq_metrics.py): `chrf_score`, `rouge_l_score`, `exact_match_score`, and `normalized_levenshtein`.
- [`modules/copy_and_edit_aligner.py`](../modules/copy_and_edit_aligner.py): Levenshtein edit-script aligner (`KEEP`, `DELETE`, `INSERT`, `REPLACE` operations).
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Contextual N-gram token replacement lookup model (Normalization/GEC regime) + TF-IDF TextRank sentence extractor (Summarization regime).
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Encoder-Decoder Seq2Seq OOF trainer with Label Smoothing, $N$-best beam search, and MBR reranking.
- [`postprocess.py`](../postprocess.py): Consecutive n-gram loop cleaner, length-ratio guardrails, and Multi-Model MBR Ensemble Combiner.
