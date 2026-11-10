# `00_core/indonesian_nlp/` — Fine-Grained Indonesian & Nusantara NLP Toolkit

Standalone, zero-network Indonesian linguistic normalizers, adversarial de-obfuscators, phonological syllabifiers, regional dialect scorers, and structured domain extractors tailored for Gemastik Data Mining & Indonesian NLP competitions.

## Atomic Scripts

| Script | Key Classes & Functions | Purpose |
| :--- | :--- | :--- |
| `slang_and_alay_normalizer.py` | `IndonesianTextNormalizer`, `IndonesianMorphologicalAnalyzer`, `expand_indonesian_reduplication`, `convert_old_indonesian_spelling`, `normalize_char_elongation` | Colloquial/singkatan dictionary (`yg`, `gak`/`ora`/`mboten`/`teu`, `bgt`, etc.), numeric/quote reduplication expander (`kata2` $\rightarrow$ `kata-kata`, `anak"` $\rightarrow$ `anak-anak`), Old Spelling (*Van Ophuijsen/Soewandi*: `oe` $\rightarrow$ `u`, `dj` $\rightarrow$ `j`, `tj` $\rightarrow$ `c`), elongation collapser, and rule-based Indonesian prefix/suffix analyzer |
| `deobfuscator_judol_toxic.py` | `JudolToxicDeobfuscator`, `strip_zero_width_chars`, `normalize_unicode_homoglyphs`, `collapse_spaced_and_punctuated_sequences`, `decode_indonesian_leetspeak` | De-obfuscates *Judi Online* (`g4c0r`, `m4xw1n`, `s.l.o.t`, `j*d*l`), *Pinjol* (`p1nj0l`), and censored toxic tokens; strips zero-width chars, normalizes Cyrillic/Greek homoglyphs, and extracts GBDT obfuscation features |
| `prosody_pantun_puisi.py` | `IndonesianProsodyAnalyzer`, `syllabify_indonesian_word`, `classify_stanza_rhyme_scheme` | Deterministic Indonesian phonological syllabifier (*Pemenggalan Suku Kata* protecting `ng, ny, sy, kh` & diphthongs `ai, au, oi, ei`), 8–12 Pantun syllable rule checker, final/internal rhyme analyzer (`a-b-a-b` Pantun vs. `a-a-a-a` Syair vs. `a-a` Gurindam), and *sampiran* vs. *isi* similarity |
| `nusantara_dialect_detector.py` | `NusantaraDialectDetector`, `DIALECT_MARKERS` | Lexical & phonotactic scorer across 10 Nusantara varieties (Formal Indonesian, Colloquial Jakartan, Javanese *Ngoko/Krama*, Sundanese, Minangkabau, Balinese, Bugis/Makassar, Banjarese, Madurese, Regional Malay) + Code-Switching Entropy |
| `indonesian_domain_extractors.py` | `IndonesianDomainFeatureExtractor`, `extract_indonesian_addresses`, `parse_rupiah_amounts`, `extract_indonesian_dates_and_times`, `extract_indonesian_legal_markers` | High-precision extractors for Indonesian Administrative Hierarchy (`Provinsi`..`RT/RW`), Currency (`Rp 15.000,-`, `1,5 juta`, `2,4 miliar` $\rightarrow$ float), Indonesian Dates/Timezones (`WIB`/`WITA`/`WIT`), and Legal *Putusan MA* sections (`Menimbang`, `Mengingat`, `Mengadili`, `Pasal`, `UU`) |

## Quick Verification
```bash
python3 -m py_compile 00_core/indonesian_nlp/*.py
for s in 00_core/indonesian_nlp/*.py; do python3 "$s" --smoke-test; done
```
