# Winning Solutions & Curated References — Task 11: Semi-Structured Log, Markup, Payload & Code

## 1. Benchmark & Competition Winning Solutions Table

| Competition / Benchmark | Task & Metric | Winning Architecture & Backbone | Key Winning Techniques |
| :--- | :--- | :--- | :--- |
| **LogHub / HDFS, BGL & Thunderbird Benchmarks (ICSE / DSN Winners)** | System Log Anomaly Detection (F1 / Precision-Recall AUC) | **Drain3 Log Template Miner** + TF-IDF on Template Sequences + **LightGBM / CatBoost** | Masking dynamic variables (`<IP>`, `<BLK_ID>`, `<HEX>`, `<NUM>`) via Drain parse trees; computing template frequency vectors, inter-arrival time deltas, and unseen-template flags; GBDT consistently outperforms heavy LSTMs/Transformers on tabularized template counts while running 50x faster. |
| **Kaggle Google AI4Code — Understand Code in Python Notebooks (1st–3rd Place)** | Ordering Markdown & Code Cells (Kendall Tau $\tau$) | `microsoft/codebert-base` + `deberta-v3-large` + Pairwise GBDT Reranker | Preserving code indentation, AST-like token patterns (`def`, `import`, `class`, variable assignments), and cell position ratios; concatenating local cell context with notebook-level global summary tokens. |
| **Malicious URL, SQLi & WAF Payload Detection Benchmarks (IEEE S&P / Kaggle Cybersecurity)** | Phishing URL & Injection Payload Classification (Macro-F1 / ROC-AUC) | **Char `(2, 6)` TF-IDF** (`analyzer="char"`) + Lexical/Entropy GBDT + **1D Multi-Scale Byte/Char CNN** | Sliding-window **Shannon entropy** to catch Base64/hex obfuscation and DGA domains; counting unbalanced quotes/brackets (`'`, `--`, `/*`, `<script>`); character transition Markov probabilities to spot non-natural symbol sequences. |
| **Web DOM & E-Commerce Product Extraction (WDC / SWDE / MarkupLM Benchmarks)** | DOM Node & Web Page Classification (Macro-F1) | **DOM XPath / CSS Path Serialization** + `DeBERTa-v3` / `MarkupLM` + GBDT | Serializing ancestor tag paths (`html>body>div.product>span.price`) and attribute names alongside text content (`[TAG:span.price] Rp 150.000 [/TAG]`) instead of stripping HTML tags with BeautifulSoup `.get_text()`. |

---

## 2. Distilled Architectural Playbook

### A. Why `analyzer="char"` (Not `char_wb`) Is Critical for Payloads & Logs
- In standard prose NLP, `analyzer="char_wb"` pads n-grams only within whitespace-delimited words.
- In URLs, SQL injections (`1'/**/OR/**/1=1--`), file paths (`/var/log/auth.log`), and minified HTML/JS, **there is often no whitespace at all**, or the boundary between punctuation and alphanumeric characters IS the signal.
- Always use `TfidfVectorizer(analyzer="char", ngram_range=(2, 6), sublinear_tf=True, lowercase=False)` so case-sensitive encodings (Base64, camelCase identifiers, SQL keywords) and cross-delimiter transitions are preserved.

### B. Three-Pillar Feature Engineering for Semi-Structured Text
1. **Structural Skeleton Tokens (`modules/dom_and_markup_tokenizer.py` & `modules/drain_log_template_miner.py`):**
   - Extract the *invariant skeleton* separately from the *variable values*:
     - For HTML/XML: sequence of `tag.class#id` paths and attribute keys.
     - For JSON: flattened dot-separated key paths (`request.headers.user_agent`) and value type tokens (`TYPE_INT`, `TYPE_STR`, `TYPE_NULL`).
     - For Logs: Drain template ID (`T_0042`) and masked template string (`Failed password for <USER> from <IP> port <NUM>`).
2. **Information-Theoretic & Obfuscation Signals (`modules/payload_entropy_and_ast_features.py`):**
   - **Global & Max-Window Shannon Entropy:** Normal prose has character entropy $\approx 3.8–4.3$ bits/char; encrypted/Base64/hex-encoded payloads spike to $> 4.8–5.5$ bits/char, while repetitive padding attacks drop to $< 2.0$ bits/char.
   - **Character Transition Markov Score:** Log-probability of bigrams under a clean reference transition matrix immediately flags random domain generation (DGA), obfuscated variable names (`_0x4a8f`), and binary-in-text dumps.
3. **Syntax & AST Balance Counters:**
   - Unmatched quotes (`'`, `"`, `` ` ``) and asymmetric brackets (`(`, `)`, `<`, `>`, `{`, `}`) are the #1 signature of injection payloads (SQLi, XSS, SSTI) and truncated/corrupted log records.

### C. Post-Processing via Template Determinism (`postprocess.py`)
- In system log and automated scrape datasets, up to 60–90% of test rows map to a **Drain log template** or **DOM/JSON schema signature** already seen in the training set.
- If a template occurs $\ge N_{\min}$ times in out-of-fold training data with empirical class purity $\ge 98\%$, blending the model's instance probability toward the template's empirical prior fixes noisy parameter variations and improves both LogLoss and Macro-F1.
