# Task 11: Semi-Structured Log, Markup, Payload & Code Data

## 1. Formulation & Why Standard NLP Preprocessing Fails
- **Input:** Non-prose text strings where syntax, punctuation, delimiters, nesting hierarchy, and byte-level anomalies carry the primary discriminative signal:
  1. **Raw HTML / XML / DOM Fragments:** E-commerce product scrapes, phishing page HTML, web node classification.
  2. **JSON / Key-Value / Config Dumps:** API payloads, NoSQL audit trails, nested configuration records.
  3. **System & Cybersecurity Logs:** Linux `syslog`, Apache/Nginx access logs, HDFS/Spark logs, firewall/SIEM alerts.
  4. **URL / SQLi / XSS / Command-Injection Payloads:** Web Application Firewall (WAF) detection, malicious URL classification.
  5. **Source Code & AST Snippets:** Vulnerability detection, code authorship/language classification, notebook cell ordering (AI4Code).
- **Critical Rule:** **NEVER strip punctuation, brackets (`< > { } [ ] ( )`), slashes, quotes, or hex digits!** Naive regex cleaning (`re.sub(r"[^a-zA-Z0-9\s]", " ", text)`) destroys HTML tag boundaries, SQL injection tautologies (`' OR 1=1 --`), directory traversal (`../../etc/passwd`), and code control flow.

## 2. Validation Strategy & Leakage Traps
- **Log Template / Session Grouping (`StratifiedGroupKFold`):** System logs come in bursts from the same `host_id`, `session_id`, `block_id` (e.g., HDFS block IDs), or deterministic Drain log template. Split by session/host/template cluster so the model generalizes to unseen log sessions.
- **Domain / Repository Grouping:** For malicious URLs or source code, group by registered domain (`example.com`) or repository/project ID to prevent memorizing domain-specific paths or variable naming conventions.
- **Temporal Ordering:** When timestamps exist in logs, verify performance on a forward-in-time holdout split to catch concept drift (new attack signatures or software version upgrades).

## 3. Escalation Playbook
- **Tier 1 (CPU, <2 mins):**
  - **Structure-Preserving Tokenization** (DOM tag paths `html>body>div.price`, JSON key paths `user.role`, Drain log template tokens `<IP>`, `<HEX>`, `<PATH>`) + **Byte/Char `(2, 6)` TF-IDF** (`analyzer="char"`, not `char_wb`!) + **Structural & Entropy Features** (Shannon entropy, symbol density, bracket depth, hex/base64 counts, SQL/XSS/Code keyword hits) $\rightarrow$ Out-of-Fold **LightGBM / HistGBDT + LogisticRegression + Ridge** blend (`baselines/tier1_fast_cpu.py`).
- **Tier 2 (Deep Learning / Byte & Code Backbones):**
  - Pretrained code/byte models (`microsoft/codebert-base`, `google/byt5-small`, `microsoft/deberta-v3-small` with structural special tokens `[TAG_OPEN]`, `[TAG_CLOSE]`, `[KEY]`, `[VAL]`, `[IP]`, `[HEX]`, `[PATH]`) or a fast **Multi-Scale 1D Byte/Char CNN + Transformer Encoder** (`baselines/tier2_strong_dl.py`).
- **Tier 3 (Competition Tricks & Post-Processing):**
  - **Drain Log Template Mining (`modules/drain_log_template_miner.py`):** Group millions of noisy log lines into fixed structural templates (`template_id`), extract variable parameters, and compute per-template frequency & transition features.
  - **Sliding-Window Shannon Entropy (`modules/payload_entropy_and_ast_features.py`):** Detects obfuscated JavaScript, Base64 shellcode, encrypted payloads, and randomized domain generation algorithms (DGA).
  - **Template-Level Consistency Post-Processing (`postprocess.py`):** Calibrate test predictions that map to deterministic training log templates or AST signatures, followed by temperature scaling and Nelder-Mead class multiplier optimization.

## 4. Directory Contents
- `eda_and_data.py` — Sub-modality auto-detector (`html_xml`, `json_kv`, `system_log`, `url_sql_payload`, `source_code`), character-class & symbol density auditor, nesting depth profiler, and train-test template overlap checker.
- `baselines/tier1_fast_cpu.py` — Standalone OOF pipeline combining Structure-Preserving Tokenization + Char `(2, 6)` TF-IDF + Structural/Entropy features into a GBDT + Linear ensemble (`oof_tier1_semistruct.npy`, `test_tier1_semistruct.npy`, `sub_tier1_semistruct.csv`).
- `baselines/tier2_strong_dl.py` — Byte/Character-Aware Sequence Classifier (`CodeBERT` / `DeBERTa` with structural tokens + 1D Multi-Scale Char/Byte CNN-Transformer fallback) trained with OOF validation.
- `modules/dom_and_markup_tokenizer.py` — Parses raw HTML/XML/JSON strings into hierarchical tag paths (`html>body>div.price`), attribute tokens, JSON key paths (`request.auth.token`), and Transformer-ready `[TAG:...]` serialized sequences.
- `modules/drain_log_template_miner.py` — Pure-Python Drain-style fixed-depth parse tree for log template clustering, variable masking (`<IP>`, `<HEX>`, `<UUID>`, `<PATH>`, `<NUM>`), and parameter list extraction.
- `modules/payload_entropy_and_ast_features.py` — Sliding-window Shannon entropy, character transition Markov log-likelihoods, bracket/quote balance checks, and SQL/XSS/Shell/AST structural counters.
- `postprocess.py` — Deterministic log-template / AST-signature consistency calibration + Temperature Scaling + Nelder-Mead threshold/multiplier optimization.
- `winners/SUMMARY.md` — Curated references on Log Anomaly Detection (LogHub / Drain3 + GBDT), Kaggle Google AI4Code, Malicious URL/Payload detection, and Web DOM extraction competitions.
