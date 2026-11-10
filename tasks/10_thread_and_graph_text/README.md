# Task 10: Conversational Threads, Reply Trees & Graph-Structured Text

## 1. Formulation & Common Competition Metrics
- **Input:** Posts, comments, or documents linked by relational structure:
  - **Reply Trees / Conversational Threads:** `(post_id, parent_id, root_id, thread_id, user_id, timestamp, text)` (e.g., Twitter/X rumor stance threads, Reddit comment trees, forum Q&A).
  - **User-Post Bipartite Networks:** Posts authored or shared by overlapping `user_id` nodes.
  - **Citation / Hyperlink Document Graphs:** `(source_doc_id, target_doc_id)` directed edges.
- **Output:** Node-level class probabilities (stance `Support/Deny/Query/Comment`, sarcasm/toxicity in context, rumor veracity, topic classification).
- **Metrics:** Macro-$F_1$, Micro-$F_1$, LogLoss, ROC-AUC.

## 2. Validation Strategy & Leakage Traps
- **Connected-Component / Thread `GroupKFold` Rule:** Never split nodes from the same `thread_id` (or weakly connected graph component) across train and validation! Doing so leaks parent/child context and inflates CV via label autocorrelation. Always use `create_thread_component_folds` in [`eda_and_data.py`](eda_and_data.py).
- **Self-Label Leakage in Same-User / Neighbor Aggregation:** When aggregating 1st-stage OOF probabilities across nodes authored by the same `user_id`, **you must subtract the target node's own probability** (`(sum_p - p_self) / (count - 1)`) via [`modules/graph_label_propagation.py`](modules/graph_label_propagation.py).

## 3. Escalation Playbook
- **Tier 1 (CPU Fast — Target + Parent/Root Context TF-IDF + Tree Topology Features):**
  - Separate TF-IDF spaces for `[TARGET]`, `[PARENT]`, and `[ROOT]` text + parent-child cosine similarity + tree depth, subtree descendant count, response latency, and author activity features (`baselines/tier1_fast_cpu.py`).
- **Tier 2 (Context-Window Transformer + 2nd-Stage Relational Graph Stacking):**
  - *Stage 1:* Dynamic token-budget serializer `"[CLS] [ROOT] {root} [SEP] [PARENT] {parent} [SEP] [TARGET] {target} [SEP]"` trained out-of-fold (`modules/thread_context_serializer.py` & `baselines/tier2_strong_dl.py`).
  - *Stage 2 (Relational Graph Stacking):* Aggregates 1st-stage OOF probabilities of `parent`, `root`, `children_mean`, `siblings_mean`, and `leave-one-out same_user_mean` alongside tree topology features into a 2nd-stage meta-classifier (`modules/graph_label_propagation.py` & `baselines/tier2_strong_dl.py`).
- **Tier 3 (Graph Laplacian / Markov Random Field Post-Processing):**
  - Smooths node probability distributions over the normalized adjacency matrix (`P_smooth = (1 - alpha) * P + alpha * A_norm @ P`) with separate signed/unsigned weights for parent-child and same-author edges (`postprocess.py`).

## 4. Directory Contents
- [`eda_and_data.py`](eda_and_data.py) — Audits thread tree depth, branching factor, orphan parent links, user post power-law, **homophily / label autocorrelation** $P(y_{\text{child}} \mid y_{\text{parent}})$, and builds connected-component `GroupKFold` splits.
- [`baselines/tier1_fast_cpu.py`](baselines/tier1_fast_cpu.py) — Target + Parent + Root TF-IDF + Tree & User Graph Topology Features + OOF GBDT/Linear classifier exporting `oof_tier1_thread.npy`, `test_tier1_thread.npy`, `sub_tier1_thread.csv`.
- [`baselines/tier2_strong_dl.py`](baselines/tier2_strong_dl.py) — Context-Window Transformer (`[ROOT] + [PARENT] + [TARGET]`) + 2nd-stage Relational Graph Stacking meta-model.
- [`modules/thread_context_serializer.py`](modules/thread_context_serializer.py) — Builds token-budget-aware hierarchical context strings (`[ROOT]`, `[PARENT]`, `[PREV_SIBLING]`, `[TARGET]`, `[CHILD_SUMMARY]`).
- [`modules/tree_topology_features.py`](modules/tree_topology_features.py) — Extracts tree depth, normalized depth, direct reply out-degree, total subtree descendants, sibling rank, response latency, and parent-child lexical shift.
- [`modules/graph_label_propagation.py`](modules/graph_label_propagation.py) — Leak-free Out-Of-Fold relational neighbor probability aggregator (`parent`, `root`, `children`, `siblings`, and leave-one-out `same_user`).
- [`postprocess.py`](postprocess.py) — Graph Laplacian / MRF probability smoothing over reply-tree and same-user edges tuned on OOF Macro-$F_1$.
- [`winners/SUMMARY.md`](winners/SUMMARY.md) — Curated references on SemEval RumourEval 1st-place solutions, conversational sarcasm/hate-speech detection, and Graph + Text stacking.
