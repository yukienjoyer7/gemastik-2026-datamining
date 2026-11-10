# Winning Solutions & Curated References — Conversational Threads & Graph Text

## 1. Benchmark Competition & Shared-Task Hall of Fame

| Competition / Shared Task | Metric | 1st Place Architecture | Critical Winning Tricks |
| :--- | :--- | :--- | :--- |
| **SemEval RumourEval (SDQC Stance & Veracity in Reply Trees)** | Macro $F_1$ | Context-Window RoBERTa/DeBERTa (`[ROOT] + [PARENT] + [TARGET]`) + 2nd-Stage Tree GBDT | Explicit token budget reservation per tree zone (`root=64, parent=96, target=128`), 2nd-stage relational probability stacking over parent/root/child OOF predictions. |
| **Reddit / Twitter Conversational Sarcasm & Hate Speech Detection** | Macro $F_1$ / ROC-AUC | Segment-Aware Cross-Encoder + Author History Leave-One-Out Stacking | Sarcasm and implicit toxicity often invert the literal sentiment of `[PARENT]`. Modeling `(parent_text, target_text)` jointly + leave-one-out user prior boosted $F_1$ by $+6.0$ points. |
| **Citation & Hyperlink Document Classification (Cora / OGB-Arxiv / WebKB style)** | Accuracy / Macro $F_1$ | Fine-Tuned LM Node Embeddings + Out-Of-Fold Relational GBDT + Correct & Smooth (C&S) | Combining fine-tuned language model OOF logits with normalized Graph Laplacian probability propagation (`P_smooth = (1 - alpha) * P + alpha * A_norm @ P`). |

---

## 2. Deep-Dive Breakdown of Winning Patterns

### A. Token-Budgeted Conversational Serialization (`modules/thread_context_serializer.py`)
1. **Why Naive Concatenation Fails in Reply Trees:**
   - In Reddit/Twitter/forum threads, the root OP (`[ROOT]`) is often a 400-word article or rant, while the target comment (`[TARGET]`) is a 15-word reply.
   - If you concatenate `root + parent + target` and apply standard right-truncation at `max_length=256`, the `[TARGET]` post gets completely cut off!
2. **Zone-Budgeted Serialization:**
   - Allocate explicit token budgets to each structural zone:
     $$\texttt{[CLS] [ROOT] } \text{trunc}(x_{\text{root}}, B_r) \texttt{ [SEP] [PARENT] } \text{trunc}(x_{\text{parent}}, B_p) \texttt{ [SEP] [TARGET] } \text{trunc}(x_{\text{target}}, B_t) \texttt{ [SEP]}$$
   - Adding distinct segment/zone embeddings (`0=ROOT, 1=PARENT, 2=TARGET`) allows attention heads to distinguish whether a claim came from the OP or the reply (`baselines/tier2_strong_dl.py`).

---

### B. Two-Stage Relational Graph Stacking vs. End-to-End GNNs (`modules/graph_label_propagation.py`)
1. **Why 2-Stage OOF Stacking Beats End-to-End GNNs on Text Graphs:**
   - Jointly backpropagating through a large Transformer *and* a multi-hop Graph Neural Network (GNN) causes severe GPU memory bottlenecks and rapid overfitting on small thread counts.
   - Instead, competition winners decouple the problem into two leak-free stages:
     - **Stage 1:** Train the Context-Window Transformer (and Tier-1 TF-IDF model) using Connected-Component `GroupKFold` to produce calibrated Out-Of-Fold node probabilities $\mathbf{P}^{(1)} \in \mathbb{R}^{N \times C}$.
     - **Stage 2 (Relational Stacking):** Build a tabular feature matrix for each node $i$ containing:
       - $\mathbf{p}_i^{(1)}$ (self probability)
       - $\mathbf{p}_{\text{parent}(i)}^{(1)}$ and $\mathbf{p}_{\text{root}(i)}^{(1)}$
       - $\frac{1}{|\mathcal{C}(i)|}\sum_{c \in \mathcal{C}(i)} \mathbf{p}_c^{(1)}$ (direct children mean probability)
       - $\frac{1}{|\mathcal{U}(i)| - 1}\sum_{j \in \mathcal{U}(i) \setminus \{i\}} \mathbf{p}_j^{(1)}$ (**leave-one-out** same-user probability)
       - Structural topology features (`tree_depth`, `normalized_depth`, `out_degree`, `total_descendants`, `log_latency_from_parent`, `parent_child_jaccard`).
     - Train a regularized Linear / GBDT meta-model on these relational features.

---

### C. Graph Laplacian / MRF Post-Processing (`postprocess.py`)
- When strong label homophily exists across parent-child or same-author edges (verified via `audit_thread_and_graph_structure` in [`eda_and_data.py`](../eda_and_data.py)), applying 1–2 steps of normalized adjacency smoothing:
  $$\mathbf{P}_{\text{smooth}} = (1 - \alpha_{\text{tree}} - \alpha_{\text{user}})\mathbf{P} + \alpha_{\text{tree}}\tilde{\mathbf{A}}_{\text{tree}}\mathbf{P} + \alpha_{\text{user}}\tilde{\mathbf{A}}_{\text{user}}\mathbf{P}$$
  smooths out isolated noisy predictions and improves Macro-$F_1$ with zero extra model training.

---

## 3. Implementation Mapping in This Repository
- [`eda_and_data.py`](../eda_and_data.py): Audits tree depth, branching factor, orphan links, **parent-child & same-user homophily**, and builds connected-component `GroupKFold` splits.
- [`baselines/tier1_fast_cpu.py`](../baselines/tier1_fast_cpu.py): Multi-zone (`[TARGET]`, `[PARENT]`, `[ROOT]`, `[CHILDREN]`) TF-IDF + parent-child cosine similarity + tree topology features.
- [`baselines/tier2_strong_dl.py`](../baselines/tier2_strong_dl.py): Budgeted Context-Window Transformer + 2nd-stage Relational Graph Stacking pipeline.
- [`modules/thread_context_serializer.py`](../modules/thread_context_serializer.py): Token-budgeted serializer for `[ROOT]`, `[PARENT]`, `[PREV_SIBLING]`, `[TARGET]`, and `[CHILD_SUMMARY]`.
- [`modules/tree_topology_features.py`](../modules/tree_topology_features.py): Extracts depth, normalized depth, out-degree, subtree descendants, sibling rank, response latency, and parent-child lexical shift.
- [`modules/graph_label_propagation.py`](../modules/graph_label_propagation.py): Leak-free relational neighbor & leave-one-out user probability aggregator.
- [`postprocess.py`](../postprocess.py): Graph Laplacian / MRF probability smoothing over reply-tree and same-user edges.
