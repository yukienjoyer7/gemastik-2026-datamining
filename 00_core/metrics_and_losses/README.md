# `00_core/metrics_and_losses/` — Metrics, Losses & Threshold Optimization

## Scope
Exact metric implementations, post-training threshold/cutpoint optimizers on Out-Of-Fold (OOF) predictions, and custom loss functions for PyTorch & GBDTs.

## Planned Standalone Scripts
- `threshold_optimizers.py` — Global, per-class, and rank-based threshold optimization (`scipy.optimize.minimize` Nelder-Mead / Powell, Optuna) for Macro-F1, Micro-F1, MCC, Jaccard, and Quadratic Weighted Kappa (QWK).
- `pytorch_losses.py` — Drop-in PyTorch losses: Focal Loss, Label Smoothing Cross-Entropy, Asymmetric Loss (ASL for multilabel), ZLPR (Zero-bounded Log-sum-exp & Pairwise Rank-based loss), Cosine/Pearson Correlation Loss, R-Drop KL divergence.
- `ranking_and_ir_metrics.py` — Fast vectorized MAP@K, NDCG@K, MRR@K, Recall@K.
