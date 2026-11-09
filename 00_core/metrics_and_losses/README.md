# `00_core/metrics_and_losses/` — Metrics, Losses & Threshold Optimization

## Scope
Exact metric implementations, post-training threshold/cutpoint optimizers on Out-Of-Fold (OOF) predictions, and custom loss functions for PyTorch.

## Scripts & CLI Usage
- `threshold_optimizers.py` — `optimize_binary_threshold` (`f1`, `mcc`, `fbeta`, `jaccard`), `optimize_multiclass_multipliers` (Powell / Nelder-Mead for Macro-F1), `optimize_multilabel_thresholds` (with Empirical Bayes shrinkage toward global threshold for rare labels), and `OptimizedRounder` (Nelder-Mead cutpoint optimization for Quadratic Weighted Kappa QWK).
  ```bash
  python3 00_core/metrics_and_losses/threshold_optimizers.py --smoke-test
  ```
- `pytorch_losses.py` — Standalone PyTorch `nn.Module` losses: `FocalLoss` (binary & multiclass), `AsymmetricLossOptimized` (ASL for imbalanced multilabel), `ZLPRLoss`, `PearsonCorrelationLoss`, `CosineSimilarityLoss`, `RDropLoss`, and `BiTemperedLogisticLoss`.
  ```bash
  python3 00_core/metrics_and_losses/pytorch_losses.py --smoke-test
  ```
- `ranking_and_ir_metrics.py` — Vectorized NumPy implementations of `apk`, `map_at_k`, `ndcg_at_k` (list & dense matrix), `mrr_at_k`, `recall_at_k`, `hit_rate_at_k`, and `fbeta_retrieval_score`.
  ```bash
  python3 00_core/metrics_and_losses/ranking_and_ir_metrics.py --smoke-test
  ```
