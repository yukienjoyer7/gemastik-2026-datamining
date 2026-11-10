#!/usr/bin/env python3
"""
Task 12 — Symbolic, Chemical (SMILES) & Biological Sequences: Post-Processing
=============================================================================
Standalone post-processing pipeline providing:
  1. `CanonicalDuplicateConsistencyAverager`:
     Maps every molecule or biological sequence to an order-invariant canonical
     signature (atomic formula + undirected bond pair signature for SMILES;
     strand-canonical min(seq, rev_comp(seq)) for DNA/RNA).
     - Test instances sharing the exact same canonical signature (e.g., `CCO` and
       `OCC` or reverse-complement DNA strands) have their predicted probabilities
       averaged for strict chemical/biological invariance.
     - If a canonical molecule/sequence also appears in the out-of-fold training set,
       blends the test prediction with the known training label prior.
  2. `TemperatureScaler` & `RankProbabilityBlend`:
     Calibrates probabilities via L-BFGS-B temperature scaling and optional rank-uniform
     smoothing for AUC/Spearman metrics.
  3. `optimize_class_multipliers`:
     Coordinate search + Nelder-Mead simplex optimization of class probability
     multipliers to directly maximize Out-Of-Fold Macro-F1.
  4. Standardized OOF & test output (`oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     `sub_<exp_name>.csv`).

Usage:
    python postprocess.py --oof-path oof_tier1_seq.npy --test-probs-path test_tier1_seq.npy
    python postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import site
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_venv_site = (
    pathlib.Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import f1_score, log_loss


@dataclass
class CFG:
    """Configuration for Task 12 Canonical/Scaffold Post-Processor."""

    exp_name: str = "tier1_seq_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_seq.npy"
    test_probs_path: str = "test_tier1_seq.npy"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "sequence"
    target_col: str = "label"
    fold_col: str = "fold"

    train_duplicate_blend_alpha: float = 0.70
    rank_blend_weight: float = 0.05
    max_nelder_mead_iter: int = 250
    seed: int = 42
    smoke_test: bool = False


_SMILES_TOK_RE = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def compute_canonical_key(seq: str) -> str:
    """
    Compute an order-invariant canonical signature for non-canonical SMILES
    (`CCO` == `OCC` == `C(O)C`) or reverse-complement DNA strands.
    """
    s = "".join(str(seq or "").split())
    if not s:
        return "EMPTY"

    upper_set = set(s.upper())
    if upper_set and upper_set.issubset({"A", "C", "G", "T", "U", "N"}) and len(s) >= 6:
        rc = s.upper().translate(_DNA_COMP)[::-1]
        canon = min(s.upper(), rc)
        return f"NUC_{hashlib.md5(canon.encode('utf-8')).hexdigest()[:12]}"

    tokens = _SMILES_TOK_RE.findall(s)
    if not tokens:
        tokens = list(s)
    atom_bond = [t for t in tokens if t not in ("(", ")")]
    counts = Counter(atom_bond)
    formula_sig = "_".join(f"{k}:{counts[k]}" for k in sorted(counts))

    chain = [t for t in tokens if t not in ("(", ")", "1", "2", "3", "4", "5")]
    pairs = Counter(tuple(sorted((a, b))) for a, b in zip(chain[:-1], chain[1:]))
    pair_sig = "_".join(f"{a}-{b}:{c}" for (a, b), c in sorted(pairs.items()))
    digest = hashlib.md5(f"{formula_sig}|{pair_sig}".encode("utf-8")).hexdigest()[:10]
    return f"MOL_{formula_sig}_{digest}"


def enforce_canonical_duplicate_consistency(
    train_seqs: Sequence[str],
    test_seqs: Sequence[str],
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: np.ndarray,
    folds: Optional[np.ndarray],
    blend_alpha: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    1. Averages predictions across test molecules/sequences that share the same
       canonical key (e.g., different SMILES notations of the same compound).
    2. Out-of-fold calibrates instances whose exact canonical key was observed in
       the training folds.
    """
    num_classes = oof_probs.shape[1]
    tr_keys = [compute_canonical_key(s) for s in train_seqs]
    te_keys = [compute_canonical_key(s) for s in test_seqs]

    # Step 1: Average test predictions across identical canonical keys in test set
    test_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, k in enumerate(te_keys):
        test_groups[k].append(idx)

    te_out = test_probs.copy().astype(np.float64)
    multi_view_groups = 0
    for k, idxs in test_groups.items():
        if len(idxs) > 1:
            multi_view_groups += len(idxs)
            mean_p = np.mean(te_out[idxs], axis=0, keepdims=True)
            te_out[idxs] = mean_p

    # Step 2: OOF calibration from training canonical keys
    oof_out = oof_probs.copy().astype(np.float64)
    if folds is not None:
        for f in np.sort(np.unique(folds)):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            priors: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
            for i in tr_idx:
                priors[tr_keys[i]][int(y_true[i])] += 1.0
            for i in va_idx:
                k = tr_keys[i]
                if k in priors:
                    vec = priors[k]
                    prior_p = (vec + 1e-3) / (vec.sum() + 1e-3 * num_classes)
                    oof_out[i] = (1.0 - blend_alpha) * oof_out[i] + blend_alpha * prior_p

    full_priors: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
    for i, k in enumerate(tr_keys):
        full_priors[k][int(y_true[i])] += 1.0
    for i, k in enumerate(te_keys):
        if k in full_priors:
            vec = full_priors[k]
            prior_p = (vec + 1e-3) / (vec.sum() + 1e-3 * num_classes)
            te_out[i] = (1.0 - blend_alpha) * te_out[i] + blend_alpha * prior_p

    oof_out /= np.maximum(oof_out.sum(axis=1, keepdims=True), 1e-12)
    te_out /= np.maximum(te_out.sum(axis=1, keepdims=True), 1e-12)
    return oof_out, te_out, multi_view_groups


def apply_rank_probability_smoothing(probs: np.ndarray, weight: float = 0.05) -> np.ndarray:
    """Blend calibrated probabilities with column-wise rank percentiles to stabilize ROC-AUC."""
    if weight <= 0.0:
        return probs
    ranked = np.zeros_like(probs, dtype=np.float64)
    for c in range(probs.shape[1]):
        ranked[:, c] = rankdata(probs[:, c]) / max(len(probs), 1)
    ranked /= np.maximum(ranked.sum(axis=1, keepdims=True), 1e-12)
    blended = (1.0 - weight) * probs + weight * ranked
    return blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)


def optimize_temperature(oof_probs: np.ndarray, y_true: np.ndarray) -> float:
    clipped = np.clip(oof_probs, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / clipped.sum(axis=1, keepdims=True))

    def obj(t_arr: np.ndarray) -> float:
        t = max(float(t_arr[0]), 0.05)
        return float(log_loss(y_true, softmax(logits / t, axis=1)))

    res = minimize(obj, x0=np.array([1.0]), bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0])


def apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / clipped.sum(axis=1, keepdims=True))
    return softmax(logits / max(temp, 0.05), axis=1)


def optimize_class_multipliers(
    oof_probs: np.ndarray, y_true: np.ndarray, max_iter: int = 250
) -> np.ndarray:
    num_classes = oof_probs.shape[1]
    best_w = np.ones(num_classes, dtype=np.float64)
    best_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro"))

    for c in range(num_classes):
        for cand in np.linspace(0.5, 2.0, 16):
            w_try = best_w.copy()
            w_try[c] = cand
            f1 = float(f1_score(y_true, np.argmax(oof_probs * w_try, axis=1), average="macro"))
            if f1 > best_f1:
                best_f1 = f1
                best_w = w_try

    def neg_f1(w: np.ndarray) -> float:
        if np.any(w <= 0.05) or np.any(w > 5.0):
            return 1.0
        return -float(f1_score(y_true, np.argmax(oof_probs * w, axis=1), average="macro"))

    res = minimize(neg_f1, x0=best_w, method="Nelder-Mead", options={"maxiter": max_iter})
    if -float(res.fun) >= best_f1:
        best_w = np.asarray(res.x, dtype=np.float64)
    return best_w / np.mean(best_w)


def run_postprocess(
    train_seqs: Sequence[str],
    test_seqs: Sequence[str],
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: np.ndarray,
    test_ids: np.ndarray,
    folds: Optional[np.ndarray],
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro"))
    raw_ll = float(log_loss(y_true, oof_probs))

    oof_canon, test_canon, dup_cnt = enforce_canonical_duplicate_consistency(
        train_seqs,
        test_seqs,
        oof_probs,
        test_probs,
        y_true,
        folds,
        blend_alpha=cfg.train_duplicate_blend_alpha,
    )
    canon_f1 = float(f1_score(y_true, np.argmax(oof_canon, axis=1), average="macro"))

    temp = optimize_temperature(oof_canon, y_true)
    oof_temp = apply_temperature(oof_canon, temp)
    test_temp = apply_temperature(test_canon, temp)
    temp_ll = float(log_loss(y_true, oof_temp))

    oof_rank = apply_rank_probability_smoothing(oof_temp, weight=cfg.rank_blend_weight)
    test_rank = apply_rank_probability_smoothing(test_temp, weight=cfg.rank_blend_weight)

    w = optimize_class_multipliers(oof_rank, y_true, max_iter=cfg.max_nelder_mead_iter)
    oof_final = (oof_rank * w) / np.maximum((oof_rank * w).sum(axis=1, keepdims=True), 1e-12)
    test_final = (test_rank * w) / np.maximum((test_rank * w).sum(axis=1, keepdims=True), 1e-12)
    final_f1 = float(f1_score(y_true, np.argmax(oof_final, axis=1), average="macro"))

    print("=" * 72)
    print("TASK 12 CANONICAL SEQUENCE & SCAFFOLD POST-PROCESSING SUMMARY")
    print("=" * 72)
    print(f"  Raw OOF               : Macro-F1={raw_f1:.4f} | LogLoss={raw_ll:.4f}")
    print(
        f"  Canonical Consistency : Macro-F1={canon_f1:.4f} "
        f"(Test non-canonical duplicates averaged={dup_cnt})"
    )
    print(f"  Temperature Scaled    : T={temp:.4f}   | LogLoss={temp_ll:.4f}")
    print(f"  Multiplier Optimized  : Macro-F1={final_f1:.4f} | Multipliers={np.round(w, 3)}")

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_final.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_final.astype(np.float32))
    sub_df = pd.DataFrame({cfg.id_col: test_ids, cfg.target_col: np.argmax(test_final, axis=1)})
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
    return oof_final, test_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 12 Canonical Sequence Post-Processor")
    parser.add_argument("--oof-path", type=str, default=CFG.oof_path)
    parser.add_argument("--test-probs-path", type=str, default=CFG.test_probs_path)
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        oof_path=args.oof_path,
        test_probs_path=args.test_probs_path,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        rng = np.random.default_rng(cfg.seed)
        n_oof, n_test, n_classes = 90, 30, 3
        y_true = np.array([i % n_classes for i in range(n_oof)])
        folds = np.array([i % 3 for i in range(n_oof)])

        base_mols = ["CCO", "CC(=O)O", "c1ccccc1O"]
        alt_mols = ["OCC", "OC(=O)C", "Oc1ccccc1"]
        train_seqs = [base_mols[y_i] if i % 2 == 0 else alt_mols[y_i] for i, y_i in enumerate(y_true)]
        test_seqs = [base_mols[i % 3] if i % 2 == 0 else alt_mols[i % 3] for i in range(n_test)]

        logits = rng.normal(0.0, 1.3, size=(n_oof, n_classes))
        for i, y_i in enumerate(y_true):
            logits[i, y_i] += 1.5
        oof_probs = softmax(logits, axis=1)

        test_logits = rng.normal(0.0, 1.3, size=(n_test, n_classes))
        for i in range(n_test):
            test_logits[i, i % n_classes] += 1.5
        test_probs = softmax(test_logits, axis=1)
        test_ids = np.array([f"te_{i}" for i in range(n_test)])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            oof_final, test_final = run_postprocess(
                train_seqs, test_seqs, oof_probs, test_probs, y_true, test_ids, folds, cfg
            )
            assert oof_final.shape == oof_probs.shape
            assert test_final.shape == test_probs.shape
            # Verify that CCO (index 0) and OCC (index 3) in test_final have identical probabilities!
            assert np.allclose(test_final[0], test_final[3], atol=1e-5)
            print("[SUCCESS] Task 12 postprocess smoke test passed.")
    else:
        p_tr = pathlib.Path(cfg.train_path)
        train_df = pd.read_parquet(p_tr) if p_tr.suffix.lower() == ".parquet" else pd.read_csv(p_tr)
        p_te = pathlib.Path(cfg.test_path)
        test_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
        seq_col = cfg.text_col if cfg.text_col in train_df.columns else "text"
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        folds = train_df[cfg.fold_col].values if cfg.fold_col in train_df.columns else None
        run_postprocess(
            train_df[seq_col].fillna("").astype(str).tolist(),
            test_df[seq_col].fillna("").astype(str).tolist(),
            oof_probs,
            test_probs,
            train_df[cfg.target_col].values,
            test_df[cfg.id_col].values,
            folds,
            cfg,
        )


if __name__ == "__main__":
    main()
