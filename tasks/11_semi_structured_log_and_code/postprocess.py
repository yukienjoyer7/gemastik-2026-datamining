#!/usr/bin/env python3
"""
Task 11 — Semi-Structured Log, Markup & Code: Template-Consistency Post-Processing
==================================================================================
Standalone post-processing pipeline providing:
  1. `TemplateConsistencyCalibrator`:
     Extracts each row's deterministic structural template (masking variable IPs,
     timestamps, hex IDs, strings, and numbers). If a template occurs at least
     `min_support` times in the training fold with empirical class purity >= `min_purity`,
     blends the predicted probability vector toward the template's empirical prior.
  2. `TemperatureScaler`:
     Calibrates OOF probabilities via L-BFGS-B to minimize LogLoss.
  3. `optimize_class_multipliers`:
     Coordinate search + Nelder-Mead simplex optimization of class probability
     multipliers to directly maximize Out-Of-Fold Macro-F1.
  4. Standardized OOF & test output (`oof_<exp_name>.npy`, `test_<exp_name>.npy`,
     `sub_<exp_name>.csv`).

Usage:
    python postprocess.py --oof-path oof_tier1_semistruct.npy --test-probs-path test_tier1_semistruct.npy
    python postprocess.py --smoke-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import site
import sys
import tempfile
from collections import defaultdict
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
from sklearn.metrics import f1_score, log_loss


@dataclass
class CFG:
    """Configuration for Task 11 Template-Consistency Post-Processor."""

    exp_name: str = "tier1_semistruct_postproc"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    oof_path: str = "oof_tier1_semistruct.npy"
    test_probs_path: str = "test_tier1_semistruct.npy"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "text"
    target_col: str = "label"
    fold_col: str = "fold"

    min_template_support: int = 3
    min_template_purity: float = 0.95
    template_blend_alpha: float = 0.65
    max_nelder_mead_iter: int = 250
    seed: int = 42
    smoke_test: bool = False


def extract_deterministic_signature(raw_text: str) -> str:
    """
    Map a raw log line, DOM snippet, JSON record, or code block to its canonical
    invariant structural template by masking IPs, hex numbers, strings, and numbers.
    """
    s = str(raw_text or "").strip()
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<TIME>", s)
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "<IP>", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b|\bblk_-?\d+\b", "<HEX>", s)
    s = re.sub(r'"[^"]*"|\'[^\']*\'', "<STR>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", s)
    s = re.sub(r"\s+", " ", s)
    return s


class TemplateConsistencyCalibrator:
    """
    Learns empirical class distributions for deterministic log/AST/DOM templates.
    If all (or >= `min_purity`) training instances of template `T` with support >= `min_support`
    share label `c`, test instances matching `T` are calibrated toward that template's distribution.
    """

    def __init__(
        self,
        min_support: int = 3,
        min_purity: float = 0.95,
        blend_alpha: float = 0.65,
    ) -> None:
        self.min_support = min_support
        self.min_purity = min_purity
        self.blend_alpha = blend_alpha
        self.template_priors_: Dict[str, np.ndarray] = {}

    def fit(
        self, texts: Sequence[str], y_true: np.ndarray, num_classes: int
    ) -> "TemplateConsistencyCalibrator":
        counts_by_tmpl: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(num_classes, dtype=np.float64)
        )
        for t, y_i in zip(texts, y_true):
            sig = extract_deterministic_signature(t)
            counts_by_tmpl[sig][int(y_i)] += 1.0

        self.template_priors_.clear()
        for sig, cnt_vec in counts_by_tmpl.items():
            total = float(cnt_vec.sum())
            if total >= self.min_support:
                purity = float(cnt_vec.max()) / total
                if purity >= self.min_purity:
                    smoothed = (cnt_vec + 1e-3) / (total + 1e-3 * num_classes)
                    self.template_priors_[sig] = smoothed
        return self

    def transform(self, texts: Sequence[str], probs: np.ndarray) -> Tuple[np.ndarray, int]:
        out = probs.copy().astype(np.float64)
        matched_count = 0
        for i, t in enumerate(texts):
            sig = extract_deterministic_signature(t)
            if sig in self.template_priors_:
                prior = self.template_priors_[sig]
                out[i] = (1.0 - self.blend_alpha) * out[i] + self.blend_alpha * prior
                matched_count += 1
        out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out, matched_count


def fit_oof_template_calibration(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: np.ndarray,
    folds: Optional[np.ndarray],
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    num_classes = oof_probs.shape[1]
    if folds is None:
        cal = TemplateConsistencyCalibrator(
            cfg.min_template_support, cfg.min_template_purity, cfg.template_blend_alpha
        ).fit(train_texts, y_true, num_classes)
        oof_cal, oof_hits = cal.transform(train_texts, oof_probs)
        test_cal, test_hits = cal.transform(test_texts, test_probs)
        return oof_cal, test_cal, oof_hits, test_hits

    oof_cal = np.zeros_like(oof_probs, dtype=np.float64)
    oof_hits = 0
    for f in np.sort(np.unique(folds)):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        cal_f = TemplateConsistencyCalibrator(
            cfg.min_template_support, cfg.min_template_purity, cfg.template_blend_alpha
        ).fit([train_texts[i] for i in tr_idx], y_true[tr_idx], num_classes)
        va_cal, hits = cal_f.transform([train_texts[i] for i in va_idx], oof_probs[va_idx])
        oof_cal[va_idx] = va_cal
        oof_hits += hits

    full_cal = TemplateConsistencyCalibrator(
        cfg.min_template_support, cfg.min_template_purity, cfg.template_blend_alpha
    ).fit(train_texts, y_true, num_classes)
    test_cal, test_hits = full_cal.transform(test_texts, test_probs)
    return oof_cal, test_cal, oof_hits, test_hits


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
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: np.ndarray,
    test_ids: np.ndarray,
    folds: Optional[np.ndarray],
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_f1 = float(f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro"))
    raw_ll = float(log_loss(y_true, oof_probs))

    oof_tmpl, test_tmpl, oof_hits, test_hits = fit_oof_template_calibration(
        train_texts, test_texts, oof_probs, test_probs, y_true, folds, cfg
    )
    tmpl_f1 = float(f1_score(y_true, np.argmax(oof_tmpl, axis=1), average="macro"))
    tmpl_ll = float(log_loss(y_true, oof_tmpl))

    temp = optimize_temperature(oof_tmpl, y_true)
    oof_temp = apply_temperature(oof_tmpl, temp)
    test_temp = apply_temperature(test_tmpl, temp)
    temp_ll = float(log_loss(y_true, oof_temp))

    w = optimize_class_multipliers(oof_temp, y_true, max_iter=cfg.max_nelder_mead_iter)
    oof_final = (oof_temp * w) / np.maximum((oof_temp * w).sum(axis=1, keepdims=True), 1e-12)
    test_final = (test_temp * w) / np.maximum((test_temp * w).sum(axis=1, keepdims=True), 1e-12)
    final_f1 = float(f1_score(y_true, np.argmax(oof_final, axis=1), average="macro"))

    print("=" * 72)
    print("TASK 11 TEMPLATE-CONSISTENCY POST-PROCESSING SUMMARY")
    print("=" * 72)
    print(f"  Raw OOF               : Macro-F1={raw_f1:.4f} | LogLoss={raw_ll:.4f}")
    print(
        f"  Template Calibrated   : Macro-F1={tmpl_f1:.4f} | LogLoss={tmpl_ll:.4f} "
        f"(OOF hits={oof_hits}/{len(oof_probs)}, Test hits={test_hits}/{len(test_probs)})"
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
    parser = argparse.ArgumentParser(description="Task 11 Template-Consistency Post-Processor")
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
        n_oof, n_test, n_classes = 120, 40, 3
        templates = {
            0: "2026-11-10 12:00:{s:02d} sshd[{p}]: Accepted publickey from 10.0.0.{ip}",
            1: "GET /item.php?id={p}' UNION SELECT 1,2 -- from 192.168.1.{ip}",
            2: '{{"event": "deploy", "node": {p}, "ip": "172.16.0.{ip}"}}',
        }
        y_true = np.array([i % n_classes for i in range(n_oof)])
        folds = np.array([i % 4 for i in range(n_oof)])
        train_texts = [
            templates[y_i].format(s=i % 60, p=100 + i, ip=2 + (i % 200))
            for i, y_i in enumerate(y_true)
        ]
        test_texts = [
            templates[i % n_classes].format(s=i % 60, p=500 + i, ip=10 + (i % 200))
            for i in range(n_test)
        ]

        logits = rng.normal(0.0, 1.4, size=(n_oof, n_classes))
        for i, y_i in enumerate(y_true):
            logits[i, y_i] += 1.4
        oof_probs = softmax(logits, axis=1)

        test_logits = rng.normal(0.0, 1.4, size=(n_test, n_classes))
        for i in range(n_test):
            test_logits[i, i % n_classes] += 1.4
        test_probs = softmax(test_logits, axis=1)
        test_ids = np.array([f"te_{i}" for i in range(n_test)])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            oof_final, test_final = run_postprocess(
                train_texts, test_texts, oof_probs, test_probs, y_true, test_ids, folds, cfg
            )
            assert oof_final.shape == oof_probs.shape
            assert test_final.shape == test_probs.shape
            raw_f1 = f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro")
            post_f1 = f1_score(y_true, np.argmax(oof_final, axis=1), average="macro")
            assert post_f1 >= raw_f1
            print("[SUCCESS] Task 11 postprocess smoke test passed.")
    else:
        p_tr = pathlib.Path(cfg.train_path)
        train_df = pd.read_parquet(p_tr) if p_tr.suffix.lower() == ".parquet" else pd.read_csv(p_tr)
        p_te = pathlib.Path(cfg.test_path)
        test_df = pd.read_parquet(p_te) if p_te.suffix.lower() == ".parquet" else pd.read_csv(p_te)
        oof_probs = np.load(cfg.oof_path)
        test_probs = np.load(cfg.test_probs_path)
        folds = train_df[cfg.fold_col].values if cfg.fold_col in train_df.columns else None
        run_postprocess(
            train_df[cfg.text_col].fillna("").astype(str).tolist(),
            test_df[cfg.text_col].fillna("").astype(str).tolist(),
            oof_probs,
            test_probs,
            train_df[cfg.target_col].values,
            test_df[cfg.id_col].values,
            folds,
            cfg,
        )


if __name__ == "__main__":
    main()
