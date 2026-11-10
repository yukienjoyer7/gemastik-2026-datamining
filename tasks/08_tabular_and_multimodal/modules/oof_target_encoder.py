"""
Task 08 — Out-Of-Fold Empirical Bayes Smoothed Target Encoder
=============================================================
Standalone leak-free target encoder `OOFTargetEncoder`:
  - Computes `TE(c) = (n_c * mean_c + m * global_prior) / (n_c + m)` strictly inside each CV fold
  - Supports binary, continuous regression, and multiclass (`one-vs-rest` class probability) targets
  - Fits full-dataset Empirical Bayes mapping for test-set transformation

Usage:
  python3 oof_target_encoder.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_venv_site = (
    Path.home()
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _venv_site.exists() and str(_venv_site) not in sys.path:
    site.addsitedir(str(_venv_site))

import numpy as np
import pandas as pd


@dataclass
class CFG:
    exp_name: str = "oof_target_encoder"
    seed: int = 42
    smoothing_m: float = 10.0


class OOFTargetEncoder:
    """
    Leak-free Out-Of-Fold Empirical Bayes Smoothed Target Encoder.
    Formula: `TE(c) = (n_c * mean_c + m * global_prior) / (n_c + m)`
    Supports binary/continuous 1D targets as well as multiclass targets (one-vs-rest columns).
    """

    def __init__(
        self, cols: Sequence[str], smoothing: float = 10.0, multiclass: bool = False
    ) -> None:
        self.cols = list(cols)
        self.smoothing = smoothing
        self.multiclass = multiclass
        self.global_priors_: Dict[str, float] = {}
        self.mapping_: Dict[Tuple[str, str], Dict[object, float]] = {}
        self.classes_: List = []

    def fit_transform_oof(
        self, df: pd.DataFrame, target: Sequence | np.ndarray, fold_col: str = "fold"
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        y = np.asarray(target)
        folds = df[fold_col].to_numpy()
        unique_folds = np.unique(folds)

        if self.multiclass:
            self.classes_ = sorted(np.unique(y).tolist())
            target_cols = {
                f"cls_{c}": (y == c).astype(np.float64) for c in self.classes_
            }
        else:
            target_cols = {"target": y.astype(np.float64)}

        for t_name, y_vec in target_cols.items():
            global_prior = float(np.mean(y_vec))
            self.global_priors_[t_name] = global_prior

            for col in self.cols:
                feat_name = (
                    f"{col}_te" if not self.multiclass else f"{col}_te_{t_name}"
                )
                oof_vals = np.full(len(df), global_prior, dtype=np.float32)

                for f in unique_folds:
                    tr_mask = folds != f
                    va_mask = folds == f
                    tr_s = pd.Series(y_vec[tr_mask]).groupby(df.loc[tr_mask, col].values)
                    cnts = tr_s.count()
                    means = tr_s.mean()
                    tr_prior = float(np.mean(y_vec[tr_mask]))
                    smooth_map = (
                        (cnts * means + self.smoothing * tr_prior)
                        / (cnts + self.smoothing)
                    ).to_dict()
                    oof_vals[va_mask] = (
                        df.loc[va_mask, col]
                        .map(smooth_map)
                        .fillna(tr_prior)
                        .to_numpy(dtype=np.float32)
                    )

                out[feat_name] = oof_vals

                full_s = pd.Series(y_vec).groupby(df[col].values)
                f_cnts = full_s.count()
                f_means = full_s.mean()
                self.mapping_[(col, t_name)] = (
                    (f_cnts * f_means + self.smoothing * global_prior)
                    / (f_cnts + self.smoothing)
                ).to_dict()

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        t_names = (
            [f"cls_{c}" for c in self.classes_] if self.multiclass else ["target"]
        )
        for t_name in t_names:
            prior = self.global_priors_[t_name]
            for col in self.cols:
                feat_name = (
                    f"{col}_te" if not self.multiclass else f"{col}_te_{t_name}"
                )
                s_map = self.mapping_.get((col, t_name), {})
                out[feat_name] = (
                    df[col].map(s_map).fillna(prior).astype(np.float32)
                )
        return out


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 60
    df = pd.DataFrame(
        {
            "cat_a": rng.choice(["A", "B", "C", "D"], size=n),
            "cat_b": rng.choice(["X", "Y"], size=n),
            "fold": np.arange(n) % 5,
            "target_bin": rng.integers(0, 2, size=n),
            "target_multi": rng.integers(0, 3, size=n),
        }
    )

    te_bin = OOFTargetEncoder(cols=["cat_a", "cat_b"], smoothing=cfg.smoothing_m)
    oof_bin = te_bin.fit_transform_oof(df, df["target_bin"], fold_col="fold")
    test_bin = te_bin.transform(df.iloc[:12])

    te_mc = OOFTargetEncoder(cols=["cat_a"], smoothing=cfg.smoothing_m, multiclass=True)
    oof_mc = te_mc.fit_transform_oof(df, df["target_multi"], fold_col="fold")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_bin.to_numpy(dtype=np.float32))
        np.save(out_dir / f"test_{cfg.exp_name}.npy", test_bin.to_numpy(dtype=np.float32))
        test_bin.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] OOFTargetEncoder binary shape={oof_bin.shape}, "
        f"multiclass shape={oof_mc.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 OOF Target Encoder Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
