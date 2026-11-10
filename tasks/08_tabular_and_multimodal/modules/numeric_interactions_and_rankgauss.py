"""
Task 08 — Numeric Interaction Generator & RankGauss Scaler
==========================================================
Standalone numerical feature engineering & preprocessing module:
  1. `NumericInteractionGenerator`: Builds pairwise ratios (`a / b`), differences (`a - b`),
     and products (`a * b`) among top-correlated numerical columns.
  2. `RankGaussScaler`: Median imputation + `QuantileTransformer(output_distribution='normal')`
     essential for training Tabular Neural Networks (ResNet / Gated MLP) alongside GBDTs.

Usage:
  python3 numeric_interactions_and_rankgauss.py --smoke-test
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

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
from sklearn.preprocessing import QuantileTransformer


@dataclass
class CFG:
    exp_name: str = "numeric_interactions_rankgauss"
    seed: int = 42
    n_quantiles: int = 50


class NumericInteractionGenerator:
    """Generates pairwise ratios, differences, and products for numerical columns."""

    def __init__(self, cols: Sequence[str], max_cols: int = 8) -> None:
        self.cols: List[str] = list(cols)[:max_cols]

    def fit(
        self, df: pd.DataFrame, target: Sequence | np.ndarray | None = None
    ) -> "NumericInteractionGenerator":
        if target is not None and len(self.cols) > 2:
            y = np.asarray(target, dtype=np.float64)
            corrs = []
            for c in self.cols:
                vals = df[c].fillna(df[c].median()).to_numpy(dtype=np.float64)
                if np.std(vals) > 1e-9 and np.std(y) > 1e-9:
                    corr = abs(float(np.corrcoef(vals, y)[0, 1]))
                else:
                    corr = 0.0
                corrs.append((corr, c))
            corrs.sort(reverse=True)
            self.cols = [c for _, c in corrs]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for i in range(len(self.cols)):
            for j in range(i + 1, len(self.cols)):
                c1, c2 = self.cols[i], self.cols[j]
                v1 = df[c1].to_numpy(dtype=np.float32)
                v2 = df[c2].to_numpy(dtype=np.float32)
                out[f"{c1}_minus_{c2}"] = v1 - v2
                out[f"{c1}_times_{c2}"] = v1 * v2
                out[f"{c1}_div_{c2}"] = v1 / (np.abs(v2) + 1e-5)
        return out


class RankGaussScaler:
    """
    Applies median imputation + `QuantileTransformer(output_distribution='normal')`
    (RankGauss), transforming skewed/heavy-tailed tabular features into smooth Gaussians for Neural Nets.
    """

    def __init__(self, n_quantiles: int = 100, random_state: int = 42) -> None:
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.medians_: np.ndarray = np.empty(0)
        self.qt_: QuantileTransformer | None = None

    def fit(self, X: np.ndarray) -> "RankGaussScaler":
        X_arr = np.asarray(X, dtype=np.float32)
        self.medians_ = np.nanmedian(X_arr, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        X_filled = np.where(np.isnan(X_arr), self.medians_[None, :], X_arr)
        n_q = max(5, min(self.n_quantiles, len(X_filled)))
        self.qt_ = QuantileTransformer(
            n_quantiles=n_q, output_distribution="normal", random_state=self.random_state
        )
        self.qt_.fit(X_filled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_filled = np.where(np.isnan(X_arr), self.medians_[None, :], X_arr)
        assert self.qt_ is not None
        return self.qt_.transform(X_filled).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _run_smoke_test(cfg: CFG) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 60
    df = pd.DataFrame(
        {
            "num_1": rng.normal(100.0, 20.0, size=n),
            "num_2": rng.exponential(5.0, size=n),
            "num_3": rng.uniform(1.0, 50.0, size=n),
        }
    )
    y = (df["num_1"] * 0.5 - df["num_2"] * 2.0 > 40.0).astype(int)

    gen = NumericInteractionGenerator(cols=["num_1", "num_2", "num_3"]).fit(df, y)
    inter_df = gen.transform(df)

    rg = RankGaussScaler(n_quantiles=cfg.n_quantiles, random_state=cfg.seed)
    scaled = rg.fit_transform(inter_df.to_numpy())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        np.save(out_dir / f"oof_{cfg.exp_name}.npy", scaled)
        np.save(out_dir / f"test_{cfg.exp_name}.npy", scaled[:10])
        inter_df.head(10).to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)

    print(
        f"[Smoke Test OK] Interactions shape={inter_df.shape} | "
        f"RankGauss mean={float(np.mean(scaled)):.3f}, std={float(np.std(scaled)):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08 Numeric Interactions & RankGauss Module")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast module smoke test")
    args = parser.parse_args()

    cfg = CFG()
    _run_smoke_test(cfg)


if __name__ == "__main__":
    main()
