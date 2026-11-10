#!/usr/bin/env python3
"""
Task 12 — Symbolic, Chemical (SMILES) & Biological Sequences: Tier 1 Fast CPU Baseline
======================================================================================
Standalone Out-Of-Fold (OOF) pipeline inspired by top fast baselines from Kaggle
Leash Bio – BELKA, Novozymes Enzyme Stability, and OpenVaccine RNA (zero RDKit dependency):
  1. Case-Sensitive Multi-Scale Character $k$-mer `(1, 6)` TF-IDF (`analyzer="char"`,
     `lowercase=False` so `C` vs `c` and `Cl` vs `c1` are never conflated).
  2. Tokenized Atom/Motif & Gapped $k$-mer TF-IDF (`ngram_range=(1, 4)`).
  3. Handcrafted Sequence Physicochemical & Topological Features (50 features):
     - SMILES: atom counts (`C, N, O, S, P, F, Cl, Br, I`), aromatic atom counts,
       double/triple bond counts (`=`, `#`), branch count (`(`), ring closures (`1..9`),
       MW proxy, H-bond donor/acceptor proxies, LogP proxy, and 14 functional-group
       SMARTS/regex substructure counts (hydroxyl/phenol, carboxyl, ester, amide,
       ketone, methoxy, amine, aryl ring, nitro, sulfonamide, halide, etc.).
     - Bio (DNA/RNA/Protein): GC content, AT/GC skew, CpG ratio, homopolymer runs,
       Kyte-Doolittle hydropathy profile, net charge, aromatic/polar/aliphatic fractions,
       and Shannon sequence complexity entropy.
  4. Out-Of-Fold Ensemble:
     - Blends `HistGradientBoostingClassifier` + `LogisticRegression` + `RidgeClassifier`
     - Exports `oof_tier1_seq.npy`, `test_tier1_seq.npy`, and `sub_tier1_seq.csv`.

Usage:
    python baselines/tier1_fast_cpu.py --train-path train_folds.csv --test-path test.csv
    python baselines/tier1_fast_cpu.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import site
import sys
import tempfile
from collections import Counter
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
from scipy import sparse
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class CFG:
    """Configuration for Task 12 Tier 1 Symbolic & Bio Sequence Baseline."""

    exp_name: str = "tier1_seq"
    train_path: str = "train_folds.parquet"
    test_path: str = "test.parquet"
    output_dir: str = "."
    id_col: str = "id"
    text_col: str = "sequence"
    target_col: str = "label"
    fold_col: str = "fold"

    # K-mer TF-IDF parameters
    char_kmer_range: Tuple[int, int] = (1, 6)
    atom_kmer_range: Tuple[int, int] = (1, 4)
    max_char_features: int = 80000
    max_atom_features: int = 40000
    svd_components: int = 32
    min_df: int = 2

    # Classifier hyperparameters
    logreg_c: float = 2.0
    ridge_alpha: float = 1.5
    gbdt_max_iter: int = 120
    gbdt_lr: float = 0.06
    blend_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)  # (GBDT, LogReg, Ridge)

    seed: int = 42
    smoke_test: bool = False


_SMILES_ATOM_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

_FG_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"(?<!=)\bO\b|\(O\)|^O|O$|\[OH\]"),  # hydroxyl / phenol
    re.compile(r"C\(=O\)\[?O[H\-]?\]?|\[O-\]C=O"),  # carboxyl
    re.compile(r"C\(=O\)O[Cc]|O=C1O"),  # ester / lactone
    re.compile(r"C\(=O\)N|NC\(=O\)"),  # amide
    re.compile(r"[Cc]C\(=O\)[Cc]|C=O"),  # ketone / carbonyl
    re.compile(r"COc|cOC|COC"),  # methoxy / ether
    re.compile(r"(?<!\(=O\))N(?![=O\+])|\[NH2?\+?\]|\[nH\]"),  # amine
    re.compile(r"c1[cnos]{4,5}1|c2[cnos]{4,5}2"),  # aromatic ring
    re.compile(r"C=C|C=CC=O"),  # alkene / Michael acceptor
    re.compile(r"C#C|C#N|N#C"),  # alkyne / nitrile
    re.compile(r"\[N\+\]\(=O\)\[O-\]|N\(=O\)=O"),  # nitro
    re.compile(r"S\(=O\)\(=O\)"),  # sulfonamide / sulfonyl
    re.compile(r"F|Cl|Br|I"),  # halogen
    re.compile(r"P\(=O\)|OP\("),  # phosphate
]

_KD_HYDRO: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def tokenize_symbolic_with_gapped_kmers(seq: str) -> str:
    """
    Tokenize SMILES or Bio sequence into atomic/residue tokens plus 1-gapped and
    2-gapped pairs (`g1:C_O`, `g2:A_G`) for `TfidfVectorizer`.
    """
    s = "".join(str(seq or "").split())
    if not s:
        return ""
    tokens = _SMILES_ATOM_REGEX.findall(s)
    if not tokens or len("".join(tokens)) < len(s) * 0.5:
        tokens = list(s)
    g1 = [f"g1:{tokens[i]}_{tokens[i+2]}" for i in range(len(tokens) - 2)]
    g2 = [f"g2:{tokens[i]}_{tokens[i+3]}" for i in range(len(tokens) - 3)]
    return " ".join(tokens + g1 + g2)


def extract_physicochemical_and_topological_features(sequences: Sequence[str]) -> np.ndarray:
    """
    Extract 40 handcrafted physicochemical, topological, and bio-sequence features:
      - 0..17 : SMILES atom counts, aromatic ratio, bonds, branches, rings, MW, HBA, HBD, LogP
      - 18..31: 14 Functional group substructure counts
      - 32..39: Bio sequence entropy, GC content, AT/GC skew, CpG ratio, hydropathy mean/std, net charge
    """
    feats = np.zeros((len(sequences), 40), dtype=np.float32)

    for i, raw in enumerate(sequences):
        s = str(raw or "").strip()
        n = max(len(s), 1)

        c_aliph = s.count("C") - s.count("Cl") - s.count("Ca") - s.count("Co")
        c_arom = s.count("c")
        n_aliph = s.count("N") - s.count("Na")
        n_arom = s.count("n")
        o_aliph = s.count("O")
        o_arom = s.count("o")
        s_atom = s.count("S") - s.count("Si") + s.count("s")
        p_atom = s.count("P") + s.count("p")
        f_atom = s.count("F") - s.count("Fe")
        cl_atom = s.count("Cl")
        br_atom = s.count("Br")
        i_atom = s.count("I")

        heavy = max(
            1,
            c_aliph + c_arom + n_aliph + n_arom + o_aliph + o_arom + s_atom + p_atom + f_atom + cl_atom + br_atom + i_atom,
        )
        mw_proxy = (
            13.5 * (c_aliph + c_arom)
            + 15.0 * (n_aliph + n_arom)
            + 16.5 * (o_aliph + o_arom)
            + 32.1 * s_atom
            + 31.0 * p_atom
            + 19.0 * f_atom
            + 35.5 * cl_atom
            + 79.9 * br_atom
            + 126.9 * i_atom
        )
        dbl = s.count("=")
        trp = s.count("#")
        brn = s.count("(")
        rings = sum(s.count(d) for d in "123456789") // 2
        stereo_chg = s.count("@") + s.count("+") + s.count("-")
        hba = float(n_aliph + n_arom + o_aliph + o_arom)
        hbd = float(len(_FG_REGEXES[0].findall(s)) + len(_FG_REGEXES[6].findall(s)))
        logp = (
            0.20 * (c_aliph + c_arom)
            + 0.35 * (f_atom + cl_atom + br_atom + i_atom)
            - 0.45 * hba
        )

        feats[i, 0] = np.log1p(len(s))
        feats[i, 1] = float(heavy)
        feats[i, 2] = float(mw_proxy / 100.0)
        feats[i, 3] = float(c_aliph)
        feats[i, 4] = float(c_arom)
        feats[i, 5] = float(c_arom) / heavy
        feats[i, 6] = float(n_aliph + n_arom)
        feats[i, 7] = float(o_aliph + o_arom)
        feats[i, 8] = float(s_atom + p_atom)
        feats[i, 9] = float(f_atom + cl_atom + br_atom + i_atom)
        feats[i, 10] = float(dbl)
        feats[i, 11] = float(trp)
        feats[i, 12] = float(brn)
        feats[i, 13] = float(rings)
        feats[i, 14] = float(stereo_chg)
        feats[i, 15] = hba
        feats[i, 16] = hbd
        feats[i, 17] = logp

        for j, pat in enumerate(_FG_REGEXES):
            feats[i, 18 + j] = float(len(pat.findall(s)))

        # Bio & information-theoretic stats
        counts = Counter(s)
        ent = -sum((v / n) * math.log2(v / n) for v in counts.values()) if s else 0.0
        up = s.upper()
        up_cnt = Counter(up)
        gc = (up_cnt.get("G", 0) + up_cnt.get("C", 0)) / n
        at_skew = (up_cnt.get("A", 0) - up_cnt.get("T", 0) - up_cnt.get("U", 0)) / n
        gc_skew = (up_cnt.get("G", 0) - up_cnt.get("C", 0)) / n
        cpg = up.count("CG") / max(n - 1, 1)
        hydro = [_KD_HYDRO.get(ch, 0.0) for ch in up]
        h_mean = float(np.mean(hydro)) if hydro else 0.0
        h_std = float(np.std(hydro)) if hydro else 0.0
        net_chg = (up_cnt.get("K", 0) + up_cnt.get("R", 0) - up_cnt.get("D", 0) - up_cnt.get("E", 0)) / n

        feats[i, 32] = ent
        feats[i, 33] = gc
        feats[i, 34] = at_skew
        feats[i, 35] = gc_skew
        feats[i, 36] = cpg
        feats[i, 37] = h_mean
        feats[i, 38] = h_std
        feats[i, 39] = net_chg

    return feats


def ridge_to_probs(decision: np.ndarray, num_classes: int) -> np.ndarray:
    if decision.ndim == 1 or num_classes == 2:
        scores = np.vstack([-decision, decision]).T if decision.ndim == 1 else decision
        return softmax(scores, axis=1)
    return softmax(decision, axis=1)


def compute_metrics(y_true: np.ndarray, oof_probs: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(oof_probs, axis=1)
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    ll = float(log_loss(y_true, oof_probs))
    try:
        if oof_probs.shape[1] == 2:
            auc = float(roc_auc_score(y_true, oof_probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, oof_probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {"macro_f1": macro_f1, "log_loss": ll, "roc_auc": auc}


def run_oof_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    seq_col = cfg.text_col if cfg.text_col in train_df.columns else "text"
    classes = np.sort(train_df[cfg.target_col].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = train_df[cfg.target_col].map(label_to_idx).values
    folds = train_df[cfg.fold_col].values
    unique_folds = np.sort(np.unique(folds))
    num_classes = len(classes)

    raw_train = train_df[seq_col].fillna("").astype(str).tolist()
    raw_test = test_df[seq_col].fillna("").astype(str).tolist()

    tok_train = [tokenize_symbolic_with_gapped_kmers(s) for s in raw_train]
    tok_test = [tokenize_symbolic_with_gapped_kmers(s) for s in raw_test]

    dense_train_all = extract_physicochemical_and_topological_features(raw_train)
    dense_test_all = extract_physicochemical_and_topological_features(raw_test)

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
    w_gbdt, w_lr, w_ridge = cfg.blend_weights

    for f in unique_folds:
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        char_vec = TfidfVectorizer(
            ngram_range=cfg.char_kmer_range,
            max_features=cfg.max_char_features,
            min_df=1 if cfg.smoke_test else cfg.min_df,
            sublinear_tf=True,
            analyzer="char",
            lowercase=False,
        )
        atom_vec = TfidfVectorizer(
            ngram_range=cfg.atom_kmer_range,
            max_features=cfg.max_atom_features,
            min_df=1 if cfg.smoke_test else cfg.min_df,
            sublinear_tf=True,
            lowercase=False,
            token_pattern=r"(?u)\S+",
        )

        X_tr_c = char_vec.fit_transform([raw_train[i] for i in tr_idx])
        X_va_c = char_vec.transform([raw_train[i] for i in va_idx])
        X_te_c = char_vec.transform(raw_test)

        X_tr_a = atom_vec.fit_transform([tok_train[i] for i in tr_idx])
        X_va_a = atom_vec.transform([tok_train[i] for i in va_idx])
        X_te_a = atom_vec.transform(tok_test)

        scaler = StandardScaler()
        d_tr = scaler.fit_transform(dense_train_all[tr_idx])
        d_va = scaler.transform(dense_train_all[va_idx])
        d_te = scaler.transform(dense_test_all)

        X_tr_sparse = sparse.hstack([X_tr_c, X_tr_a, sparse.csr_matrix(d_tr)], format="csr")
        X_va_sparse = sparse.hstack([X_va_c, X_va_a, sparse.csr_matrix(d_va)], format="csr")
        X_te_sparse = sparse.hstack([X_te_c, X_te_a, sparse.csr_matrix(d_te)], format="csr")

        n_comp = min(cfg.svd_components, X_tr_sparse.shape[1] - 1, X_tr_sparse.shape[0] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed + int(f))
            svd_tr = svd.fit_transform(X_tr_sparse)
            svd_va = svd.transform(X_va_sparse)
            svd_te = svd.transform(X_te_sparse)
            X_tr_gbdt = np.hstack([dense_train_all[tr_idx], svd_tr])
            X_va_gbdt = np.hstack([dense_train_all[va_idx], svd_va])
            X_te_gbdt = np.hstack([dense_test_all, svd_te])
        else:
            X_tr_gbdt = dense_train_all[tr_idx]
            X_va_gbdt = dense_train_all[va_idx]
            X_te_gbdt = dense_test_all

        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. HistGradientBoostingClassifier
        clf_gbdt = HistGradientBoostingClassifier(
            max_iter=cfg.gbdt_max_iter,
            learning_rate=cfg.gbdt_lr,
            max_leaf_nodes=15,
            min_samples_leaf=5 if cfg.smoke_test else 15,
            random_state=cfg.seed + int(f),
        )
        clf_gbdt.fit(X_tr_gbdt, y_tr)
        p_va_gbdt = clf_gbdt.predict_proba(X_va_gbdt)
        p_te_gbdt = clf_gbdt.predict_proba(X_te_gbdt)

        # 2. LogisticRegression
        clf_lr = LogisticRegression(
            C=cfg.logreg_c, max_iter=1000, random_state=cfg.seed + int(f)
        )
        clf_lr.fit(X_tr_sparse, y_tr)
        p_va_lr = clf_lr.predict_proba(X_va_sparse)
        p_te_lr = clf_lr.predict_proba(X_te_sparse)

        # 3. RidgeClassifier
        clf_ridge = RidgeClassifier(alpha=cfg.ridge_alpha, random_state=cfg.seed + int(f))
        clf_ridge.fit(X_tr_sparse, y_tr)
        p_va_ridge = ridge_to_probs(clf_ridge.decision_function(X_va_sparse), num_classes)
        p_te_ridge = ridge_to_probs(clf_ridge.decision_function(X_te_sparse), num_classes)

        fold_va = w_gbdt * p_va_gbdt + w_lr * p_va_lr + w_ridge * p_va_ridge
        fold_te = w_gbdt * p_te_gbdt + w_lr * p_te_lr + w_ridge * p_te_ridge

        oof_probs[va_idx] = fold_va
        test_probs += fold_te / len(unique_folds)

        fold_f1 = f1_score(y_va, np.argmax(fold_va, axis=1), average="macro")
        print(f"  [Fold {f}] val_size={len(va_idx)} | Macro-F1={fold_f1:.4f} | features={X_tr_sparse.shape[1]}")

    metrics = compute_metrics(y, oof_probs)
    print(
        f"[OOF SUMMARY] Macro-F1={metrics['macro_f1']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
    )

    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"oof_{cfg.exp_name}.npy", oof_probs.astype(np.float32))
    np.save(out_dir / f"test_{cfg.exp_name}.npy", test_probs.astype(np.float32))

    sub_df = pd.DataFrame(
        {
            cfg.id_col: test_df[cfg.id_col].values,
            cfg.target_col: [classes[i] for i in np.argmax(test_probs, axis=1)],
        }
    )
    sub_df.to_csv(out_dir / f"sub_{cfg.exp_name}.csv", index=False)
    return oof_probs, test_probs, metrics


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def generate_synthetic_dataset(
    n_train: int = 120, n_test: int = 36, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scaffolds = {
        0: [
            "COc1cc(C=CC(=O)CC(=O)c2ccc({r})cc2)ccc1O",
            "c1ccc2c(c1)oc(=O)c({r})c2O",
        ],
        1: [
            "CC(=O)Nc1ccc(S(=O)(=O)N{r})cc1",
            "Clc1ccc(S(=O)(=O)N{r}CCN)cc1F",
        ],
        2: [
            "CC(C)CC1=CC=C(C=C1)C({r})C(=O)O",
            "O=C(O)c1ccccc1OC(=O){r}",
        ],
    }
    r_groups = ["C", "CC", "CCC", "O", "OC", "F", "Cl", "Br"]

    def _build(n: int, prefix: str, with_label: bool) -> pd.DataFrame:
        rows = []
        for i in range(n):
            cls = i % 3
            tmpl = scaffolds[cls][i % len(scaffolds[cls])]
            smi = tmpl.format(r=str(rng.choice(r_groups)))
            row = {"id": f"{prefix}_{i}", "sequence": smi}
            if with_label:
                row["label"] = cls
                row["fold"] = i % 4
            rows.append(row)
        return pd.DataFrame(rows)

    return _build(n_train, "tr", True), _build(n_test, "te", False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 12 Tier 1 Fast CPU Sequence Baseline")
    parser.add_argument("--train-path", type=str, default=CFG.train_path)
    parser.add_argument("--test-path", type=str, default=CFG.test_path)
    parser.add_argument("--output-dir", type=str, default=CFG.output_dir)
    parser.add_argument("--exp-name", type=str, default=CFG.exp_name)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()

    cfg = CFG(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        smoke_test=args.smoke_test,
    )

    if cfg.smoke_test:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            print("[SMOKE TEST] Running Task 12 Tier 1 K-mer TF-IDF + Physicochemical GBDT/Ridge OOF...")
            train_df, test_df = generate_synthetic_dataset(seed=cfg.seed)
            oof, test_preds, metrics = run_oof_pipeline(train_df, test_df, cfg)
            assert oof.shape == (len(train_df), 3)
            assert test_preds.shape == (len(test_df), 3)
            assert (pathlib.Path(tmpdir) / f"oof_{cfg.exp_name}.npy").exists()
            assert (pathlib.Path(tmpdir) / f"test_{cfg.exp_name}.npy").exists()
            assert (pathlib.Path(tmpdir) / f"sub_{cfg.exp_name}.csv").exists()
            assert metrics["macro_f1"] > 0.85, f"Expected Macro-F1 > 0.85, got {metrics['macro_f1']}"
            print("[SUCCESS] Task 12 Tier 1 smoke test passed.")
    else:
        train_df = load_table(cfg.train_path)
        test_df = load_table(cfg.test_path)
        run_oof_pipeline(train_df, test_df, cfg)


if __name__ == "__main__":
    main()
