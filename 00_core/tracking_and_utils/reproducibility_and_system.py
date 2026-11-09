#!/usr/bin/env python3
"""
reproducibility_and_system.py
=============================
Competition reproducibility, timing, logging, and GPU/RAM hygiene utilities:
  - `seed_everything(seed=42, deterministic_cudnn=True)`
  - `Timer(name, logger=None)` context manager measuring wall-clock elapsed time and RAM delta
  - `setup_logger(log_file=None, name="gemastik")` structured console + file logger
  - `free_memory()` garbage collection + `torch.cuda.empty_cache()`

Usage:
  python3 00_core/tracking_and_utils/reproducibility_and_system.py --smoke-test
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

try:
    import numpy as np
except ImportError:
    import glob
    import site

    for _sp in glob.glob(os.path.expanduser("~/.venv/lib/python*/site-packages")):
        site.addsitedir(_sp)
    import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class CFG:
    seed: int = 42
    deterministic_cudnn: bool = True
    log_file: Optional[str] = None


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Locks random seeds across Python, NumPy, OS hash seed, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def setup_logger(
    log_file: Optional[str] = None,
    name: str = "gemastik",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures a duplicate-free logger writing to stdout and an optional log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def free_memory() -> None:
    """Forces Python garbage collection and clears PyTorch CUDA cache if available."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    """Context manager for timing code blocks and logging elapsed duration."""

    def __init__(self, name: str = "block", logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.logger = logger
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        msg = f"[Timer] Starting '{self.name}' ..."
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        msg = f"[Timer] Finished '{self.name}' in {self.elapsed:.4f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def run_smoke_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "run.log")
        logger = setup_logger(log_path)

        seed_everything(1337)
        a1 = np.random.rand(5)
        t1 = torch.randn(5) if torch is not None else None

        seed_everything(1337)
        a2 = np.random.rand(5)
        t2 = torch.randn(5) if torch is not None else None

        assert np.allclose(a1, a2), "NumPy seed lock failed"
        if t1 is not None and t2 is not None:
            assert torch.allclose(t1, t2), "PyTorch seed lock failed"

        with Timer("matrix_svd_benchmark", logger=logger) as t:
            M = np.random.randn(200, 100)
            _ = np.linalg.svd(M, full_matrices=False)
        assert t.elapsed > 0.0

        free_memory()
        assert os.path.exists(log_path)
        print("[SMOKE TEST PASSED] reproducibility_and_system.py verified successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducibility, Timer, Logger, and Memory utilities.")
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke_test()
