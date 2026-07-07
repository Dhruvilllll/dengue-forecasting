"""Cross-cutting utilities: logging, seeding, and small IO helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single stream handler.

    Idempotent: repeated calls with the same ``name`` do not stack handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness we rely on, for reproducibility.

    Covers Python's ``random``, NumPy, and the ``PYTHONHASHSEED`` env var.
    XGBoost/LightGBM/scikit-learn estimators additionally receive the same
    seed explicitly through the model factory.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(paths: Iterable[str | os.PathLike]) -> None:
    """Create every directory in ``paths`` (and parents) if missing."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared-error, computed in a version-agnostic way.

    (Avoids relying on the ``squared=`` kwarg of ``mean_squared_error`` which
    was removed in newer scikit-learn releases.)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
