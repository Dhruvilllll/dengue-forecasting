"""Data cleaning, missing-value imputation and feature scaling.

Design decisions (each is deliberate and mirrors the R workflow's philosophy):

1. **Per-city, chronological processing.** Cleaning happens inside each city's
   own time-ordered frame so that no information ever crosses city boundaries or
   flows backward in time.

2. **Redundant column removal.** ``reanalysis_sat_precip_amt_mm`` is byte-for-byte
   identical to ``precipitation_amt_mm`` in the DengAI data, so it is dropped —
   keeping it would double-count one signal and inflate its apparent importance.

3. **Past-only imputation (forward-fill then back-fill).** Weather is temporally
   smooth, so the best estimate for a missing week is the most recent observed
   week (``ffill``). A single trailing ``bfill`` fills the handful of *leading*
   gaps that have no past to carry forward. This is causal: a value is never
   imputed from the future during the bulk of the series.

4. **Scaling fit on TRAIN only.** ``StandardScaler`` statistics are learned from
   the training portion exclusively and then applied to the hold-out/test rows,
   so no test-set information leaks into the transform.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean a single city's frame: drop redundancies, impute, sanity-check."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.target = cfg.data.target
        self.date_col = cfg.data.date_col
        self.drop_redundant = list(cfg.data.drop_redundant)

    def feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the weather/climate predictor columns (exclude keys+target)."""
        non_features = set(self.cfg.data.key_cols) | {self.target}
        return [c for c in df.columns if c not in non_features]

    def clean(self, df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        """Return a cleaned copy of ``df`` for one city.

        Parameters
        ----------
        df : DataFrame
            A single city's chronologically-ordered frame.
        is_train : bool
            Whether the frame carries the target column (affects logging only).
        """
        df = df.copy()

        # 1) Drop known-redundant columns if present.
        present = [c for c in self.drop_redundant if c in df.columns]
        if present:
            df = df.drop(columns=present)
            logger.info("Dropped redundant columns: %s", present)

        feat_cols = self.feature_columns(df)

        # 2) Report missingness before imputing (useful for the report/plots).
        missing_before = df[feat_cols].isna().sum()
        total_missing = int(missing_before.sum())
        if total_missing:
            worst = missing_before[missing_before > 0].sort_values(ascending=False)
            logger.info(
                "Missing feature cells before impute: %d (worst: %s)",
                total_missing,
                ", ".join(f"{k}={v}" for k, v in worst.head(3).items()),
            )

        # 3) Past-only imputation: forward-fill, then back-fill leading gaps.
        df[feat_cols] = df[feat_cols].ffill().bfill()

        remaining = int(df[feat_cols].isna().sum().sum())
        logger.info("Missing feature cells after impute: %d", remaining)

        return df


class FeatureScaler:
    """Thin, leakage-safe wrapper around :class:`~sklearn.preprocessing.StandardScaler`.

    Fit on the training matrix only; ``transform`` is then reused for the
    hold-out and test matrices. Column order is preserved so downstream feature
    names stay aligned.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.columns_ = list(X.columns)
        self.scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("FeatureScaler.transform called before fit().")
        X = X[self.columns_]  # enforce identical column order
        arr = self.scaler.transform(X.values)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def missing_value_summary(frames: dict[str, pd.DataFrame],
                          feature_cols: List[str]) -> pd.DataFrame:
    """Build a tidy % missing table (feature x group) for the diagnostics plot.

    Parameters
    ----------
    frames : dict[str, DataFrame]
        Group label (e.g. ``"sj_train"``) -> raw (un-imputed) frame.
    feature_cols : list[str]
        Columns to report on.

    Returns
    -------
    DataFrame
        Index = feature name, columns = group labels, values = percent missing.
    """
    out = {}
    for label, df in frames.items():
        cols = [c for c in feature_cols if c in df.columns]
        out[label] = (df[cols].isna().mean() * 100.0)
    table = pd.DataFrame(out)
    # Order worst-first by the maximum missingness across groups.
    table = table.reindex(table.max(axis=1).sort_values(ascending=False).index)
    return table
