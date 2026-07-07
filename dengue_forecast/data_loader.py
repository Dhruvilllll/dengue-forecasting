"""Raw data loading and per-city separation.

Responsibilities
----------------
* Read the three DengAI CSVs (train features, train labels, test features).
* Merge features with labels on the composite key ``(city, year, weekofyear)``.
* Parse ``week_start_date`` to a real datetime and sort chronologically.
* Split each frame into the two independently-modeled cities.

No cleaning, imputation, or feature engineering happens here — this module only
*acquires* and *organises* the raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class RawData:
    """Container for the loaded, merged, per-city raw frames.

    Attributes
    ----------
    train : dict[str, pandas.DataFrame]
        City code -> merged training frame (features + ``total_cases``).
    test : dict[str, pandas.DataFrame]
        City code -> test features frame (no target).
    """

    train: Dict[str, pd.DataFrame]
    test: Dict[str, pd.DataFrame]


class DataLoader:
    """Load DengAI CSVs and hand back tidy per-city frames."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.date_col = cfg.data.date_col
        self.cities = cfg.data.cities

    def _read_csv(self, path: str) -> pd.DataFrame:
        logger.info("Reading %s", path)
        return pd.read_csv(path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column and sort each city chronologically."""
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        # Chronological order within each city; keeps the two series separable.
        df = df.sort_values(["city", self.date_col]).reset_index(drop=True)
        return df

    def load(self) -> RawData:
        """Load and merge everything, returning a :class:`RawData` bundle."""
        feats = self._read_csv(self.cfg.paths.features_train)
        labels = self._read_csv(self.cfg.paths.labels_train)
        test = self._read_csv(self.cfg.paths.features_test)

        # Merge features + labels on the natural key. Left-join keeps every
        # feature row; the DengAI files are already row-aligned so no rows drop.
        merged = feats.merge(
            labels,
            on=["city", "year", "weekofyear"],
            how="left",
            validate="one_to_one",
        )
        logger.info(
            "Merged training frame: %d rows, %d cols (target present: %s)",
            merged.shape[0],
            merged.shape[1],
            self.cfg.data.target in merged.columns,
        )

        merged = self._prepare(merged)
        test = self._prepare(test)

        train_by_city = {c: merged[merged.city == c].reset_index(drop=True)
                         for c in self.cities}
        test_by_city = {c: test[test.city == c].reset_index(drop=True)
                        for c in self.cities}

        for c in self.cities:
            logger.info(
                "City '%s': train=%d weeks (%s -> %s), test=%d weeks",
                c,
                len(train_by_city[c]),
                train_by_city[c][self.date_col].min().date(),
                train_by_city[c][self.date_col].max().date(),
                len(test_by_city[c]),
            )

        return RawData(train=train_by_city, test=test_by_city)
