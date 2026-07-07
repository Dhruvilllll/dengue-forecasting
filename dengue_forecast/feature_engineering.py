"""Temporal feature engineering, target transformation and feature selection.

The biology of dengue transmission introduces a multi-week delay between a
weather event and the resulting case count (mosquito breeding -> larval
maturation -> biting -> viral incubation, a ~3-6 week chain). We encode this
with three complementary feature families, all built **causally** (only ever
looking backward in time):

* **Lag features** — each weather variable shifted back by 1/2/3/4 weeks.
* **Rolling statistics** — trailing means and std over 4/8/12-week windows,
  which denoise weather (mosquito populations react to *sustained* conditions,
  not one-week blips).
* **Seasonality** — cyclic ``sin``/``cos`` encodings of the week-of-year and
  month. Calendar position is known for any future date, so this is *not*
  leakage and captures dengue's strong annual cycle.

Lags/rolling are constructed on the train and test series **joined in date
order**, so the first test weeks legitimately inherit their history from the
tail of train — a lag only looks backward, so there is no leakage. Rows whose
lags are undefined (the first few training weeks) are dropped from *train* only;
every test week is retained because all of them must remain predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class CityDataset:
    """Model-ready matrices for a single city (features engineered, unscaled).

    Attributes
    ----------
    X_train, X_test : DataFrame
        Engineered feature matrices (numeric only).
    y_train : Series
        Raw ``total_cases`` aligned to ``X_train``.
    dates_train, dates_test : Series
        ``week_start_date`` aligned to each matrix (for plotting/reporting).
    feature_names : list[str]
        The engineered feature columns.
    """

    city: str
    X_train: pd.DataFrame
    y_train: pd.Series
    dates_train: pd.Series
    X_test: pd.DataFrame
    dates_test: pd.Series
    feature_names: List[str]
    # Autoregressive (past-case) feature columns, if AR mode is enabled. These
    # are populated with real past cases on the train/hold-out rows, but are
    # NaN placeholders on the test rows (filled recursively at forecast time).
    case_lag_cols: List[str] = field(default_factory=list)


class FeatureEngineer:
    """Builds lag / rolling / seasonal features for one city at a time."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fe = cfg.feature_engineering
        self.target = cfg.data.target
        self.date_col = cfg.data.date_col
        self.key_cols = list(cfg.data.key_cols)
        # Autoregressive (nowcast) settings — default OFF.
        self.autoregressive = bool(getattr(self.fe, "autoregressive", False))
        self.case_lags = list(getattr(self.fe, "autoregressive_case_lags", []))
        self.case_rolling = list(getattr(self.fe, "autoregressive_case_rolling", []))

    # ------------------------------------------------------------------ #
    # Individual feature families
    # ------------------------------------------------------------------ #
    def _base_features(self, df: pd.DataFrame) -> List[str]:
        non_features = set(self.key_cols) | {self.target}
        return [c for c in df.columns if c not in non_features]

    def _add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclic encodings of week-of-year and month."""
        if not self.fe.add_seasonality:
            return df
        woy = df["weekofyear"].astype(float)
        df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
        df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
        month = df[self.date_col].dt.month.astype(float)
        df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        return df

    def _add_lags_and_rolls(self, df: pd.DataFrame,
                            base_cols: List[str]) -> pd.DataFrame:
        """Add lag and trailing rolling mean/std features for ``base_cols``.

        ``df`` must already be sorted by date (train tail followed by test).
        """
        new_cols = {}
        for col in base_cols:
            s = df[col]
            for lag in self.fe.lag_weeks:
                new_cols[f"{col}_lag_{lag}"] = s.shift(lag)
            for win in self.fe.rolling_windows:
                # shift(1) makes the window strictly trailing (excludes the
                # current week) -> no target-adjacent leakage for that week.
                roll = s.shift(1).rolling(window=win, min_periods=max(2, win // 2))
                new_cols[f"{col}_rollmean_{win}"] = roll.mean()
                new_cols[f"{col}_rollstd_{win}"] = roll.std()
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def build(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
              city: str) -> CityDataset:
        """Engineer features for one city and return a :class:`CityDataset`.

        Lags/rolls are built on the concatenated (train -> test) timeline so the
        earliest test rows inherit their history from the training tail.
        """
        train_df = train_df.sort_values(self.date_col).reset_index(drop=True)
        test_df = test_df.sort_values(self.date_col).reset_index(drop=True)

        base_cols = self._base_features(train_df)

        # Tag origin so we can split back apart after joint feature construction.
        tr = train_df.copy()
        te = test_df.copy()
        tr["_split"] = "train"
        te["_split"] = "test"
        # Ensure the target column exists on test for a clean concat.
        if self.target not in te.columns:
            te[self.target] = np.nan

        combined = pd.concat([tr, te], ignore_index=True)
        combined = combined.sort_values(self.date_col).reset_index(drop=True)

        combined = self._add_seasonality(combined)
        combined = self._add_lags_and_rolls(combined, base_cols)

        feature_names = [
            c for c in combined.columns
            if c not in set(self.key_cols) | {self.target, "_split"}
        ]

        # Split back apart.
        tr_out = combined[combined["_split"] == "train"].copy()
        te_out = combined[combined["_split"] == "test"].copy()

        # Drop *train* rows with undefined weather lag/rolling values (the
        # earliest weeks). Test rows are all retained — they inherit valid
        # history from the training tail.
        before = len(tr_out)
        tr_out = tr_out.dropna(subset=feature_names)
        logger.info(
            "[%s] dropped %d leading train weeks with undefined lags "
            "(%d -> %d)", city, before - len(tr_out), before, len(tr_out),
        )

        # Any residual NaNs in test weather features (rare, from rolling std
        # with few points) are back/forward filled within the test block.
        te_out[feature_names] = te_out[feature_names].ffill().bfill()

        # --- Autoregressive (past-case) features, if enabled -------------- #
        # These encode recent weekly case counts. On train/hold-out rows they
        # hold REAL past cases (a legitimate 1-week-ahead nowcast); on test rows
        # they are NaN placeholders, filled recursively during forecasting.
        case_lag_cols: List[str] = []
        if self.autoregressive and (self.case_lags or self.case_rolling):
            tr_out = tr_out.reset_index(drop=True)
            cases = tr_out[self.target].astype(float)
            # Build every case feature into one frame, then concat once (avoids
            # per-insert DataFrame fragmentation).
            new_train = {}
            for lag in self.case_lags:
                new_train[f"cases_lag_{lag}"] = cases.shift(lag)
            for win in self.case_rolling:
                new_train[f"cases_rollmean_{win}"] = (
                    cases.shift(1).rolling(win, min_periods=2).mean()
                )
            case_lag_cols = list(new_train)
            tr_out = pd.concat([tr_out, pd.DataFrame(new_train)], axis=1)
            before_ar = len(tr_out)
            tr_out = tr_out.dropna(subset=case_lag_cols)
            logger.info(
                "[%s] autoregressive ON: +%d case features, dropped %d more "
                "leading weeks (%d -> %d)", city, len(case_lag_cols),
                before_ar - len(tr_out), before_ar, len(tr_out),
            )
            # NaN placeholders on test rows (filled recursively at forecast time).
            placeholders = pd.DataFrame(
                np.nan, index=te_out.index, columns=case_lag_cols
            )
            te_out = pd.concat([te_out, placeholders], axis=1)

        all_features = feature_names + case_lag_cols

        ds = CityDataset(
            city=city,
            X_train=tr_out[all_features].reset_index(drop=True),
            y_train=tr_out[self.target].reset_index(drop=True).astype(float),
            dates_train=tr_out[self.date_col].reset_index(drop=True),
            X_test=te_out[all_features].reset_index(drop=True),
            dates_test=te_out[self.date_col].reset_index(drop=True),
            feature_names=all_features,
            case_lag_cols=case_lag_cols,
        )
        logger.info(
            "[%s] engineered %d features (train=%d, test=%d)",
            city, len(all_features), len(ds.X_train), len(ds.X_test),
        )
        return ds


# ---------------------------------------------------------------------- #
# Target transform helpers
# ---------------------------------------------------------------------- #
def transform_target(y: np.ndarray, method: str) -> np.ndarray:
    """Forward target transform. Currently supports ``log1p`` and ``none``."""
    if method == "log1p":
        return np.log1p(y)
    return np.asarray(y, dtype=float)


def inverse_transform_target(y: np.ndarray, method: str) -> np.ndarray:
    """Invert :func:`transform_target` and clip negatives (counts are >= 0)."""
    if method == "log1p":
        y = np.expm1(y)
    return np.clip(np.asarray(y, dtype=float), 0.0, None)


# ---------------------------------------------------------------------- #
# Feature selection (filter method: mutual information)
# ---------------------------------------------------------------------- #
class FeatureSelector:
    """Select the top-``k`` features by mutual information with the target.

    Mutual information is preferred over Pearson correlation because
    climate -> mosquito relationships are non-linear and threshold-like; MI
    captures *any* dependency and is scale-free. It is a **filter** method, so
    the identical feature set is handed to every model, making the comparison a
    fair test of the *estimator* rather than of different inputs. Fit on the
    training portion only.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.enabled = cfg.feature_selection.enabled
        self.k = cfg.feature_selection.k
        self.seed = cfg.project.random_seed
        self.selector_: SelectKBest | None = None
        self.selected_: List[str] | None = None
        self.scores_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y_transformed: np.ndarray) -> "FeatureSelector":
        if not self.enabled:
            self.selected_ = list(X.columns)
            return self
        k = min(self.k, X.shape[1])
        score_func = lambda X_, y_: mutual_info_regression(
            X_, y_, random_state=self.seed
        )
        self.selector_ = SelectKBest(score_func=score_func, k=k)
        self.selector_.fit(X.values, y_transformed)
        mask = self.selector_.get_support()
        self.selected_ = [c for c, keep in zip(X.columns, mask) if keep]
        self.scores_ = pd.Series(
            self.selector_.scores_, index=X.columns
        ).sort_values(ascending=False)
        logger.info("Selected %d/%d features by mutual information",
                    len(self.selected_), X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_ is None:
            raise RuntimeError("FeatureSelector.transform called before fit().")
        return X[self.selected_]
