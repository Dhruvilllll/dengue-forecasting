"""Data loading, cleaning and feature engineering — shape and integrity checks."""

from __future__ import annotations

import numpy as np

from dengue_forecast.config import load_config
from dengue_forecast.data_loader import DataLoader
from dengue_forecast.feature_engineering import FeatureEngineer
from dengue_forecast.preprocessing import DataCleaner


def test_raw_row_counts_and_order(raw, cfg):
    assert len(raw.train["sj"]) == 936
    assert len(raw.train["iq"]) == 520
    assert len(raw.test["sj"]) == 260
    assert len(raw.test["iq"]) == 156
    # Chronological order within each city.
    for city in cfg.data.cities:
        dates = raw.train[city][cfg.data.date_col]
        assert dates.is_monotonic_increasing


def test_cleaning_removes_redundant_and_all_nans(raw, cfg):
    cleaner = DataCleaner(cfg)
    cleaned = cleaner.clean(raw.train["sj"], is_train=True)
    # Redundant duplicate column dropped.
    for col in cfg.data.drop_redundant:
        assert col not in cleaned.columns
    # No missing feature cells remain after imputation.
    feat_cols = cleaner.feature_columns(cleaned)
    assert cleaned[feat_cols].isna().sum().sum() == 0


def test_engineered_matrices_have_no_nans(iq_dataset):
    assert not iq_dataset.X_train.isna().any().any()
    assert not iq_dataset.X_test.isna().any().any()


def test_all_test_weeks_retained(iq_dataset):
    # Every test week must remain predictable (none dropped).
    assert len(iq_dataset.X_test) == 156


def test_seasonality_and_lag_features_present(iq_dataset):
    names = set(iq_dataset.feature_names)
    for col in ["woy_sin", "woy_cos", "month_sin", "month_cos"]:
        assert col in names
    # At least one lag and one rolling feature exist.
    assert any(c.endswith("_lag_4") for c in names)
    assert any("_rollmean_" in c for c in names)


def test_autoregressive_mode_adds_case_features():
    """AR mode adds past-case columns (filled on train, NaN placeholders on test)."""
    cfg = load_config()  # fresh config to avoid mutating the session fixture
    setattr(cfg.feature_engineering, "autoregressive", True)
    raw = DataLoader(cfg).load()
    cleaner, eng = DataCleaner(cfg), FeatureEngineer(cfg)
    tr = cleaner.clean(raw.train["iq"], is_train=True)
    te = cleaner.clean(raw.test["iq"], is_train=False)
    ds = eng.build(tr, te, "iq")
    assert ds.case_lag_cols, "AR mode must create case-lag columns"
    # Train rows carry REAL past cases; test rows are NaN placeholders.
    assert not ds.X_train[ds.case_lag_cols].isna().any().any()
    assert ds.X_test[ds.case_lag_cols].isna().all().all()
