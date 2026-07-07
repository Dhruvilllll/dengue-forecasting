"""Leakage guards and the target transform round-trip."""

from __future__ import annotations

import numpy as np

from dengue_forecast.feature_engineering import (
    inverse_transform_target,
    transform_target,
)


def test_holdout_is_strictly_after_train(pipeline, iq_dataset, iq_stack):
    """The temporal split must place every hold-out week AFTER every train week."""
    split = iq_stack["split"]
    train_dates = iq_dataset.dates_train.iloc[:split]
    holdout_dates = iq_stack["dates_ho"]
    assert train_dates.max() < holdout_dates.min()


def test_selection_and_scaling_are_column_aligned(cfg, iq_stack):
    selected = iq_stack["selected_features"]
    # SelectKBest keeps exactly k features (213 engineered > k, so exactly k).
    assert len(selected) == cfg.feature_selection.k
    assert iq_stack["X_tr"].shape[1] == cfg.feature_selection.k
    # Scaler columns match the selected features, in order.
    assert list(iq_stack["scaler"].columns_) == list(selected)
    # Train and hold-out matrices share identical columns.
    assert list(iq_stack["X_tr"].columns) == list(iq_stack["X_ho"].columns)


def test_scaler_fit_on_train_only(iq_stack):
    """The scaler's mean should reflect the TRAIN portion, not the hold-out."""
    scaler = iq_stack["scaler"].scaler
    # Standardized train features are ~zero-mean by construction.
    assert np.allclose(iq_stack["X_tr"].mean().values, 0.0, atol=1e-6)


def test_target_transform_roundtrips():
    y = np.array([0.0, 1.0, 5.0, 50.0, 400.0])
    back = inverse_transform_target(transform_target(y, "log1p"), "log1p")
    assert np.allclose(back, y, atol=1e-6)


def test_inverse_transform_clips_negative_counts():
    # A very negative log prediction must not yield a negative case count.
    out = inverse_transform_target(np.array([-10.0, -1.0, 2.0]), "log1p")
    assert (out >= 0).all()
