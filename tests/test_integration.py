"""Fast end-to-end checks through the real pipeline internals.

These avoid the expensive full tuning sweep: an untuned linear model is driven
through the actual split/select/scale/evaluate stack, and the tuner is exercised
on tiny synthetic data so both the Grid and Randomized search paths are covered.
"""

from __future__ import annotations

import numpy as np

from dengue_forecast.config import load_config
from dengue_forecast.models import ModelFactory
from dengue_forecast.tuning import HyperparameterTuner


def test_linear_model_end_to_end(pipeline, iq_stack):
    """Train an untuned model through the real stack; sanity-check the result."""
    result = pipeline._train_one("linear_regression", iq_stack, "iq")
    # Metrics are finite and on the original case scale.
    assert np.isfinite([result.mae, result.rmse, result.r2]).all()
    assert result.mae >= 0 and result.rmse >= 0
    # Predictions are valid case counts (non-negative) and aligned to hold-out.
    assert (result.y_pred >= 0).all()
    assert len(result.y_pred) == len(result.y_true) == len(iq_stack["dates_ho"])
    assert not result.tuned  # linear models are never tuned


def test_tuned_model_end_to_end(pipeline, iq_stack):
    """Drive a TUNED model through the real pipeline (guards the tuning path)."""
    result = pipeline._train_one("decision_tree", iq_stack, "iq")
    assert result.tuned
    assert result.best_params  # a tuned model records its winning params
    assert np.isfinite([result.mae, result.rmse, result.r2]).all()
    assert (result.y_pred >= 0).all()
    # Tree models expose feature importances for the plots/report.
    assert result.feature_importance is not None
    assert len(result.feature_importance) == len(iq_stack["selected_features"])


def test_grid_search_path_returns_fitted_tree():
    cfg = load_config()
    tuner = HyperparameterTuner(cfg, ModelFactory(cfg))
    rng = np.random.RandomState(0)
    X = rng.rand(120, 6)
    y = np.log1p(rng.poisson(8, 120).astype(float))
    res = tuner.tune("decision_tree", X, y)  # GridSearchCV
    assert res.search_type == "grid"
    assert hasattr(res.best_estimator, "predict")
    assert res.best_estimator.predict(X[:3]).shape == (3,)


def test_randomized_search_path_runs():
    cfg = load_config()
    setattr(cfg.tuning, "n_iter", 4)  # keep the synthetic search quick
    tuner = HyperparameterTuner(cfg, ModelFactory(cfg))
    rng = np.random.RandomState(1)
    X = rng.rand(120, 6)
    y = np.log1p(rng.poisson(8, 120).astype(float))
    res = tuner.tune("random_forest", X, y)  # RandomizedSearchCV
    assert res.search_type == "random"
    assert res.best_cv_score >= 0
