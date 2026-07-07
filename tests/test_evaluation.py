"""Evaluation metrics, ranking and combined (pooled) scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_forecast.evaluation import (
    ModelResult,
    combined_scores,
    compute_metrics,
    rank_models,
    results_to_frame,
)


def test_perfect_prediction_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"] == 1.0


def test_known_metric_values():
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([1.0, -1.0, 7.0])
    m = compute_metrics(y_true, y_pred)
    assert np.isclose(m["MAE"], (1 + 1 + 3) / 3)
    assert np.isclose(m["RMSE"], np.sqrt((1 + 1 + 9) / 3))


def _mk(city, key, mae, r2=0.0):
    n = 10
    return ModelResult(
        city=city, model_key=key, model_name=key.title(),
        mae=mae, rmse=mae * 1.5, r2=r2,
        y_true=np.arange(n, dtype=float),
        y_pred=np.arange(n, dtype=float) + mae,
        dates=pd.Series(pd.date_range("2020-01-01", periods=n, freq="W")),
    )


def test_rank_orders_best_first_per_city():
    results = [_mk("iq", "a", 9.0), _mk("iq", "b", 3.0), _mk("iq", "c", 6.0)]
    ranked = rank_models(results_to_frame(results), by="MAE")
    ordered = ranked.sort_values("Rank")["MAE"].tolist()
    assert ordered == [3.0, 6.0, 9.0]
    assert ranked.loc[ranked["Rank"] == 1, "Model"].iloc[0] == "B"


def test_combined_scores_pool_both_cities():
    results = [_mk("sj", "a", 4.0), _mk("iq", "a", 2.0),
               _mk("sj", "b", 8.0), _mk("iq", "b", 6.0)]
    combined = combined_scores(results)
    # One pooled row per model, ranked by MAE, best first.
    assert set(combined["Model"]) == {"A", "B"}
    assert combined.iloc[0]["Model"] == "A"
    assert combined.iloc[0]["Rank"] == 1
