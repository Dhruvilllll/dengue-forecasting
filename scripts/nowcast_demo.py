#!/usr/bin/env python
"""Autoregressive Nowcast — the combined-signal model that gets our best results.

This is the presentation-ready demonstration of the headline result: feeding the
models BOTH families of signal at once bridges "environmental theory" and the
"on-the-ground outbreak reality" and roughly halves the prediction error.

  * Climate / weather signal  — rainfall, temperature, vegetation (+ their lags,
    rolling means and seasonality). The long-term conditions for mosquito
    breeding: "is it a high-risk season?"
  * Recent-case signal        — lagged weekly dengue counts. The immediate
    transmission reality: "has the outbreak already started?"

For each city it trains the two tuned tree models the presentation highlights
(Random Forest, XGBoost) on:
    (A) climate only            -> the weather-only baseline
    (B) climate + recent cases  -> the autoregressive nowcast
evaluates both on the same chronological hold-out, prints the before/after
comparison, and saves the nowcast predictions.

Run:
    python scripts/nowcast_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dengue_forecast.config import load_config  # noqa: E402
from dengue_forecast.data_loader import DataLoader  # noqa: E402
from dengue_forecast.evaluation import compute_metrics  # noqa: E402
from dengue_forecast.feature_engineering import (  # noqa: E402
    FeatureEngineer,
    FeatureSelector,
    inverse_transform_target,
    transform_target,
)
from dengue_forecast.models import ModelFactory  # noqa: E402
from dengue_forecast.preprocessing import DataCleaner, FeatureScaler  # noqa: E402
from dengue_forecast.tuning import HyperparameterTuner  # noqa: E402
from dengue_forecast.utils import ensure_dirs, get_logger, set_global_seed  # noqa: E402

logger = get_logger("nowcast_demo")

# The two tree models the presentation focuses on.
DEMO_MODELS = ["random_forest", "xgboost"]


def build_dataset(cfg, raw, city: str, use_cases: bool):
    """Build the engineered dataset for a city, with or without case features."""
    # Toggle the autoregressive (recent-case) features on/off before building.
    setattr(cfg.feature_engineering, "autoregressive", use_cases)
    cleaner, engineer = DataCleaner(cfg), FeatureEngineer(cfg)
    train_clean = cleaner.clean(raw.train[city], is_train=True)
    test_clean = cleaner.clean(raw.test[city], is_train=False)
    return engineer.build(train_clean, test_clean, city)


def prepare_split(cfg, ds):
    """Chronological split + log-target + MI selection + scaling (train-fit)."""
    n = len(ds.X_train)
    split = int(np.floor(n * cfg.split.train_frac))
    X_tr_raw, X_ho_raw = ds.X_train.iloc[:split], ds.X_train.iloc[split:]
    y_tr_raw, y_ho_raw = ds.y_train.iloc[:split], ds.y_train.iloc[split:]

    y_tr_log = transform_target(y_tr_raw.values, cfg.feature_engineering.target_transform)
    selector = FeatureSelector(cfg).fit(X_tr_raw, y_tr_log)
    scaler = FeatureScaler().fit(selector.transform(X_tr_raw))
    X_tr = scaler.transform(selector.transform(X_tr_raw))
    X_ho = scaler.transform(selector.transform(X_ho_raw))
    dates_ho = ds.dates_train.iloc[split:].reset_index(drop=True)
    return X_tr, y_tr_log, X_ho, y_ho_raw.values.astype(float), dates_ho


def train_eval(cfg, factory, tuner, key, X_tr, y_tr_log, X_ho, y_ho):
    """Tune the model on the training fold, evaluate on the hold-out (real scale)."""
    est = tuner.tune(key, X_tr.values, y_tr_log).best_estimator
    y_pred = inverse_transform_target(
        est.predict(X_ho.values), cfg.feature_engineering.target_transform
    )
    return compute_metrics(y_ho, y_pred), y_pred


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg.project.random_seed)
    raw = DataLoader(cfg).load()
    factory = ModelFactory(cfg)
    tuner = HyperparameterTuner(cfg, factory)

    out_dir = Path(cfg.paths.output_dir) / "nowcast_demo"
    ensure_dirs([out_dir])

    rows = []
    for city in cfg.data.cities:
        logger.info("===== %s =====", city.upper())
        for label, use_cases in [("Climate only", False),
                                 ("Climate + Cases", True)]:
            ds = build_dataset(cfg, raw, city, use_cases)
            X_tr, y_tr_log, X_ho, y_ho, dates_ho = prepare_split(cfg, ds)
            for key in DEMO_MODELS:
                metrics, y_pred = train_eval(
                    cfg, factory, tuner, key, X_tr, y_tr_log, X_ho, y_ho
                )
                rows.append({
                    "City": city, "Signal": label,
                    "Model": factory.display_name(key),
                    "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                })
                logger.info("  %-16s %-14s MAE=%.2f RMSE=%.2f R2=%.3f",
                            label, factory.display_name(key),
                            metrics["MAE"], metrics["RMSE"], metrics["R2"])
                # Save the nowcast (combined-signal) hold-out predictions.
                if use_cases:
                    pd.DataFrame({
                        "week_start_date": dates_ho.values,
                        "actual": y_ho,
                        "predicted": np.rint(y_pred).astype(int),
                    }).to_csv(out_dir / f"{city}_{key}_nowcast_holdout.csv",
                              index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "signal_comparison.csv", index=False)

    # ---- Headline summary -------------------------------------------------- #
    print("\n" + "=" * 68)
    print("AUTOREGRESSIVE NOWCAST — climate-only vs climate+cases (hold-out RMSE)")
    print("=" * 68)
    pivot = comparison.pivot_table(index=["City", "Model"], columns="Signal",
                                   values="RMSE")
    pivot["Improvement"] = (pivot["Climate only"] - pivot["Climate + Cases"])
    pivot["% cut"] = 100 * pivot["Improvement"] / pivot["Climate only"]
    print(pivot.round(2).to_string())
    print("=" * 68)
    best_wo = comparison[comparison.Signal == "Climate only"]["RMSE"].mean()
    best_c = comparison[comparison.Signal == "Climate + Cases"]["RMSE"].mean()
    print(f"Average RMSE:  climate-only {best_wo:.1f}  ->  climate+cases "
          f"{best_c:.1f}   ({100*(best_wo-best_c)/best_wo:.0f}% lower error)")
    print(f"Predictions + comparison saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
