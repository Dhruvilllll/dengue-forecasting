"""End-to-end orchestrator: raw CSVs -> cleaned -> features -> models -> report.

The :class:`DenguePipeline` runs the full workflow **independently per city**:

1. Load & merge raw data (features + labels), split per city.
2. Clean (drop redundancy, past-only imputation).
3. Engineer lag / rolling / seasonal features.
4. Temporal hold-out split (earliest 75% train, most-recent 25% test).
5. Log-transform the target; select top-k features (MI); standard-scale — all
   fit on the training portion only (leakage-safe).
6. Train the 8 permitted models (tuning the 4 tree models), evaluate on the
   hold-out with MAE / RMSE / R^2.
7. Render all figures, write metrics/ranking/report CSVs & text.
8. Refit each city's best model on 100% of train and forecast the 416-week
   test set into a DengAI-style submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import Config, load_config
from .data_loader import DataLoader, RawData
from .evaluation import (
    ModelResult,
    combined_scores,
    compute_metrics,
    format_comparison_report,
    rank_models,
    results_to_frame,
)
from .feature_engineering import (
    CityDataset,
    FeatureEngineer,
    FeatureSelector,
    inverse_transform_target,
    transform_target,
)
from .models import ModelFactory
from .preprocessing import DataCleaner, FeatureScaler, missing_value_summary
from .tuning import HyperparameterTuner
from .utils import ensure_dirs, get_logger, set_global_seed
from .visualization import Visualizer

logger = get_logger(__name__)


class DenguePipeline:
    """Coordinates every stage of the dengue forecasting workflow."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        set_global_seed(self.cfg.project.random_seed)

        self.loader = DataLoader(self.cfg)
        self.cleaner = DataCleaner(self.cfg)
        self.engineer = FeatureEngineer(self.cfg)
        self.factory = ModelFactory(self.cfg)
        self.tuner = HyperparameterTuner(self.cfg, self.factory)
        self.viz = Visualizer(self.cfg)

        self.target = self.cfg.data.target
        self.target_transform = self.cfg.feature_engineering.target_transform
        self.train_frac = self.cfg.split.train_frac
        self.cities = list(self.cfg.data.cities)

        for d in [self.cfg.paths.output_dir, self.cfg.paths.figures_dir,
                  self.cfg.paths.models_dir, self.cfg.paths.reports_dir,
                  self.cfg.paths.predictions_dir]:
            ensure_dirs([d])

        self.results: List[ModelResult] = []
        self._best_by_city: Dict[str, ModelResult] = {}

    # ================================================================== #
    # Stage helpers
    # ================================================================== #
    def _temporal_split(self, n: int) -> int:
        """Return the index that separates train (earliest) from hold-out."""
        return int(np.floor(n * self.train_frac))

    def _prepare_city(self, raw: RawData, city: str):
        """Clean + engineer features for one city; return dataset + cleaned frames."""
        train_clean = self.cleaner.clean(raw.train[city], is_train=True)
        test_clean = self.cleaner.clean(raw.test[city], is_train=False)
        ds = self.engineer.build(train_clean, test_clean, city)
        return ds, train_clean

    def _fit_transform_stack(self, ds: CityDataset):
        """Split, log-transform target, select features, scale — leakage-safe.

        Returns a dict with the scaled train/hold-out arrays and the fitted
        selector/scaler for later full-data refits and test forecasting.
        """
        n = len(ds.X_train)
        split = self._temporal_split(n)

        X_tr_raw, X_ho_raw = ds.X_train.iloc[:split], ds.X_train.iloc[split:]
        y_tr_raw, y_ho_raw = ds.y_train.iloc[:split], ds.y_train.iloc[split:]

        # Target transform (fit-free; just apply on train target).
        y_tr_t = transform_target(y_tr_raw.values, self.target_transform)

        # Feature selection — fit on TRAIN portion & transformed target.
        selector = FeatureSelector(self.cfg).fit(X_tr_raw, y_tr_t)
        X_tr_sel = selector.transform(X_tr_raw)
        X_ho_sel = selector.transform(X_ho_raw)

        # Scaling — fit on TRAIN portion only.
        scaler = FeatureScaler().fit(X_tr_sel)
        X_tr_s = scaler.transform(X_tr_sel)
        X_ho_s = scaler.transform(X_ho_sel)

        return {
            "split": split,
            "selector": selector,
            "scaler": scaler,
            "X_tr": X_tr_s, "X_ho": X_ho_s,
            "y_tr_t": y_tr_t,
            "y_tr_raw": y_tr_raw, "y_ho_raw": y_ho_raw,
            "dates_ho": ds.dates_train.iloc[split:].reset_index(drop=True),
            "selected_features": selector.selected_,
        }

    def _extract_importance(self, estimator, feature_names: List[str]) -> pd.Series | None:
        """Return a feature-importance Series for tree models, else ``None``."""
        if hasattr(estimator, "feature_importances_"):
            return pd.Series(estimator.feature_importances_, index=feature_names)
        return None

    def _train_one(self, key: str, stack: dict, city: str) -> ModelResult:
        """Train/tune a single model, evaluate on the hold-out, package result."""
        X_tr, y_tr_t = stack["X_tr"].values, stack["y_tr_t"]
        X_ho = stack["X_ho"].values
        features = stack["selected_features"]
        tuned = False
        best_params: dict = {}

        if self.factory.is_tunable(key):
            res = self.tuner.tune(key, X_tr, y_tr_t)
            estimator = res.best_estimator
            best_params = res.best_params
            tuned = True
        else:
            estimator = self.factory.create(key)
            estimator.fit(X_tr, y_tr_t)

        # Predict on hold-out and invert the target transform.
        y_pred_t = estimator.predict(X_ho)
        y_pred = inverse_transform_target(y_pred_t, self.target_transform)
        y_true = stack["y_ho_raw"].values.astype(float)

        metrics = compute_metrics(y_true, y_pred)
        importance = self._extract_importance(estimator, features)

        # Persist the trained hold-out model for reproducibility.
        model_dir = Path(self.cfg.paths.models_dir) / city
        ensure_dirs([model_dir])
        joblib.dump(estimator, model_dir / f"{key}.joblib")

        return ModelResult(
            city=city,
            model_key=key,
            model_name=self.factory.display_name(key),
            mae=metrics["MAE"], rmse=metrics["RMSE"], r2=metrics["R2"],
            y_true=y_true, y_pred=y_pred,
            dates=stack["dates_ho"],
            best_params=best_params,
            feature_importance=importance,
            tuned=tuned,
        )

    # ================================================================== #
    # Figure generation
    # ================================================================== #
    def _make_model_figures(self, r: ModelResult) -> None:
        self.viz.actual_vs_predicted(r)
        self.viz.residual_plot(r)
        self.viz.prediction_error_scatter(r)
        self.viz.residual_histogram(r)
        self.viz.feature_importance(r)      # no-op for linear models
        self.viz.model_dashboard(r)

    def _make_eda_figures(self, raw: RawData, prepared: dict) -> None:
        """Correlation, missing-value, target and MI feature-selection figures."""
        # Missing-value diagnostics from the RAW (un-imputed) frames.
        feat_cols = self.cleaner.feature_columns(
            raw.train[self.cities[0]]
        )
        frames = {}
        for c in self.cities:
            frames[f"{c}_train"] = raw.train[c]
            frames[f"{c}_test"] = raw.test[c]
        missing_table = missing_value_summary(frames, feat_cols)
        if missing_table.to_numpy().sum() > 0:
            self.viz.missing_values(missing_table)

        # Target overview (both cities).
        y_by_city, dates_by_city = {}, {}
        for c in self.cities:
            df = raw.train[c]
            y_by_city[c] = df[self.target]
            dates_by_city[c] = df[self.cfg.data.date_col]
        self.viz.target_overview(y_by_city, dates_by_city, self.target)

        # Per-city correlation heatmap (base features + target) and MI scores.
        for c in self.cities:
            clean = prepared[c]["train_clean"]
            base_cols = self.cleaner.feature_columns(clean) + [self.target]
            self.viz.correlation_heatmap(clean[base_cols], c, self.target)
            stack = prepared[c]["stack"]
            self.viz.feature_selection_scores(
                stack["selector"].scores_, stack["selected_features"], c,
            )

    # ================================================================== #
    # Test-set forecasting (refit on full training data)
    # ================================================================== #
    def _recursive_forecast(self, ds: CityDataset, estimator, selector,
                            scaler) -> np.ndarray:
        """Recursively forecast the test weeks in autoregressive mode.

        The test weeks have no known case history, so we walk them in date
        order: for week ``t`` we fill the case-lag features from the most recent
        *known* cases (actual training cases at first, then our own predictions),
        predict, append the prediction to the history, and continue. Weather
        features for each test week are already known.
        """
        eng = self.engineer
        known: List[float] = list(np.asarray(ds.y_train, dtype=float))
        X_test = ds.X_test.reset_index(drop=True).copy()
        preds: List[float] = []
        for i in range(len(X_test)):
            row = X_test.iloc[[i]].copy()
            for lag in eng.case_lags:
                row[f"cases_lag_{lag}"] = known[-lag] if len(known) >= lag else np.nan
            for win in eng.case_rolling:
                recent = known[-win:]
                row[f"cases_rollmean_{win}"] = (
                    float(np.mean(recent)) if len(recent) >= 2 else np.nan
                )
            X_row = scaler.transform(selector.transform(row))
            pred = inverse_transform_target(
                estimator.predict(X_row.values), self.target_transform
            )[0]
            preds.append(pred)
            known.append(pred)  # feed the prediction forward as next week's lag
        return np.asarray(preds)

    def _forecast_test(self, raw: RawData, prepared: dict) -> pd.DataFrame:
        """Refit each city's best model on ALL training weeks; forecast test."""
        rows = []
        for c in self.cities:
            ds: CityDataset = prepared[c]["ds"]
            best = self._best_by_city[c]

            # Re-fit selection + scaling on the FULL training set.
            y_full_t = transform_target(ds.y_train.values, self.target_transform)
            selector = FeatureSelector(self.cfg).fit(ds.X_train, y_full_t)
            X_full_sel = selector.transform(ds.X_train)
            scaler = FeatureScaler().fit(X_full_sel)
            X_full_s = scaler.transform(X_full_sel)

            estimator = (self.tuner.tune(best.model_key, X_full_s.values, y_full_t)
                         .best_estimator if self.factory.is_tunable(best.model_key)
                         else self.factory.create(best.model_key))
            if not self.factory.is_tunable(best.model_key):
                estimator.fit(X_full_s.values, y_full_t)

            if ds.case_lag_cols:
                # Autoregressive mode: past case counts are unknown for the
                # future test weeks, so forecast recursively — each prediction
                # is fed back as the input lag for the following week.
                preds = self._recursive_forecast(ds, estimator, selector, scaler)
            else:
                X_test_s = scaler.transform(selector.transform(ds.X_test))
                preds = inverse_transform_target(
                    estimator.predict(X_test_s.values), self.target_transform
                )
            preds = np.rint(preds).astype(int)

            test_meta = raw.test[c].loc[:, ["city", "year", "weekofyear"]].copy()
            test_meta = test_meta.reset_index(drop=True)
            test_meta["total_cases"] = preds
            rows.append(test_meta)
            logger.info("[%s] test forecast via best model '%s' (%d weeks)",
                        c, best.model_name, len(test_meta))

        submission = pd.concat(rows, ignore_index=True)
        out = Path(self.cfg.paths.predictions_dir) / "submission.csv"
        submission.to_csv(out, index=False)
        logger.info("Wrote submission -> %s", out)
        return submission

    # ================================================================== #
    # Main entry point
    # ================================================================== #
    def run(self) -> Dict[str, object]:
        """Execute the full pipeline and return a summary dict of artefacts."""
        logger.info("=== Dengue forecasting pipeline START ===")
        raw = self.loader.load()

        prepared: dict = {}
        # ---- Per-city preparation + modeling ---------------------------- #
        for c in self.cities:
            logger.info("----- Preparing city '%s' -----", c)
            ds, train_clean = self._prepare_city(raw, c)
            stack = self._fit_transform_stack(ds)
            prepared[c] = {"ds": ds, "train_clean": train_clean, "stack": stack}

            logger.info("----- Training models for '%s' -----", c)
            city_results: List[ModelResult] = []
            for key in self.factory.model_keys():
                r = self._train_one(key, stack, c)
                logger.info("[%s] %-20s MAE=%.3f RMSE=%.3f R2=%.3f",
                            c, r.model_name, r.mae, r.rmse, r.r2)
                self.results.append(r)
                city_results.append(r)
                self._make_model_figures(r)

            # Best model for this city by hold-out MAE.
            best = min(city_results, key=lambda x: x.mae)
            self._best_by_city[c] = best
            self.viz.all_models_timeline(c, city_results)

        # ---- EDA / diagnostic figures ---------------------------------- #
        self._make_eda_figures(raw, prepared)

        # ---- Aggregate evaluation -------------------------------------- #
        metrics = results_to_frame(self.results)
        ranked = rank_models(metrics, by="MAE")
        combined = combined_scores(self.results)
        self.viz.model_comparison_bars(metrics)

        # ---- Persist tables & reports ---------------------------------- #
        reports = Path(self.cfg.paths.reports_dir)
        metrics.to_csv(reports / "all_metrics.csv", index=False)
        ranked.to_csv(reports / "ranked_metrics.csv", index=False)
        combined.to_csv(reports / "combined_scores.csv", index=False)

        best_rows = pd.DataFrame([
            {"City": c, "Best Model": r.model_name, "MAE": r.mae,
             "RMSE": r.rmse, "R2": r.r2}
            for c, r in self._best_by_city.items()
        ])
        best_rows.to_csv(reports / "best_models.csv", index=False)

        report_txt = format_comparison_report(ranked, combined)
        (reports / "model_comparison_report.txt").write_text(report_txt,
                                                             encoding="utf-8")
        # Hold-out predictions per city/model.
        preds_dir = Path(self.cfg.paths.predictions_dir)
        for r in self.results:
            pd.DataFrame({
                "week_start_date": r.dates.values,
                "actual": r.y_true,
                "predicted": r.y_pred,
            }).to_csv(preds_dir / f"{r.city}_{r.model_key}_holdout.csv", index=False)

        # ---- Test-set forecast (best model per city) ------------------- #
        self._forecast_test(raw, prepared)

        # Machine-readable run summary.
        summary = {
            "best_by_city": {c: r.model_name for c, r in self._best_by_city.items()},
            "combined_best": combined.iloc[0]["Model"],
            "n_models": len(self.factory.model_keys()),
            "cities": self.cities,
        }
        (reports / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")

        logger.info("\n%s", report_txt)
        logger.info("=== Dengue forecasting pipeline DONE ===")
        return {"metrics": metrics, "ranked": ranked, "combined": combined,
                "report": report_txt, "summary": summary}
