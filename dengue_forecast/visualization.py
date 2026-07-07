"""Publication-quality visualization module (all figures rendered at 300 DPI).

Every figure uses a single, consistent style (fonts, palette, city colors) so
the whole set reads as one coherent report suitable for a paper or portfolio.

Figure catalogue
----------------
EDA / diagnostics (per city or global):
    * correlation heatmap
    * missing-value visualization (bar + heatmap)
    * target time-series & distribution (raw vs log1p)
    * mutual-information feature-selection scores

Per model x per city:
    * actual-vs-predicted line chart
    * residual plot (residuals over time)
    * prediction-error scatter (actual vs predicted)
    * residual distribution histogram
    * feature importance (tree-based models)
    * a combined 6-panel diagnostic dashboard

Comparison:
    * model comparison bar chart (MAE / RMSE / R^2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import Config  # noqa: E402
from .evaluation import ModelResult  # noqa: E402
from .utils import ensure_dirs, get_logger  # noqa: E402

logger = get_logger(__name__)


class Visualizer:
    """Renders and saves every figure in a single, consistent visual style."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.viz = cfg.viz
        self.dpi = int(self.viz.dpi)
        self.figsize = tuple(self.viz.figsize_default)
        self.colors = self.viz.colors.to_dict()
        self.fig_root = Path(cfg.paths.figures_dir)
        ensure_dirs([self.fig_root])
        self._apply_style()

    # ------------------------------------------------------------------ #
    # Styling / IO helpers
    # ------------------------------------------------------------------ #
    def _apply_style(self) -> None:
        """Apply a consistent global matplotlib/seaborn style."""
        try:
            plt.style.use(self.viz.style)
        except (OSError, ValueError):
            plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(self.viz.palette)
        plt.rcParams.update({
            "figure.dpi": 110,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })

    def _city_dir(self, city: str, subdir: str = "") -> Path:
        d = self.fig_root / city / subdir if subdir else self.fig_root / city
        ensure_dirs([d])
        return d

    def _save(self, fig: plt.Figure, path: Path) -> str:
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure %s", path)
        return str(path)

    def _city_color(self, city: str) -> str:
        return self.colors.get(city, self.colors["predicted"])

    # ================================================================== #
    # EDA / DIAGNOSTIC FIGURES
    # ================================================================== #
    def correlation_heatmap(self, df: pd.DataFrame, city: str,
                            target: str) -> str:
        """Heatmap of feature-feature and feature-target correlations."""
        numeric = df.select_dtypes(include=[np.number])
        # Keep the target adjacent for easy reading if present.
        corr = numeric.corr()
        n = corr.shape[0]
        size = max(8, min(0.5 * n + 3, 22))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, cbar_kws={"shrink": 0.6, "label": "Pearson r"},
            annot=n <= 16, fmt=".2f", annot_kws={"size": 7}, ax=ax,
        )
        ax.set_title(f"Correlation Heatmap — {city.upper()}", pad=14)
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
        return self._save(fig, self._city_dir(city, "eda") / "correlation_heatmap.png")

    def missing_values(self, missing_table: pd.DataFrame) -> str:
        """Two-panel missing-value visualization: bar (max %) + heatmap (% grid)."""
        fig, axes = plt.subplots(
            1, 2, figsize=(15, max(6, 0.32 * len(missing_table))),
            gridspec_kw={"width_ratios": [1, 1.1]},
        )
        # Panel 1 — worst missingness per feature (max across groups).
        worst = missing_table.max(axis=1).sort_values()
        colors = [self.colors["bad"] if v > 0 else self.colors["good"]
                  for v in worst.values]
        axes[0].barh(worst.index, worst.values, color=colors)
        axes[0].set_xlabel("Max missing across groups (%)")
        axes[0].set_title("Missingness by Feature")
        axes[0].tick_params(axis="y", labelsize=8)

        # Panel 2 — % missing per feature x group heatmap.
        sns.heatmap(
            missing_table, cmap="Reds", vmin=0, annot=True, fmt=".1f",
            linewidths=0.4, cbar_kws={"label": "% missing"},
            annot_kws={"size": 7}, ax=axes[1],
        )
        axes[1].set_title("% Missing per Feature x Group")
        axes[1].tick_params(axis="y", labelsize=8)
        fig.suptitle("Missing Value Diagnostics (before imputation)",
                     fontsize=15, fontweight="bold")
        return self._save(fig, self.fig_root / "missing_values.png")

    def target_overview(self, y_by_city: Dict[str, pd.Series],
                        dates_by_city: Dict[str, pd.Series],
                        target: str) -> str:
        """Cases-over-time and raw-vs-log distribution, one row per city."""
        cities = list(y_by_city.keys())
        fig, axes = plt.subplots(len(cities), 2, figsize=(14, 4 * len(cities)))
        if len(cities) == 1:
            axes = axes.reshape(1, 2)
        for i, city in enumerate(cities):
            y = np.asarray(y_by_city[city], dtype=float)
            dates = dates_by_city[city]
            color = self._city_color(city)
            # Time series
            axes[i, 0].plot(dates, y, color=color, lw=1.1)
            axes[i, 0].fill_between(dates, y, color=color, alpha=0.2)
            axes[i, 0].set_title(f"{city.upper()} — weekly {target} over time")
            axes[i, 0].set_ylabel(target)
            # Distributions raw vs log1p
            sns.histplot(y, bins=40, color=self.colors["bad"], alpha=0.5,
                         label="raw", ax=axes[i, 1], stat="density")
            sns.histplot(np.log1p(y), bins=40, color=self.colors["good"],
                         alpha=0.5, label="log1p", ax=axes[i, 1], stat="density")
            skew_raw = pd.Series(y).skew()
            skew_log = pd.Series(np.log1p(y)).skew()
            axes[i, 1].set_title(
                f"{city.upper()} — target distribution "
                f"(skew {skew_raw:.2f} -> {skew_log:.2f})"
            )
            axes[i, 1].legend()
        fig.suptitle("Target Overview: Seasonality & Log-Transform",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return self._save(fig, self.fig_root / "target_overview.png")

    def feature_selection_scores(self, scores: pd.Series, selected: List[str],
                                 city: str, top_n: int = 30) -> str:
        """Horizontal bar of mutual-information scores; selected features bold."""
        top = scores.head(top_n)
        colors = [self.colors["good"] if f in selected else "#b0b0b0"
                  for f in top.index]
        fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(top))))
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_xlabel("Mutual information score")
        ax.set_title(f"Feature Selection (MI) — {city.upper()} "
                     f"(green = selected, k={len(selected)})")
        ax.tick_params(axis="y", labelsize=8)
        return self._save(fig, self._city_dir(city, "eda") / "feature_selection_mi.png")

    # ================================================================== #
    # PER-MODEL x PER-CITY FIGURES
    # ================================================================== #
    def actual_vs_predicted(self, r: ModelResult) -> str:
        """Line chart of actual vs predicted over the hold-out period."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(r.dates, r.y_true, color=self.colors["actual"], lw=1.8,
                label="Actual", marker="o", markersize=2.5)
        ax.plot(r.dates, r.y_pred, color=self.colors["predicted"], lw=1.8,
                label="Predicted", ls="--")
        ax.set_title(f"Actual vs Predicted — {r.model_name} ({r.city.upper()})")
        ax.set_xlabel("Week")
        ax.set_ylabel("Total cases")
        ax.legend(loc="upper right")
        ax.text(0.01, 0.97, f"MAE={r.mae:.2f}  RMSE={r.rmse:.2f}  R²={r.r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
        return self._save(fig, self._model_path(r, "actual_vs_predicted"))

    def residual_plot(self, r: ModelResult) -> str:
        """Residuals (actual - predicted) over time with a zero reference line."""
        resid = np.asarray(r.y_true) - np.asarray(r.y_pred)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.axhline(0, color="#666666", lw=1)
        ax.plot(r.dates, resid, color=self._city_color(r.city), lw=1.2)
        ax.fill_between(r.dates, resid, color=self._city_color(r.city), alpha=0.2)
        ax.set_title(f"Residuals over Time — {r.model_name} ({r.city.upper()})")
        ax.set_xlabel("Week")
        ax.set_ylabel("Residual (actual − predicted)")
        return self._save(fig, self._model_path(r, "residuals_over_time"))

    def prediction_error_scatter(self, r: ModelResult) -> str:
        """Scatter of actual vs predicted with the ideal y=x reference line."""
        yt, yp = np.asarray(r.y_true), np.asarray(r.y_pred)
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.scatter(yt, yp, alpha=0.55, color=self._city_color(r.city),
                   edgecolor="white", s=32)
        lim = [0, max(yt.max(), yp.max()) * 1.05 + 1]
        ax.plot(lim, lim, color="#333333", ls="--", lw=1.2, label="Ideal (y=x)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"Prediction Error Scatter — {r.model_name} ({r.city.upper()})")
        ax.set_xlabel("Actual cases")
        ax.set_ylabel("Predicted cases")
        ax.legend()
        return self._save(fig, self._model_path(r, "error_scatter"))

    def residual_histogram(self, r: ModelResult) -> str:
        """Distribution of residuals with a KDE and mean marker."""
        resid = np.asarray(r.y_true) - np.asarray(r.y_pred)
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(resid, bins=30, kde=True, color=self._city_color(r.city),
                     alpha=0.6, ax=ax)
        ax.axvline(float(np.mean(resid)), color=self.colors["bad"], ls="--",
                   lw=1.5, label=f"mean={np.mean(resid):.2f}")
        ax.axvline(0, color="#333333", lw=1)
        ax.set_title(f"Residual Distribution — {r.model_name} ({r.city.upper()})")
        ax.set_xlabel("Residual (actual − predicted)")
        ax.legend()
        return self._save(fig, self._model_path(r, "residual_hist"))

    def feature_importance(self, r: ModelResult, top_n: int = 20) -> Optional[str]:
        """Horizontal bar of the top-``n`` feature importances (tree models)."""
        if r.feature_importance is None or r.feature_importance.empty:
            return None
        imp = r.feature_importance.sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(imp))))
        ax.barh(imp.index[::-1], imp.values[::-1],
                color=self._city_color(r.city))
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {r.model_name} ({r.city.upper()})")
        ax.tick_params(axis="y", labelsize=8)
        return self._save(fig, self._model_path(r, "feature_importance"))

    def model_dashboard(self, r: ModelResult) -> str:
        """Combined 6-panel diagnostic dashboard for one model/city."""
        yt, yp = np.asarray(r.y_true), np.asarray(r.y_pred)
        resid = yt - yp
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # (0,0) actual vs predicted line
        axes[0, 0].plot(r.dates, yt, color=self.colors["actual"], lw=1.5,
                        label="Actual")
        axes[0, 0].plot(r.dates, yp, color=self.colors["predicted"], lw=1.5,
                        ls="--", label="Predicted")
        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].legend(fontsize=8)

        # (0,1) error scatter
        lim = [0, max(yt.max(), yp.max()) * 1.05 + 1]
        axes[0, 1].scatter(yt, yp, alpha=0.5, color=self._city_color(r.city),
                           edgecolor="white", s=25)
        axes[0, 1].plot(lim, lim, "--", color="#333333", lw=1)
        axes[0, 1].set_xlim(lim); axes[0, 1].set_ylim(lim)
        axes[0, 1].set_title("Prediction Error Scatter")
        axes[0, 1].set_xlabel("Actual"); axes[0, 1].set_ylabel("Predicted")

        # (0,2) residuals over time
        axes[0, 2].axhline(0, color="#666666", lw=1)
        axes[0, 2].plot(r.dates, resid, color=self._city_color(r.city), lw=1)
        axes[0, 2].set_title("Residuals over Time")

        # (1,0) residual histogram
        sns.histplot(resid, bins=25, kde=True, color=self._city_color(r.city),
                     alpha=0.6, ax=axes[1, 0])
        axes[1, 0].axvline(0, color="#333333", lw=1)
        axes[1, 0].set_title("Residual Distribution")

        # (1,1) feature importance (or metric summary text)
        if r.feature_importance is not None and not r.feature_importance.empty:
            imp = r.feature_importance.sort_values(ascending=False).head(12)
            axes[1, 1].barh(imp.index[::-1], imp.values[::-1],
                            color=self._city_color(r.city))
            axes[1, 1].tick_params(axis="y", labelsize=7)
            axes[1, 1].set_title("Top Feature Importances")
        else:
            axes[1, 1].axis("off")
            axes[1, 1].text(0.5, 0.5, "(linear model —\nsee coefficients)",
                            ha="center", va="center", fontsize=11)
            axes[1, 1].set_title("Feature Importance")

        # (1,2) metric summary
        axes[1, 2].axis("off")
        txt = (f"Model: {r.model_name}\nCity: {r.city.upper()}\n\n"
               f"MAE  = {r.mae:.3f}\nRMSE = {r.rmse:.3f}\nR²   = {r.r2:.3f}\n\n"
               f"Tuned: {r.tuned}")
        if r.best_params:
            params = "\n".join(f"  {k}={v}" for k, v in list(r.best_params.items())[:8])
            txt += f"\n\nBest params:\n{params}"
        axes[1, 2].text(0.02, 0.98, txt, va="top", ha="left", fontsize=10,
                        family="monospace")

        fig.suptitle(f"Diagnostic Dashboard — {r.model_name} ({r.city.upper()})",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return self._save(fig, self._model_path(r, "dashboard"))

    # ================================================================== #
    # COMPARISON FIGURES
    # ================================================================== #
    def model_comparison_bars(self, metrics: pd.DataFrame) -> str:
        """Grouped bar charts of MAE / RMSE / R^2 per model, split by city."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
            pivot = metrics.pivot(index="Model", columns="City", values=metric)
            pivot = pivot.sort_values(pivot.columns[0])
            pivot.plot(kind="bar", ax=ax,
                       color=[self._city_color(c) for c in pivot.columns])
            ax.set_title(f"{metric} by Model")
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.legend(title="City")
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
        fig.suptitle("Model Comparison — MAE (primary), RMSE, R²",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return self._save(fig, self.fig_root / "model_comparison.png")

    def all_models_timeline(self, city: str,
                            results: List[ModelResult]) -> str:
        """Overlay every model's hold-out prediction against the actuals."""
        if not results:
            return ""
        base = results[0]
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(base.dates, base.y_true, color="black", lw=2.4,
                label="Actual", zorder=10)
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(sorted(results, key=lambda x: x.mae)):
            ax.plot(r.dates, r.y_pred, lw=1.2, alpha=0.85,
                    color=cmap(i % 10), label=f"{r.model_name} (MAE={r.mae:.1f})")
        ax.set_title(f"All Models — Hold-out Predictions ({city.upper()})")
        ax.set_xlabel("Week"); ax.set_ylabel("Total cases")
        ax.legend(fontsize=8, ncol=2)
        return self._save(fig, self._city_dir(city) / "all_models_timeline.png")

    # ------------------------------------------------------------------ #
    def _model_path(self, r: ModelResult, name: str) -> Path:
        d = self._city_dir(r.city, r.model_key)
        return d / f"{name}.png"
