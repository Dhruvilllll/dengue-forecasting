"""Evaluation metrics, per-city comparison tables and model ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .utils import get_logger, rmse

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """All artefacts produced by evaluating one model on one city."""

    city: str
    model_key: str
    model_name: str
    mae: float
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    dates: pd.Series
    best_params: Dict[str, object] = field(default_factory=dict)
    feature_importance: pd.Series | None = None
    tuned: bool = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE and R^2 as a plain dict (original count scale)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def results_to_frame(results: List[ModelResult]) -> pd.DataFrame:
    """Flatten a list of :class:`ModelResult` into a tidy metrics table."""
    rows = [
        {
            "City": r.city,
            "Model": r.model_name,
            "model_key": r.model_key,
            "MAE": r.mae,
            "RMSE": r.rmse,
            "R2": r.r2,
            "Tuned": r.tuned,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def rank_models(metrics: pd.DataFrame, by: str = "MAE") -> pd.DataFrame:
    """Rank models best->worst within each city (lower MAE/RMSE is better).

    A ``Rank`` column (1 = best) is added per city. Sorting is ascending for
    error metrics; for R^2 it is descending.
    """
    ascending = by in ("MAE", "RMSE")
    out = []
    for city, grp in metrics.groupby("City"):
        grp = grp.sort_values(by, ascending=ascending).copy()
        grp["Rank"] = range(1, len(grp) + 1)
        out.append(grp)
    ranked = pd.concat(out, ignore_index=True)
    return ranked.sort_values(["City", "Rank"]).reset_index(drop=True)


def combined_scores(results: List[ModelResult]) -> pd.DataFrame:
    """Pool both cities' predictions per model into a single score.

    This mirrors how the DengAI competition reports one MAE across both cities.
    Predictions are concatenated (not averaged) before scoring, so each week
    counts once.
    """
    by_model: Dict[str, Dict[str, list]] = {}
    names: Dict[str, str] = {}
    for r in results:
        d = by_model.setdefault(r.model_key, {"t": [], "p": []})
        d["t"].append(np.asarray(r.y_true))
        d["p"].append(np.asarray(r.y_pred))
        names[r.model_key] = r.model_name

    rows = []
    for key, d in by_model.items():
        yt = np.concatenate(d["t"])
        yp = np.concatenate(d["p"])
        m = compute_metrics(yt, yp)
        rows.append({"Model": names[key], "model_key": key, **m})
    df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def format_comparison_report(ranked: pd.DataFrame,
                             combined: pd.DataFrame) -> str:
    """Produce a human-readable text report of the model comparison."""
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("DENGUE FORECASTING — MODEL COMPARISON REPORT")
    lines.append("=" * 74)
    lines.append("")
    lines.append("Metrics: MAE (primary), RMSE, R^2 — evaluated on the temporal")
    lines.append("hold-out (most-recent 25% of each city's weeks).")
    lines.append("")

    for city, grp in ranked.groupby("City"):
        lines.append(f"--- City: {city.upper()} " + "-" * (66 - len(city)))
        header = f"{'Rank':<5}{'Model':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'Tuned':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for _, row in grp.iterrows():
            lines.append(
                f"{int(row['Rank']):<5}{row['Model']:<24}"
                f"{row['MAE']:>10.3f}{row['RMSE']:>10.3f}{row['R2']:>10.3f}"
                f"{str(bool(row['Tuned'])):>8}"
            )
        best = grp.iloc[0]
        lines.append(f"  -> Best for {city.upper()}: {best['Model']} "
                     f"(MAE={best['MAE']:.3f})")
        lines.append("")

    lines.append("--- Combined (both cities pooled, DengAI-style) " + "-" * 26)
    header = f"{'Rank':<5}{'Model':<24}{'MAE':>10}{'RMSE':>10}{'R2':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in combined.iterrows():
        lines.append(
            f"{int(row['Rank']):<5}{row['Model']:<24}"
            f"{row['MAE']:>10.3f}{row['RMSE']:>10.3f}{row['R2']:>10.3f}"
        )
    lines.append("")
    lines.append(f"Overall best (combined MAE): {combined.iloc[0]['Model']} "
                 f"(MAE={combined.iloc[0]['MAE']:.3f})")
    lines.append("=" * 74)
    return "\n".join(lines)
