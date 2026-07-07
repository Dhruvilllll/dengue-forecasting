"""Hyperparameter tuning for the tree-based models only.

Search-strategy rationale
-------------------------
* **Decision Tree -> GridSearchCV.** The tree has only a few impactful knobs
  (``max_depth``, ``min_samples_leaf``, ``min_samples_split``) with a small,
  discrete set of sensible values, so an *exhaustive* grid is cheap and
  guarantees the best combination in the grid is found.

* **Random Forest / XGBoost / LightGBM -> RandomizedSearchCV.** Their joint
  hyperparameter spaces are large and continuous-ish (hundreds to thousands of
  combinations). A full grid would be prohibitively expensive; randomized search
  samples ``n_iter`` configurations and, per Bergstra & Bengio (2012), reliably
  finds near-optimal settings at a fraction of the cost because only a few
  dimensions truly matter.

Cross-validation
----------------
All searches use :class:`~sklearn.model_selection.TimeSeriesSplit` so every
validation fold is strictly *after* its training fold — no future weather is
ever used to score a past week. The scoring metric is negative MAE, matching the
official DengAI evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)

from .config import Config
from .feature_engineering import inverse_transform_target
from .models import ModelFactory
from .utils import get_logger, rmse

logger = get_logger(__name__)


@dataclass
class TuningResult:
    """Outcome of a hyperparameter search."""

    best_estimator: Any
    best_params: Dict[str, Any]
    best_cv_score: float          # positive MAE (sign-flipped back)
    search_type: str              # "grid" | "random"


class HyperparameterTuner:
    """Runs Grid/Randomized search for the four tunable tree models."""

    def __init__(self, cfg: Config, factory: ModelFactory) -> None:
        self.cfg = cfg
        self.factory = factory
        self.seed = cfg.project.random_seed
        self.cv = TimeSeriesSplit(n_splits=cfg.cv.n_splits)
        self.n_iter = cfg.tuning.n_iter
        self.target_transform = cfg.feature_engineering.target_transform
        self.scoring = self._build_scorer()

    def _build_scorer(self):
        """Return the CV scorer for the tree-model search.

        ``tune_objective: "rmse"`` builds a scorer that inverse-transforms the
        model's (log) predictions back to the original case scale *inside* each
        CV fold, then computes RMSE — so the search truly minimizes real-case
        RMSE. ``"mae"`` (default) uses the configured MAE scoring string.
        """
        objective = str(getattr(self.cfg.cv, "tune_objective", "mae")).lower()
        if objective == "rmse":
            transform = self.target_transform

            def _rmse_original(y_true, y_pred):
                yt = inverse_transform_target(y_true, transform)
                yp = inverse_transform_target(y_pred, transform)
                return rmse(yt, yp)

            logger.info("Tuning objective: RMSE on the original case scale")
            return make_scorer(_rmse_original, greater_is_better=False)
        logger.info("Tuning objective: %s", self.cfg.cv.scoring)
        return self.cfg.cv.scoring

    def _params_for(self, key: str) -> Dict[str, Any]:
        return self.cfg.tuning[key].to_dict()

    def tune(self, key: str, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """Tune model ``key`` on ``(X, y)`` and return the best refit estimator.

        ``y`` is expected already on the modeling (log1p) scale.
        """
        base = self.factory.create(key)
        spec = self._params_for(key)
        search_type = spec["search"]

        if search_type == "grid":
            grid = spec["param_grid"]
            search = GridSearchCV(
                estimator=base,
                param_grid=grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                refit=True,
            )
            n_combos = int(np.prod([len(v) for v in grid.values()]))
            logger.info("[%s] GridSearchCV over %d combinations", key, n_combos)
        elif search_type == "random":
            dist = spec["param_distributions"]
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=dist,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                random_state=self.seed,
                refit=True,
            )
            logger.info("[%s] RandomizedSearchCV: %d of a large space",
                        key, self.n_iter)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown search type {search_type!r} for {key!r}")

        search.fit(X, y)
        best_score = -float(search.best_score_)  # sign-flip the neg-scoring
        objective = str(getattr(self.cfg.cv, "tune_objective", "mae")).upper()
        logger.info("[%s] best CV %s=%.4f | params=%s",
                    key, objective, best_score, search.best_params_)
        return TuningResult(
            best_estimator=search.best_estimator_,
            best_params=dict(search.best_params_),
            best_cv_score=best_score,
            search_type=search_type,
        )
