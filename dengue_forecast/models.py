"""Model factory for the eight permitted regression estimators.

Allowed models (and nothing else):

===================  ==================================  =========  ==========
Key                  Estimator                           Family     Tuned?
===================  ==================================  =========  ==========
linear_regression    LinearRegression                    linear     no
ridge                Ridge (L2)                          linear     no
lasso                Lasso (L1)                          linear     no
elastic_net          ElasticNet (L1 + L2)                linear     no
decision_tree        DecisionTreeRegressor               tree       yes (grid)
random_forest        RandomForestRegressor               ensemble   yes (random)
xgboost              XGBRegressor                        boosting   yes (random)
lightgbm             LGBMRegressor                       boosting   yes (random)
===================  ==================================  =========  ==========

The linear family (including the regularized Ridge/Lasso/ElasticNet variants) is
deliberately **not** tuned — their fixed hyperparameters come from the config,
and they serve as strong, cheap, interpretable baselines. Tree/ensemble/boosting
models *are* tuned because their capacity is highly sensitive to depth, leaf
size, learning rate, etc.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install with `pip install xgboost`.") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("lightgbm is required. Install with `pip install lightgbm`.") from exc

from .config import Config

# Human-readable labels used in tables/plots, in a sensible display order.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge (L2)",
    "lasso": "Lasso (L1)",
    "elastic_net": "Elastic Net (L1+L2)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# Which models are linear (untuned) vs tree-based (tuned + have importances).
LINEAR_MODELS: List[str] = ["linear_regression", "ridge", "lasso", "elastic_net"]
TREE_MODELS: List[str] = ["decision_tree", "random_forest", "xgboost", "lightgbm"]


class ModelFactory:
    """Constructs freshly-initialised, seeded estimators from the config."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.seed = cfg.project.random_seed
        self._builders: Dict[str, Callable[[], object]] = {
            "linear_regression": self._linear_regression,
            "ridge": self._ridge,
            "lasso": self._lasso,
            "elastic_net": self._elastic_net,
            "decision_tree": self._decision_tree,
            "random_forest": self._random_forest,
            "xgboost": self._xgboost,
            "lightgbm": self._lightgbm,
        }

    # -- keys / metadata --------------------------------------------------- #
    def model_keys(self) -> List[str]:
        """All model keys in display order."""
        return list(MODEL_DISPLAY_NAMES.keys())

    def display_name(self, key: str) -> str:
        return MODEL_DISPLAY_NAMES[key]

    def is_tunable(self, key: str) -> bool:
        return key in TREE_MODELS

    def create(self, key: str):
        """Return a fresh estimator instance for ``key``."""
        if key not in self._builders:
            raise KeyError(f"Unknown model key: {key!r}")
        return self._builders[key]()

    # -- linear family (untuned) ------------------------------------------ #
    def _linear_regression(self):
        return LinearRegression()

    def _ridge(self):
        return Ridge(alpha=self.cfg.linear_models.ridge.alpha,
                     random_state=self.seed)

    def _lasso(self):
        lm = self.cfg.linear_models.lasso
        return Lasso(alpha=lm.alpha, max_iter=lm.max_iter, random_state=self.seed)

    def _elastic_net(self):
        lm = self.cfg.linear_models.elastic_net
        return ElasticNet(alpha=lm.alpha, l1_ratio=lm.l1_ratio,
                          max_iter=lm.max_iter, random_state=self.seed)

    # -- tree family (tuned; these are the *base* estimators) ------------- #
    def _decision_tree(self):
        return DecisionTreeRegressor(random_state=self.seed)

    def _random_forest(self):
        return RandomForestRegressor(random_state=self.seed, n_jobs=-1)

    def _xgboost(self):
        return XGBRegressor(
            random_state=self.seed,
            n_jobs=-1,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )

    def _lightgbm(self):
        return LGBMRegressor(
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
