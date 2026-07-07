"""
dengue_forecast
===============

A modular, production-quality pipeline that predicts the weekly number of
dengue fever cases for San Juan (``sj``) and Iquitos (``iq``) from historical
weather, climate and vegetation data.

The package is organised as a set of single-responsibility modules::

    config              -> typed access to config/config.yaml
    utils               -> logging, seeding, IO helpers
    data_loader         -> load + merge raw CSVs, split per city
    preprocessing       -> cleaning, missing-value imputation, scaling
    feature_engineering -> lag / rolling / seasonal feature construction
    models              -> factory for the 8 permitted regression models
    tuning              -> Grid / Randomized hyperparameter search (tree models)
    evaluation          -> MAE / RMSE / R2 metrics, comparison & ranking
    visualization       -> publication-quality (300 DPI) figures
    pipeline            -> end-to-end orchestrator tying it all together

The two cities are modeled **independently** because their climates and dengue
dynamics differ sharply.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "utils",
    "data_loader",
    "preprocessing",
    "feature_engineering",
    "models",
    "tuning",
    "evaluation",
    "visualization",
    "pipeline",
]
