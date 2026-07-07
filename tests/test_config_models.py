"""Config loading and the model factory (the 8 permitted models, nothing else)."""

from __future__ import annotations

import os

from dengue_forecast.models import (
    LINEAR_MODELS,
    TREE_MODELS,
    MODEL_DISPLAY_NAMES,
    ModelFactory,
)


def test_config_basics(cfg):
    assert cfg.project.random_seed == 42
    assert list(cfg.data.cities) == ["sj", "iq"]
    # All configured paths are resolved to absolute paths.
    assert os.path.isabs(cfg.paths.output_dir)
    assert cfg.data.target == "total_cases"


def test_exactly_eight_models(cfg):
    factory = ModelFactory(cfg)
    keys = factory.model_keys()
    assert len(keys) == 8, "exactly eight models must be exposed"
    assert set(keys) == set(MODEL_DISPLAY_NAMES)
    # No disallowed families leaked in.
    forbidden = {"svm", "svr", "knn", "mlp", "gam", "bayes", "catboost"}
    assert not (set(keys) & forbidden)


def test_only_tree_models_are_tunable(cfg):
    factory = ModelFactory(cfg)
    for key in LINEAR_MODELS:
        assert not factory.is_tunable(key), f"{key} must NOT be tuned"
    for key in TREE_MODELS:
        assert factory.is_tunable(key), f"{key} must be tunable"


def test_every_model_instantiates(cfg):
    factory = ModelFactory(cfg)
    for key in factory.model_keys():
        est = factory.create(key)
        assert hasattr(est, "fit") and hasattr(est, "predict")
