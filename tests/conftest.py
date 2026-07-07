"""Shared pytest fixtures.

Data is loaded and prepared **once per test session** (session-scoped fixtures)
so the suite stays fast — no model tuning or full pipeline run is triggered here.
Iquitos is used for the per-city fixtures because it is the smaller series.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from dengue_forecast.config import load_config  # noqa: E402
from dengue_forecast.pipeline import DenguePipeline  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def pipeline(cfg):
    return DenguePipeline(cfg)


@pytest.fixture(scope="session")
def raw(pipeline):
    return pipeline.loader.load()


@pytest.fixture(scope="session")
def iq_dataset(pipeline, raw):
    """Engineered (weather-only) dataset for Iquitos."""
    ds, _clean = pipeline._prepare_city(raw, "iq")
    return ds


@pytest.fixture(scope="session")
def iq_stack(pipeline, iq_dataset):
    """Split + log-transform + selection + scaling stack for Iquitos."""
    return pipeline._fit_transform_stack(iq_dataset)
