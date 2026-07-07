#!/usr/bin/env python
"""Command-line entry point for the dengue forecasting pipeline.

Usage
-----
    python scripts/run_pipeline.py                 # run everything
    python scripts/run_pipeline.py --config path/to/config.yaml

The script simply wires the CLI to :class:`dengue_forecast.pipeline.DenguePipeline`,
which performs all data loading, feature engineering, modeling, evaluation and
visualization. All outputs are written under the ``outputs/`` directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a plain script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dengue_forecast.config import load_config  # noqa: E402
from dengue_forecast.pipeline import DenguePipeline  # noqa: E402
from dengue_forecast.utils import get_logger  # noqa: E402

logger = get_logger("run_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dengue weekly-cases forecasting pipeline."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (defaults to config/config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger.info("Loaded config for project: %s", cfg.project.name)

    pipeline = DenguePipeline(cfg)
    outputs = pipeline.run()

    logger.info("Best model per city: %s", outputs["summary"]["best_by_city"])
    logger.info("Overall best (combined MAE): %s", outputs["summary"]["combined_best"])
    logger.info("All artefacts written under: %s", cfg.paths.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
