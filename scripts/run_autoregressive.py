#!/usr/bin/env python
"""Run the pipeline in AUTOREGRESSIVE (nowcast) mode.

This enables past-case features (`feature_engineering.autoregressive = True`)
and writes all artefacts to ``outputs_autoregressive/`` so the default
weather-only forecast outputs under ``outputs/`` are left untouched.

Autoregressive mode adds lagged weekly case counts as features, turning the task
into a short-horizon 1-week-ahead **nowcast**. Cases are highly autocorrelated,
so this greatly reduces RMSE — but it assumes recent case counts are known at
prediction time (the test set is therefore forecast recursively).

Usage
-----
    python scripts/run_autoregressive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dengue_forecast.config import load_config  # noqa: E402
from dengue_forecast.pipeline import DenguePipeline  # noqa: E402
from dengue_forecast.utils import get_logger  # noqa: E402

logger = get_logger("run_autoregressive")


def main() -> int:
    cfg = load_config()

    # 1) Enable autoregressive (nowcast) feature construction.
    setattr(cfg.feature_engineering, "autoregressive", True)

    # 2) Redirect every output path to outputs_autoregressive/ so the default
    #    weather-only deliverables under outputs/ are preserved.
    ar_root = Path(cfg.root) / "outputs_autoregressive"
    subdirs = {
        "output_dir": ar_root,
        "figures_dir": ar_root / "figures",
        "models_dir": ar_root / "models",
        "reports_dir": ar_root / "reports",
        "predictions_dir": ar_root / "predictions",
    }
    for attr, path in subdirs.items():
        setattr(cfg.paths, attr, str(path))

    logger.info("AUTOREGRESSIVE (nowcast) mode — outputs -> %s", ar_root)
    pipeline = DenguePipeline(cfg)
    outputs = pipeline.run()
    logger.info("Best model per city: %s", outputs["summary"]["best_by_city"])
    logger.info("Overall best (combined MAE): %s", outputs["summary"]["combined_best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
