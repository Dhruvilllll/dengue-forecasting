"""Typed configuration loader.

Loads ``config/config.yaml`` once and exposes it as a light, attribute-style
object so the rest of the code never hard-codes paths or hyperparameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Attribute/dict hybrid wrapper around the parsed YAML configuration.

    Nested mappings are wrapped recursively so both ``cfg.paths.output_dir`` and
    ``cfg["paths"]["output_dir"]`` work. Filesystem paths are additionally
    resolved to absolute paths relative to the project root.
    """

    def __init__(self, data: Dict[str, Any], root: Path | None = None) -> None:
        self._data = data
        self._root = root
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    def _wrap(self, value: Any) -> Any:
        """Recursively wrap nested dicts as :class:`Config` objects."""
        if isinstance(value, dict):
            return Config(value, self._root)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return the raw (unwrapped) dictionary."""
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({list(self._data.keys())})"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing ``config/`` and
    ``dataset/`` is found. Falls back to the current working directory."""
    start = (start or Path(__file__).resolve()).parent
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "config.yaml").exists():
            return candidate
    return Path.cwd()


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    """Load and return the project configuration.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config file. When omitted, the loader
        auto-discovers ``config/config.yaml`` from the project root.

    Returns
    -------
    Config
        The parsed configuration, with all ``paths.*`` entries resolved to
        absolute paths.
    """
    root = find_project_root()
    path = Path(config_path) if config_path else root / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Resolve every path under `paths:` to an absolute path from the root.
    if "paths" in raw:
        raw["paths"] = {
            k: str((root / v).resolve()) for k, v in raw["paths"].items()
        }

    cfg = Config(raw, root=root)
    setattr(cfg, "root", str(root))
    return cfg
