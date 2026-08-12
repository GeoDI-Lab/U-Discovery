from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_path


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON configuration {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be an object")
    for key in ("name", "dataset", "split", "training"):
        if key not in config:
            raise ConfigError(f"Missing required configuration key: {key}")
    return config, config_path


def resolved_dataset_path(config: dict[str, Any], config_path: Path) -> Path:
    value = config.get("dataset")
    if not isinstance(value, str) or not value:
        raise ConfigError("dataset must be a non-empty path string")
    return resolve_path(value, base=config_path.parent)
