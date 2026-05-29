"""
Loads settings.yaml once and exposes a typed config object.
Usage:  from config import cfg
"""
from __future__ import annotations
import os
import logging.config
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

_ROOT = Path(__file__).parent.parent
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
_LOGGING_PATH  = _ROOT / "config" / "logging.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Section:
    """Dict-backed namespace with attribute access."""
    def __init__(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _Section(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class Config(_Section):
    pass


def _setup_logging() -> None:
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    if _LOGGING_PATH.exists():
        data = _load_yaml(_LOGGING_PATH)
        # Make file handler path absolute
        for handler in data.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(_ROOT / handler["filename"])
        logging.config.dictConfig(data)


cfg = Config(_load_yaml(_SETTINGS_PATH))
_setup_logging()
