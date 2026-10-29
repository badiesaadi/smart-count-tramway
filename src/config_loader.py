"""
config_loader.py — Smart Count Tramway
======================================
Robust configuration loader for system parameters.
Loads YAML configuration files with fallback to safe defaults,
environment variables, and CLI overrides.

Author  : Smart Count Tramway Team
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "source": "0",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "detection": {
        "model_path": "models/yolov8n.pt",
        "confidence_threshold": 0.45,
        "iou_threshold": 0.45,
        "device": "cpu",
        "target_classes": [0],
    },
    "tracking": {
        "max_age": 30,
        "n_init": 3,
        "max_cosine_distance": 0.4,
        "nn_budget": 100,
    },
    "counting": {
        "line_position": 0.5,
        "direction_sensitivity": 15,
    },
    "reid": {
        "enabled": True,
        "similarity_threshold": 0.75,
        "ttl_seconds": 300,
    },
    "anonymization": {
        "enabled": True,
        "method": "blur",
        "blur_strength": 25,
        "face_confidence": 0.6,
    },
    "database": {
        "path": "data/tramway_counts.db",
    },
    "logging": {
        "level": "INFO",
        "log_file": "data/tramway.log",
    },
    "stop_info": {
        "current_stop": "Kharouba",
        "all_stops": [
            "Kharouba",
            "Salamandre",
            "Gare SNTF",
            "Nouvelle Gare Routière",
            "Centre Ville",
        ],
    },
}


class Config:
    """Wrapper around dictionary config with nested dot/dict access."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a nested config value using dot notation.
        Example: config.get('detection.confidence_threshold', 0.5)
        """
        keys = key_path.split(".")
        current = self._data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a raw dictionary."""
        return self._data

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def __contains__(self, item: str) -> bool:
        return item in self._data


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from a YAML file.
    Merges found values onto default configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    merged = dict(DEFAULT_CONFIG)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                # Shallow-merge top-level sections
                for sec, val in loaded.items():
                    if isinstance(val, dict) and sec in merged and isinstance(merged[sec], dict):
                        merged[sec] = {**merged[sec], **val}
                    else:
                        merged[sec] = val
            logger.debug(f"[Config] Successfully loaded configuration from {path}")
        except Exception as e:
            logger.warning(f"[Config] Failed to read {path} ({e}). Falling back to defaults.")
    else:
        logger.debug(f"[Config] Configuration file not found at {path}, using defaults.")

    return Config(merged)

