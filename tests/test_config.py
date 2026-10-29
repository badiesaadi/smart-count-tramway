"""
tests/test_config.py
====================
Unit tests for configuration loading and validation.
"""

from pathlib import Path
import tempfile
import yaml
import pytest
from src.config_loader import Config, load_config, DEFAULT_CONFIG


def test_default_config():
    """Verify default configuration values."""
    config = Config(DEFAULT_CONFIG)
    assert config.get("camera.width") == 1280
    assert config.get("camera.fps") == 30
    assert config.get("detection.confidence_threshold") == 0.45
    assert config.get("nonexistent.key", default="fallback") == "fallback"


def test_load_config_from_file():
    """Verify loading and merging from a custom YAML file."""
    custom_yaml = {
        "camera": {
            "fps": 60,
        },
        "stop_info": {
            "current_stop": "Salamandre",
        },
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml", encoding="utf-8") as f:
        yaml.dump(custom_yaml, f)
        temp_path = f.name

    try:
        cfg = load_config(temp_path)
        # Overridden fields
        assert cfg.get("camera.fps") == 60
        assert cfg.get("stop_info.current_stop") == "Salamandre"
        # Preserved defaults
        assert cfg.get("camera.width") == 1280
        assert cfg.get("detection.confidence_threshold") == 0.45
    finally:
        Path(temp_path).unlink(missing_ok=True)

