"""
Smart Count Tramway — Core Package
==================================
Modules for edge-AI detection, tracking, re-identification, and database management.
"""

from src.config_loader import Config, load_config
from src.database import Database, CountRecord
from src.detection import PersonDetector
from src.reid import ReidRegistry, AppearanceExtractor
from src.tracking import PassengerTracker

__all__ = [
    "Config",
    "load_config",
    "Database",
    "CountRecord",
    "PersonDetector",
    "ReidRegistry",
    "AppearanceExtractor",
    "PassengerTracker",
]
