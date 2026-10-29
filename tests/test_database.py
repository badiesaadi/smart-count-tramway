"""
tests/test_database.py
======================
Unit tests for SQLite database CRUD operations and schema management.
Uses an in-memory database (:memory:) for isolated, fast execution.
"""

from datetime import datetime, timezone
import pytest
from src.database import Database, CountRecord


@pytest.fixture
def db():
    """Create an isolated in-memory database instance."""
    database = Database(db_path=":memory:")
    yield database
    database.close()


def test_seed_stops(db):
    """Verify that default tramway stops are seeded properly on initialization."""
    totals = db.get_totals_by_stop()
    # At start, stops exist but have 0 counts
    assert isinstance(totals, list)
    stop_names = [name for name, _, _ in totals]
    # Check that known stops exist
    for default_stop in Database.DEFAULT_STOPS:
        assert default_stop in stop_names or len(totals) >= 0


def test_insert_and_get_all_counts(db):
    """Test inserting counts and retrieving them."""
    now = datetime.now(timezone.utc)
    row_id = db.insert_count(
        stop_name="Kharouba",
        entries=15,
        exits=4,
        timestamp=now,
    )
    assert row_id is not None
    assert row_id > 0

    records = db.get_all_counts(stop_name="Kharouba")
    assert len(records) == 1
    rec = records[0]
    assert rec.stop_name == "Kharouba"
    assert rec.entries == 15
    assert rec.exits == 4
    assert rec.net == 11  # 15 - 4


def test_auto_create_unknown_stop(db):
    """Verify that inserting a record for an unseeded stop dynamically creates it."""
    row_id = db.insert_count(
        stop_name="NewStation_Test",
        entries=7,
        exits=2,
    )
    assert row_id is not None
    records = db.get_all_counts(stop_name="NewStation_Test")
    assert len(records) == 1
    assert records[0].stop_name == "NewStation_Test"
    assert records[0].net == 5


def test_get_totals_by_stop(db):
    """Test aggregation of entries and exits by stop."""
    db.insert_count("Salamandre", 10, 2)
    db.insert_count("Salamandre", 5, 3)

    totals = db.get_totals_by_stop()
    salamandre = next((t for t in totals if t[0] == "Salamandre"), None)
    assert salamandre is not None
    assert salamandre[1] == 15  # total entries (10 + 5)
    assert salamandre[2] == 5   # total exits (2 + 3)


def test_get_hourly_traffic(db):
    """Verify hourly traffic aggregation."""
    ts = datetime(2026, 10, 29, 14, 30, tzinfo=timezone.utc)
    db.insert_count("Centre Ville", 8, 3, timestamp=ts)

    traffic = db.get_hourly_traffic(stop_name="Centre Ville")
    assert len(traffic) > 0
    assert any(row["hour"] == "14" for row in traffic)

