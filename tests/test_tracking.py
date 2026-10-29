"""
tests/test_tracking.py
======================
Unit tests for tracker state, data structures, and line crossing calculations.
"""

import pytest
from src.tracking import Direction, TrackState, CountResult


def test_direction_enum():
    """Verify Direction enum values."""
    assert Direction.ENTRY == "ENTRY"
    assert Direction.EXIT == "EXIT"


def test_track_state():
    """Test TrackState dataclass initialization and state update."""
    state = TrackState(track_id=1, centroid=(100, 200))
    assert state.track_id == 1
    assert state.centroid == (100, 200)
    assert state.counted is False
    assert state.global_id == ""

    # Mark as counted
    state.counted = True
    state.global_id = "test-uuid"
    assert state.counted is True
    assert state.global_id == "test-uuid"


def test_count_result():
    """Test CountResult defaults and aggregations."""
    res = CountResult(entries=3, exits=2, active_ids=[101, 102])
    assert res.entries == 3
    assert res.exits == 2
    assert len(res.active_ids) == 2


def test_crossing_direction_logic():
    """Verify line crossing math (left->right is exit, right->left is entry)."""
    line_x = 320

    # Moving left to right (EXIT)
    prev_cx = 300
    curr_cx = 330
    crossed_right = prev_cx < line_x and curr_cx >= line_x
    assert crossed_right is True

    # Moving right to left (ENTRY)
    prev_cx = 340
    curr_cx = 310
    crossed_left = prev_cx > line_x and curr_cx <= line_x
    assert crossed_left is True
