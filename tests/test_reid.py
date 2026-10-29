"""
tests/test_reid.py
==================
Unit tests for ReID registry and passenger appearance deduplication.
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.reid import GlobalPassenger, ReidRegistry, EMBEDDING_DIM


@pytest.fixture
def dummy_embedding():
    """Generate a reproducible L2-normalized 1280-D vector."""
    np.random.seed(42)
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_global_passenger_dataclass(dummy_embedding):
    """Test GlobalPassenger creation and attribute assignment."""
    now = time.time()
    passenger = GlobalPassenger(
        global_id="test-uuid-1234",
        embedding=dummy_embedding,
        first_seen=now,
        last_seen=now,
        count_stop="Kharouba",
    )
    assert passenger.global_id == "test-uuid-1234"
    assert passenger.count_stop == "Kharouba"
    assert np.isclose(np.linalg.norm(passenger.embedding), 1.0)


def test_reid_registry_match_and_deduplication(dummy_embedding):
    """Test registry matching and duplicate detection using mocked extractor."""
    with patch("src.reid.AppearanceExtractor") as mock_ext:
        mock_instance = MagicMock()
        mock_instance.extract.return_value = dummy_embedding
        mock_ext.return_value = mock_instance

        registry = ReidRegistry(stop_name="Kharouba", similarity_threshold=0.75, ttl_seconds=60)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)

        # First detection -> Should count as new unique passenger
        id1, should_count1 = registry.identify(dummy_frame, bbox, stop="Kharouba")
        assert should_count1 is True
        assert id1 in registry._registry
        assert registry.total_unique == 1
        assert registry.total_duplicate == 0

        # Second detection with same embedding -> Duplicate, should NOT count
        id2, should_count2 = registry.identify(dummy_frame, bbox, stop="Kharouba")
        assert should_count2 is False
        assert id2 == id1
        assert registry.total_unique == 1
        assert registry.total_duplicate == 1


def test_reid_ttl_pruning(dummy_embedding):
    """Verify that expired passenger embeddings are correctly pruned."""
    with patch("src.reid.AppearanceExtractor") as mock_ext:
        mock_instance = MagicMock()
        mock_instance.extract.return_value = dummy_embedding
        mock_ext.return_value = mock_instance

        registry = ReidRegistry(stop_name="Kharouba", similarity_threshold=0.75, ttl_seconds=1)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)

        # Register passenger with an old timestamp
        id1, _ = registry.identify(dummy_frame, bbox, stop="Kharouba")
        registry._registry[id1].last_seen = time.time() - 10  # 10s ago, TTL is 1s

        # Prune expired
        registry._prune_expired()
        assert id1 not in registry._registry
