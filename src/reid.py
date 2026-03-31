"""
reid.py — Smart Count Tramway
================================
Global Re-Identification (ReID) module.

PURPOSE
-------
DeepSORT gives each passenger a LOCAL track ID that is valid only within
one camera's view. When a passenger walks past Camera 1, they get ID=7.
When they appear in Camera 2 (same station), DeepSORT assigns them a NEW
ID (e.g. ID=3) — and we would incorrectly count them again.

This module solves that by:
  1. Extracting a visual "appearance embedding" (128-D feature vector)
     for every tracked person using a MobileNetV2 feature extractor.
  2. Maintaining a global registry of embeddings seen at the SAME station.
  3. When a new person appears, comparing their embedding against the
     registry using cosine similarity.
  4. If similarity > threshold → same person → DO NOT count again.
  5. If similarity < threshold → new person → assign new global ID, count.
  6. Embeddings expire after a configurable TTL (default 5 minutes) so a
     passenger who boards a tram, rides to the next station and boards again
     IS counted at the new station.

INTER-STATION LOGIC
-------------------
Each station runs its own ReidRegistry instance with its own embedding store.
Since a passenger physically cannot be at two stations simultaneously, there
is no cross-station sharing of the registry — so they ARE correctly counted
when they appear at a different station.

  Station A Registry: {global_id_1: embedding, global_id_2: embedding, ...}
  Station B Registry: {global_id_3: embedding, ...}   ← completely separate

Author  : Smart Count Tramway Team
Hardware: AMD Ryzen (dev) | Raspberry Pi 4 / Jetson Nano (prod)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

# Cosine similarity threshold for matching.
# 1.0 = identical vectors, 0.0 = completely different.
# 0.75 is a good balance: strict enough to avoid false matches,
# lenient enough to handle slight changes in pose/lighting.
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# How long (seconds) to keep an embedding in the registry.
# After this time, the person is assumed to have left the station area.
# Default = 5 minutes (300 seconds).
DEFAULT_TTL_SECONDS = 300

# Embedding vector dimension produced by MobileNetV2 feature extractor.
EMBEDDING_DIM = 1280


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalPassenger:
    """
    Represents one unique passenger seen at this station.

    global_id   : UUID string — unique across the entire system session.
    embedding   : 1280-D L2-normalised feature vector of their appearance.
    first_seen  : Unix timestamp of first detection.
    last_seen   : Unix timestamp of most recent detection (used for TTL).
    count_stop  : Name of the stop where this passenger was first counted.
                  If they re-appear at the SAME stop → do not count again.
                  If they appear at a DIFFERENT stop → count (new journey).
    """
    global_id  : str
    embedding  : np.ndarray
    first_seen : float
    last_seen  : float
    count_stop : str


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceExtractor:
    """
    Extracts a fixed-length appearance embedding from a person crop.

    Uses MobileNetV2 (pretrained on ImageNet) with the final classification
    head removed, leaving a 1280-dimensional feature vector per person.

    MobileNetV2 is chosen because:
      • Lightweight (~14 MB) — runs well on Raspberry Pi / Jetson Nano.
      • Fast inference (~5ms per crop on CPU).
      • Good generalisation to pedestrian appearance.

    For production, this can be swapped for a proper ReID model like
    OSNet (torchreid) for higher accuracy at the cost of more compute.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load MobileNetV2, strip the classifier head
        import torchvision.models as models
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the final classifier — keep only the feature extractor
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Standard ImageNet normalisation
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        logger.info(f"[ReID] Appearance extractor loaded on {self.device}")

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a normalised 1280-D embedding from a BGR person crop.

        Parameters
        ----------
        crop : np.ndarray
            BGR image of a single person (any size — will be resized).

        Returns
        -------
        np.ndarray
            1280-D L2-normalised float32 vector.
            L2 normalisation ensures cosine similarity = dot product,
            which is fast to compute.
        """
        if crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Convert BGR (OpenCV) → RGB (PyTorch)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self.transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        features = self.model(tensor)                               # (1, 1280, 1, 1)
        embedding = features.squeeze().cpu().numpy()               # (1280,)

        # L2 normalise so that cosine_similarity(a, b) = dot(a, b)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ReID Registry
# ─────────────────────────────────────────────────────────────────────────────

class ReidRegistry:
    """
    Maintains a registry of unique passengers seen at ONE station.

    One ReidRegistry instance per tram stop. Instances are completely
    independent — no cross-station sharing.

    Parameters
    ----------
    stop_name           : str    Name of the tram stop this registry belongs to.
    similarity_threshold: float  Cosine similarity above which two embeddings
                                 are considered the same person (default 0.75).
    ttl_seconds         : int    Seconds before an embedding expires (default 300).
    device              : str    'cpu' or 'cuda'.
    """

    def __init__(
        self,
        stop_name            : str,
        similarity_threshold : float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds          : int   = DEFAULT_TTL_SECONDS,
        device               : Optional[str] = None,
    ) -> None:
        self.stop_name            = stop_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds          = ttl_seconds

        # The registry: maps global_id (str UUID) → GlobalPassenger
        self._registry: Dict[str, GlobalPassenger] = {}

        # Appearance feature extractor
        self.extractor = AppearanceExtractor(device=device)

        # Statistics
        self.total_unique    = 0
        self.total_duplicate = 0

        logger.info(
            f"[ReID] Registry initialised for stop='{stop_name}' "
            f"threshold={similarity_threshold} ttl={ttl_seconds}s"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def identify(
        self,
        frame   : np.ndarray,
        bbox    : Tuple[int, int, int, int],
        stop    : str,
    ) -> Tuple[str, bool]:
        """
        Identify a passenger and determine if they should be counted.

        Pipeline:
          1. Crop person from frame using bbox.
          2. Extract 1280-D appearance embedding.
          3. Compare against all non-expired registry entries.
          4. If match found → return existing global_id, should_count=False.
          5. If no match    → create new entry, return new global_id, should_count=True.

        Parameters
        ----------
        frame : np.ndarray   Full BGR frame.
        bbox  : tuple        (x1, y1, x2, y2) bounding box of the person.
        stop  : str          Name of the stop where this detection occurred.

        Returns
        -------
        (global_id, should_count)
            global_id    : str   UUID for this passenger.
            should_count : bool  True = new unique passenger, count them.
                                 False = already seen at this station, skip.
        """
        # Step 1: Prune expired entries first
        self._prune_expired()

        # Step 2: Crop person from frame
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        # Step 3: Extract embedding
        embedding = self.extractor.extract(crop)

        # Step 4: Search registry for a match
        match_id, similarity = self._find_match(embedding)

        if match_id is not None:
            # ── Known passenger ──────────────────────────────────────────────
            # Update their last_seen timestamp (refreshes TTL)
            self._registry[match_id].last_seen = time.time()
            self._registry[match_id].embedding = embedding  # update with latest

            self.total_duplicate += 1
            logger.debug(
                f"[ReID] DUPLICATE — global_id={match_id[:8]}... "
                f"similarity={similarity:.3f} stop={stop}"
            )
            return match_id, False  # do NOT count

        else:
            # ── New passenger ─────────────────────────────────────────────────
            global_id = str(uuid.uuid4())
            now = time.time()
            self._registry[global_id] = GlobalPassenger(
                global_id  = global_id,
                embedding  = embedding,
                first_seen = now,
                last_seen  = now,
                count_stop = stop,
            )
            self.total_unique += 1
            logger.debug(
                f"[ReID] NEW passenger — global_id={global_id[:8]}... "
                f"stop={stop} total_unique={self.total_unique}"
            )
            return global_id, True  # DO count

    def get_stats(self) -> dict:
        """Return current registry statistics."""
        return {
            "stop"            : self.stop_name,
            "registry_size"   : len(self._registry),
            "total_unique"    : self.total_unique,
            "total_duplicate" : self.total_duplicate,
            "duplicate_rate"  : (
                self.total_duplicate /
                max(1, self.total_unique + self.total_duplicate)
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_match(
        self, query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Compare query_embedding against all registry entries.

        Uses cosine similarity (= dot product for L2-normalised vectors).
        Returns the best matching global_id and its similarity score,
        or (None, 0.0) if no match exceeds the threshold.

        Time complexity: O(N) where N = registry size.
        For a typical tram stop with <200 people at a time, this is instant.
        """
        best_id         : Optional[str] = None
        best_similarity : float         = 0.0

        for gid, passenger in self._registry.items():
            # Cosine similarity = dot product (both vectors are L2-normalised)
            similarity = float(np.dot(query_embedding, passenger.embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_id         = gid

        if best_similarity >= self.similarity_threshold:
            return best_id, best_similarity

        return None, best_similarity

    def _prune_expired(self) -> None:
        """
        Remove registry entries older than ttl_seconds.

        Called at the start of every identify() call to keep the registry
        from growing indefinitely during long recording sessions.
        """
        now     = time.time()
        expired = [
            gid for gid, p in self._registry.items()
            if (now - p.last_seen) > self.ttl_seconds
        ]
        for gid in expired:
            del self._registry[gid]
            logger.debug(f"[ReID] Expired global_id={gid[:8]}...")
