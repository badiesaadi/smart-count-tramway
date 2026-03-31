"""
tracking.py — Smart Count Tramway
===================================
Multi-object tracking using DeepSORT + Global ReID.

NEW in this version:
  - Integrates ReidRegistry from reid.py.
  - Each tracked person is identified globally before counting.
  - If the same person is seen by Camera 1 then Camera 2 at the SAME stop,
    they are NOT double-counted.
  - If they appear at a DIFFERENT stop, they ARE counted (new journey).

Virtual Line Crossing Logic (unchanged):
─────────────────────────────────────────
  A vertical line is drawn at a fixed X pixel position (line_y parameter,
  reused for X in vertical mode).

  Cross LEFT → RIGHT  (x increases): EXIT  (toward platform)
  Cross RIGHT → LEFT  (x decreases): ENTRY (into tram)

  Mathematically:
    prev_cx < line_x  AND  curr_cx >= line_x  →  EXIT
    prev_cx > line_x  AND  curr_cx <= line_x  →  ENTRY

Author  : Smart Count Tramway Team
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.reid import ReidRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    ENTRY = "ENTRY"
    EXIT  = "EXIT"


@dataclass
class TrackState:
    """Per-track state kept between frames."""
    track_id  : int
    centroid  : Tuple[int, int]   # (cx, cy)
    counted   : bool  = False     # True once this track crossed the line
    global_id : str   = ""        # UUID assigned by ReidRegistry


@dataclass
class CountResult:
    """Returned by PassengerTracker.update() for each frame."""
    entries    : int = 0
    exits      : int = 0
    active_ids : List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PassengerTracker:
    """
    Wraps DeepSORT tracking + virtual line crossing + global ReID.

    Parameters
    ----------
    stop_name   : str   Name of the tram stop (used by ReID registry).
    line_y      : int   X-pixel position of the vertical counting line.
                        None = auto-set to horizontal midpoint on first frame.
    max_age     : int   Frames DeepSORT keeps a track alive without detection.
    embedder    : str   DeepSORT appearance model ('mobilenet' or 'torchreid').
    reid_threshold : float  Cosine similarity threshold for global ReID (0–1).
    reid_ttl    : int   Seconds before a passenger's ReID entry expires.
    use_reid    : bool  Set False to disable global ReID (faster, less accurate).
    """

    def __init__(
        self,
        stop_name      : str   = "Unknown",
        line_y         : Optional[int] = None,
        max_age        : int   = 30,
        embedder       : str   = "mobilenet",
        reid_threshold : float = 0.75,
        reid_ttl       : int   = 300,
        use_reid       : bool  = True,
    ) -> None:

        self.stop_name = stop_name
        self.line_y    = line_y   # used as X position for vertical line
        self.use_reid  = use_reid

        # ── DeepSORT tracker ─────────────────────────────────────────────────
        self.tracker = DeepSort(
            max_age  = max_age,
            n_init   = 3,       # confirm track after 3 consecutive detections
            embedder = embedder,
            half     = True,
            bgr      = True,
        )

        # ── Global ReID registry ──────────────────────────────────────────────
        if use_reid:
            self.reid = ReidRegistry(
                stop_name            = stop_name,
                similarity_threshold = reid_threshold,
                ttl_seconds          = reid_ttl,
            )
            logger.info(f"[Tracker] Global ReID enabled for stop='{stop_name}'")
        else:
            self.reid = None
            logger.info("[Tracker] Global ReID disabled")

        # Per-track state (local DeepSORT IDs)
        self._track_history: Dict[int, TrackState] = {}

        # Running session totals
        self.total_entries = 0
        self.total_exits   = 0

        logger.info(
            f"[Tracker] DeepSORT initialised "
            f"(max_age={max_age}, embedder={embedder})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame      : np.ndarray,
        detections,               # List[Detection] from detection.py
    ) -> CountResult:
        """
        Process one frame: track people, apply ReID, count line crossings.

        Steps per frame:
          1. Convert Detection list → DeepSORT input format.
          2. Run DeepSORT .update_tracks() — assigns consistent local IDs.
          3. For each confirmed track that crosses the counting line:
               a. Query ReidRegistry with the person's crop.
               b. If new person (should_count=True)  → increment counter.
               c. If duplicate (should_count=False)  → skip, no double count.
          4. Return CountResult with entries/exits for THIS frame only.
        """
        h, w = frame.shape[:2]

        # Resolve line position on first frame
        if self.line_y is None:
            self.line_y = w // 2
            logger.info(f"[Tracker] Counting line set at x={self.line_y}px")

        # ── Format for DeepSORT ───────────────────────────────────────────────
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, "person"))

        tracks = self.tracker.update_tracks(raw, frame=frame)
        result = CountResult()

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            result.active_ids.append(tid)

            # ── Crossing detection ────────────────────────────────────────────
            if tid in self._track_history:
                prev_cx = self._track_history[tid].centroid[0]

                if not self._track_history[tid].counted:
                    crossed_right = prev_cx < self.line_y and cx >= self.line_y
                    crossed_left  = prev_cx > self.line_y and cx <= self.line_y

                    if crossed_right or crossed_left:
                        direction = "EXIT" if crossed_right else "ENTRY"

                        # ── Global ReID check ─────────────────────────────────
                        # Before counting, check if this person was already
                        # counted by another camera at this same stop.
                        should_count = True
                        global_id    = ""

                        if self.use_reid and self.reid is not None:
                            bbox = (
                                max(0, x1), max(0, y1),
                                min(w, x2), min(h, y2),
                            )
                            global_id, should_count = self.reid.identify(
                                frame, bbox, self.stop_name
                            )

                        if should_count:
                            if direction == "EXIT":
                                result.exits      += 1
                                self.total_exits  += 1
                            else:
                                result.entries     += 1
                                self.total_entries += 1

                            logger.debug(
                                f"[Tracker] Track {tid} → {direction} "
                                f"global_id={global_id[:8] if global_id else 'N/A'}..."
                            )
                        else:
                            logger.debug(
                                f"[Tracker] Track {tid} → {direction} "
                                f"SKIPPED (ReID duplicate) "
                                f"global_id={global_id[:8]}..."
                            )

                        # Mark as counted regardless (prevent re-triggering)
                        self._track_history[tid].counted   = True
                        self._track_history[tid].global_id = global_id

            # Update centroid history
            prev_state = self._track_history.get(
                tid, TrackState(tid, (cx, cy))
            )
            self._track_history[tid] = TrackState(
                track_id  = tid,
                centroid  = (cx, cy),
                counted   = prev_state.counted,
                global_id = prev_state.global_id,
            )

        # Prune dead tracks
        active_set = set(result.active_ids)
        for tid in [t for t in self._track_history if t not in active_set]:
            del self._track_history[tid]

        return result

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Overlay counting line, centroids, global IDs, and totals."""
        h, w   = frame.shape[:2]
        line_x = self.line_y if self.line_y else w // 2

        # Counting line (vertical)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
        cv2.putText(
            frame, "COUNT LINE",
            (line_x + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )

        # Per-track centroid + global ID label
        for state in self._track_history.values():
            color = (0, 200, 0) if not state.counted else (0, 100, 255)
            cv2.circle(frame, state.centroid, 5, color, -1)
            label = (
                f"G:{state.global_id[:6]}"
                if state.global_id else
                f"T:{state.track_id}"
            )
            cv2.putText(
                frame, label,
                (state.centroid[0] + 6, state.centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

        # Running totals
        cv2.putText(frame, f"ENTRIES : {self.total_entries}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"EXITS   : {self.total_exits}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ReID stats overlay
        if self.use_reid and self.reid:
            stats = self.reid.get_stats()
            cv2.putText(
                frame,
                f"ReID unique:{stats['total_unique']} dup:{stats['total_duplicate']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

        return frame
