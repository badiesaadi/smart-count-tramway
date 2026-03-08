"""
tracking.py — Smart Count Tramway
===================================
Implements multi-object tracking using DeepSORT and determines
passenger direction (ENTRY vs EXIT) using virtual counting lines.

── How Virtual Line Crossing Works ─────────────────────────────────────────

  We define a horizontal line across the camera frame at a configurable
  Y-coordinate (line_y).  For each tracked object we compare the
  Y-position of its centroid in the PREVIOUS frame vs the CURRENT frame:

       ┌─────────────────────────────────────────┐
       │                  CAMERA                  │
       │  (top of frame = tram interior / door)   │
       │                                          │
       │  ──────── virtual line (line_y) ──────── │  ← passenger crosses here
       │                                          │
       │  (bottom of frame = platform / outside)  │
       └─────────────────────────────────────────┘

  Cross from TOP → BOTTOM  (y increases):  movement toward platform = EXIT
  Cross from BOTTOM → TOP  (y decreases):  movement into tram       = ENTRY

  Mathematically:
      prev_y < line_y  and  curr_y >= line_y   →  EXIT  (downward crossing)
      prev_y > line_y  and  curr_y <= line_y   →  ENTRY (upward  crossing)

  This is robust because DeepSORT maintains object identity across frames,
  so prev_y is always the same physical person as curr_y.

Author  : Smart Count Tramway Team
Hardware: AMD Ryzen (dev) | Raspberry Pi 4 / Jetson Nano (prod)
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from loguru import logger
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    """Passenger movement direction relative to the tram."""
    ENTRY = "ENTRY"    # Passenger boards the tram
    EXIT  = "EXIT"     # Passenger leaves the tram


@dataclass
class TrackState:
    """Stores the last known centroid Y for a single tracked object."""
    track_id : int
    centroid  : Tuple[int, int]   # (cx, cy) in pixels
    counted   : bool = False      # True once this track has triggered a count


@dataclass
class CountResult:
    """Returned by PassengerTracker.update() each frame."""
    entries    : int = 0          # New entries this frame
    exits      : int = 0          # New exits this frame
    active_ids : List[int] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker Class
# ─────────────────────────────────────────────────────────────────────────────

class PassengerTracker:
    """
    Wraps DeepSORT and adds virtual-line crossing logic.

    Parameters
    ----------
    line_y : int | None
        Pixel Y-coordinate of the virtual counting line.
        If None, defaults to the vertical midpoint of the frame.
    max_age : int
        Number of frames DeepSORT keeps a track alive without a detection.
        Higher → fewer ID switches but more ghost tracks.
    embedder : str
        Feature extractor used by DeepSORT for re-identification.
        'mobilenet' is fastest; 'torchreid' is more accurate.
    """

    def __init__(
        self,
        line_y: int = None,
        max_age: int = 30,
        embedder: str = "mobilenet",
    ) -> None:

        self.line_y = line_y   # Set to None initially; resolved on first frame

        # DeepSORT configuration
        # n_init=3 : a track is confirmed only after 3 consecutive detections
        #            — prevents single-frame noise from generating counts.
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            embedder=embedder,
            half=True,        # FP16 inference — significant speedup on CUDA/Jetson
            bgr=True,         # Our frames are OpenCV BGR format
        )

        # Dictionary mapping track_id → TrackState for centroid history
        self._track_history: Dict[int, TrackState] = {}

        # Running totals for the full session
        self.total_entries = 0
        self.total_exits   = 0

        logger.info(f"[Tracker] DeepSORT initialised (max_age={max_age}, embedder={embedder})")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections,               # List[Detection] from detection.py
    ) -> CountResult:
        """
        Feed one frame into the tracker and return any new counts.

        Pipeline per frame:
          1. Convert Detection objects → DeepSORT input format.
          2. Run DeepSORT .update_tracks() — assigns consistent IDs.
          3. For each confirmed track, check if its centroid crossed
             the virtual line since the last frame.
          4. Increment ENTRY / EXIT counters when a crossing is detected.

        Parameters
        ----------
        frame      : np.ndarray   BGR image (H × W × 3)
        detections : List[Detection]  from PersonDetector.detect()

        Returns
        -------
        CountResult  with entries/exits detected IN THIS FRAME ONLY.
        """
        h, w = frame.shape[:2]

        # ── Resolve line_y lazily (first frame) ──────────────────────────
        if self.line_y is None:
            self.line_y = h // 2
            logger.info(f"[Tracker] Virtual counting line set at y={self.line_y}px")

        # ── Format detections for DeepSORT ───────────────────────────────
        # DeepSORT expects a list of ([left, top, w, h], confidence, class)
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, "person"))

        # ── Run tracker ──────────────────────────────────────────────────
        tracks = self.tracker.update_tracks(raw, frame=frame)

        result = CountResult()

        for track in tracks:
            # Skip tentative tracks — only process confirmed ones
            if not track.is_confirmed():
                continue

            tid = track.track_id
            ltrb = track.to_ltrb()   # [left, top, right, bottom]
            x1, y1, x2, y2 = map(int, ltrb)

            # Centroid = geometric centre of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            result.active_ids.append(tid)

            # ── Direction logic ──────────────────────────────────────────
            if tid in self._track_history:
                prev_cy = self._track_history[tid].centroid[1]

                # A crossing is detected when the centroid passes line_y
                # between consecutive frames.  We use prev_cy and cy to
                # detect the transition, and mark the track as counted so
                # the same person cannot trigger multiple counts.

                if not self._track_history[tid].counted:

                    crossed_down = prev_cy < self.line_y and cy >= self.line_y
                    crossed_up   = prev_cy > self.line_y and cy <= self.line_y

                    if crossed_down:
                        # Centroid moved downward (toward platform) → EXIT
                        result.exits += 1
                        self.total_exits += 1
                        self._track_history[tid].counted = True
                        logger.debug(f"[Tracker] Track {tid} → EXIT  (y: {prev_cy}→{cy})")

                    elif crossed_up:
                        # Centroid moved upward (into tram) → ENTRY
                        result.entries += 1
                        self.total_entries += 1
                        self._track_history[tid].counted = True
                        logger.debug(f"[Tracker] Track {tid} → ENTRY (y: {prev_cy}→{cy})")

            # Update centroid history for this track
            self._track_history[tid] = TrackState(
                track_id=tid, centroid=(cx, cy),
                counted=self._track_history.get(tid, TrackState(tid, (cx, cy))).counted,
            )

        # ── Prune dead tracks ─────────────────────────────────────────────
        active_set = set(result.active_ids)
        dead = [tid for tid in self._track_history if tid not in active_set]
        for tid in dead:
            del self._track_history[tid]

        return result

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay counting line, centroid dots, and running totals on the frame.
        Safe to call in both dev and production (no raw faces shown).
        """
        h, w = frame.shape[:2]
        line_y = self.line_y if self.line_y else h // 2

        # ── Virtual counting line ─────────────────────────────────────────
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(
            frame, "COUNTING LINE",
            (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )

        # ── Centroid dots ─────────────────────────────────────────────────
        for state in self._track_history.values():
            color = (0, 200, 0) if not state.counted else (0, 100, 255)
            cv2.circle(frame, state.centroid, 5, color, -1)
            cv2.putText(
                frame, str(state.track_id),
                (state.centroid[0] + 6, state.centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # ── Running totals overlay ────────────────────────────────────────
        overlay_text = [
            (f"ENTRIES : {self.total_entries}", (10, 30), (0, 255, 0)),
            (f"EXITS   : {self.total_exits}",   (10, 60), (0, 0, 255)),
        ]
        for text, pos, color in overlay_text:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame
