"""
main.py — Smart Count Tramway
================================
Master execution loop that orchestrates:
  1. Video stream acquisition (file, RTSP, or webcam)
  2. Per-frame face blurring  (privacy — "at the edge")
  3. YOLOv8 person detection
  4. DeepSORT multi-object tracking
  5. Virtual-line crossing → ENTRY / EXIT determination
  6. Timestamped SQLite storage every N seconds
  7. Live preview window (optional — disable on headless Pi)

Usage examples
--------------
  # Process a local test video file:
  python main.py --source data/videos/test_clip.mp4 --stop Kharouba

  # Live RTSP camera stream:
  python main.py --source rtsp://192.168.1.100:554/stream1 --stop Salamandre

  # Webcam (index 0):
  python main.py --source 0 --stop "Gare SNTF"

  # Headless mode (no display window — for Raspberry Pi / Jetson):
  python main.py --source rtsp://... --stop Kharouba --headless

Author  : Smart Count Tramway Team
Hardware: AMD Ryzen (dev) | Raspberry Pi 4 / Jetson Nano (prod)
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from loguru import logger

# ── Project imports ───────────────────────────────────────────────────────────
from src.detection import PersonDetector
from src.tracking  import PassengerTracker
from src.database  import Database


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults  (override via CLI args or a .env file)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH   = "models/yolov8n.pt"
DEFAULT_DB_PATH      = "data/tramway_counts.db"
DEFAULT_CONFIDENCE   = 0.50
DEFAULT_LINE_Y       = None       # None = auto-set to frame midpoint
DEFAULT_SAVE_EVERY   = 30         # Flush counts to DB every N seconds
DEFAULT_DISPLAY_FPS  = True       # Overlay FPS counter on preview


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Count Tramway — Automatic passenger counter"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: file path, RTSP URL, or webcam index (default: 0)"
    )
    parser.add_argument(
        "--stop", type=str, default="Kharouba",
        help="Tram stop name for DB labelling (default: Kharouba)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"YOLOv8 weights file (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE,
        help="Detection confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--line-y", type=int, default=DEFAULT_LINE_Y,
        help="Y-pixel of counting line; auto-set to mid-frame if omitted"
    )
    parser.add_argument(
        "--save-every", type=int, default=DEFAULT_SAVE_EVERY,
        help=f"Flush counts to DB every N seconds (default: {DEFAULT_SAVE_EVERY})"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Disable the live preview window (use on Pi / Jetson without display)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Draw raw detection bboxes (privacy warning — dev only)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown handler
# ─────────────────────────────────────────────────────────────────────────────

_running = True

def _handle_sigint(sig, frame):
    """Allow Ctrl+C to flush remaining counts and close resources cleanly."""
    global _running
    logger.info("[Main] SIGINT received — shutting down gracefully…")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _running
    args = parse_args()

    # ── Logger setup ──────────────────────────────────────────────────────
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="DEBUG" if args.debug else "INFO",
    )
    logger.add("data/tramway.log", rotation="10 MB", retention="7 days")

    logger.info("=" * 55)
    logger.info("  Smart Count Tramway — Starting")
    logger.info(f"  Source  : {args.source}")
    logger.info(f"  Stop    : {args.stop}")
    logger.info(f"  Model   : {args.model}")
    logger.info("=" * 55)

    # ── Initialise components ─────────────────────────────────────────────
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    tracker  = PassengerTracker(line_y=args.line_y)
    db       = Database(db_path=args.db)

    # ── Open video stream ─────────────────────────────────────────────────
    # Convert webcam index "0" (string from CLI) to int
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"[Main] Cannot open source: {args.source}")
        sys.exit(1)

    # Read frame dimensions to configure the tracker's line position
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0

    logger.info(f"[Main] Stream opened: {frame_width}×{frame_height} @ {fps_src:.1f} FPS")

    # ── Per-window counters (flushed to DB every save_every seconds) ──────
    window_entries = 0
    window_exits   = 0
    last_save_time = time.time()

    # ── FPS tracking ──────────────────────────────────────────────────────
    frame_count = 0
    fps_display = 0.0
    fps_timer   = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # Frame loop
    # ─────────────────────────────────────────────────────────────────────
    while _running:
        ret, frame = cap.read()
        if not ret:
            logger.info("[Main] End of video stream — exiting loop.")
            break

        frame_count += 1

        # ── Step 1: Privacy — blur all faces BEFORE any processing ───────
        # This is the "at the edge" anonymisation requirement (§3.2).
        # The original frame never persists; only the blurred version is used.
        frame = detector.blur_faces(frame)

        # ── Step 2: Person detection ──────────────────────────────────────
        detections = detector.detect(frame)

        # ── Step 3: Multi-object tracking + line crossing ─────────────────
        result = tracker.update(frame, detections)

        # ── Step 4: Accumulate window counts ─────────────────────────────
        window_entries += result.entries
        window_exits   += result.exits

        # ── Step 5: Flush to DB on interval ──────────────────────────────
        now = time.time()
        if now - last_save_time >= args.save_every:
            if window_entries > 0 or window_exits > 0:
                db.insert_count(
                    stop_name=args.stop,
                    entries=window_entries,
                    exits=window_exits,
                    timestamp=datetime.now(timezone.utc),
                )
                logger.info(
                    f"[Main] Saved → stop={args.stop}  "
                    f"entries={window_entries}  exits={window_exits}"
                )
            # Reset window counters
            window_entries = 0
            window_exits   = 0
            last_save_time = now

        # ── Step 6: FPS calculation (rolling over 30 frames) ─────────────
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps_display = 30 / elapsed if elapsed > 0 else 0.0
            fps_timer = time.time()

        # ── Step 7: Live preview (skip in headless mode) ──────────────────
        if not args.headless:
            # Draw tracking overlays (counting line, centroids, totals)
            display = tracker.draw_ui(frame.copy())

            # Debug: raw detection boxes (dev only — never in production)
            if args.debug:
                display = detector.draw_detections(display, detections)

            # FPS counter
            if DEFAULT_DISPLAY_FPS:
                cv2.putText(
                    display, f"FPS: {fps_display:.1f}",
                    (frame_width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )

            cv2.imshow("Smart Count Tramway", display)

            # Press 'q' to quit interactively
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("[Main] 'q' pressed — shutting down.")
                break

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    # Flush any remaining counts in the current window
    if window_entries > 0 or window_exits > 0:
        db.insert_count(
            stop_name=args.stop,
            entries=window_entries,
            exits=window_exits,
            timestamp=datetime.now(timezone.utc),
        )
        logger.info(f"[Main] Final flush → entries={window_entries} exits={window_exits}")

    cap.release()
    cv2.destroyAllWindows()
    db.close()

    logger.info("─" * 55)
    logger.info(f"  Session total — ENTRIES : {tracker.total_entries}")
    logger.info(f"  Session total — EXITS   : {tracker.total_exits}")
    logger.info("  Smart Count Tramway — Stopped.")
    logger.info("─" * 55)


if __name__ == "__main__":
    main()
