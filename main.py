"""
main.py — Smart Count Tramway
================================
Master execution loop.

New in this version:
  --no-reid        : Disable global ReID (faster, for testing)
  --reid-threshold : Cosine similarity threshold for ReID matching (default 0.75)
  --reid-ttl       : Seconds before a ReID entry expires (default 300)

Usage examples
--------------
  py -3.12 main.py --source data/videos/test.mp4 --stop Kharouba --debug
  py -3.12 main.py --source 0 --stop Salamandre --headless
  py -3.12 main.py --source rtsp://192.168.1.100:554/s1 --stop "Gare SNTF" --headless
  py -3.12 main.py --source 0 --stop Kharouba --no-reid   # disable ReID

Author  : Smart Count Tramway Team
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from loguru import logger

from src.detection import PersonDetector
from src.tracking  import PassengerTracker
from src.database  import Database

DEFAULT_MODEL_PATH  = "models/yolov8n.pt"
DEFAULT_DB_PATH     = "data/tramway_counts.db"
DEFAULT_CONFIDENCE  = 0.50
DEFAULT_LINE_Y      = None
DEFAULT_SAVE_EVERY  = 30

_running = True

def _handle_sigint(sig, frame):
    global _running
    logger.info("[Main] SIGINT — shutting down...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Count Tramway")
    parser.add_argument("--source",         type=str,   default="0")
    parser.add_argument("--stop",           type=str,   default="Kharouba")
    parser.add_argument("--model",          type=str,   default=DEFAULT_MODEL_PATH)
    parser.add_argument("--db",             type=str,   default=DEFAULT_DB_PATH)
    parser.add_argument("--conf",           type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--line-y",         type=int,   default=DEFAULT_LINE_Y)
    parser.add_argument("--save-every",     type=int,   default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--headless",       action="store_true")
    parser.add_argument("--debug",          action="store_true")
    # ReID arguments
    parser.add_argument("--no-reid",        action="store_true",
                        help="Disable global ReID (faster, less accurate)")
    parser.add_argument("--reid-threshold", type=float, default=0.75,
                        help="Cosine similarity threshold for ReID (default 0.75)")
    parser.add_argument("--reid-ttl",       type=int,   default=300,
                        help="Seconds before ReID entry expires (default 300)")
    return parser.parse_args()


def main() -> None:
    global _running
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="DEBUG" if args.debug else "INFO",
    )
    logger.add("data/tramway.log", rotation="10 MB", retention="7 days")

    logger.info("=" * 55)
    logger.info("  Smart Count Tramway — Starting")
    logger.info(f"  Source    : {args.source}")
    logger.info(f"  Stop      : {args.stop}")
    logger.info(f"  Model     : {args.model}")
    logger.info(f"  ReID      : {'DISABLED' if args.no_reid else f'ENABLED (threshold={args.reid_threshold})'}")
    logger.info("=" * 55)

    detector = PersonDetector(
        model_path            = args.model,
        confidence_threshold  = args.conf,
    )
    tracker = PassengerTracker(
        stop_name      = args.stop,
        line_y         = args.line_y,
        reid_threshold = args.reid_threshold,
        reid_ttl       = args.reid_ttl,
        use_reid       = not args.no_reid,
    )
    db = Database(db_path=args.db)

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"[Main] Cannot open source: {args.source}")
        sys.exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0

    logger.info(f"[Main] Stream: {frame_width}×{frame_height} @ {fps_src:.1f} FPS")

    window_entries = 0
    window_exits   = 0
    last_save_time = time.time()
    frame_count    = 0
    fps_display    = 0.0
    fps_timer      = time.time()

    while _running:
        ret, frame = cap.read()
        if not ret:
            logger.info("[Main] End of stream.")
            break

        frame_count += 1

        # Privacy: blur faces before any processing
        frame = detector.blur_faces(frame)

        # Detect persons
        detections = detector.detect(frame)

        # Track + ReID + count
        result = tracker.update(frame, detections)

        window_entries += result.entries
        window_exits   += result.exits

        # Periodic DB flush
        now = time.time()
        if now - last_save_time >= args.save_every:
            if window_entries > 0 or window_exits > 0:
                db.insert_count(
                    stop_name = args.stop,
                    entries   = window_entries,
                    exits     = window_exits,
                    timestamp = datetime.now(timezone.utc),
                )
                logger.info(
                    f"[Main] Saved → stop={args.stop} "
                    f"entries={window_entries} exits={window_exits}"
                )
            window_entries = 0
            window_exits   = 0
            last_save_time = now

        # FPS calculation
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            fps_display = 30 / elapsed if elapsed > 0 else 0.0
            fps_timer   = time.time()

        # Live preview
        if not args.headless:
            display = tracker.draw_ui(frame.copy())
            if args.debug:
                display = detector.draw_detections(display, detections)
            cv2.putText(
                display, f"FPS: {fps_display:.1f}",
                (frame_width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
            cv2.imshow("Smart Count Tramway", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("[Main] 'q' pressed — stopping.")
                break

    # Final flush
    if window_entries > 0 or window_exits > 0:
        db.insert_count(
            stop_name = args.stop,
            entries   = window_entries,
            exits     = window_exits,
            timestamp = datetime.now(timezone.utc),
        )

    cap.release()
    cv2.destroyAllWindows()
    db.close()

    logger.info("─" * 55)
    logger.info(f"  Session ENTRIES : {tracker.total_entries}")
    logger.info(f"  Session EXITS   : {tracker.total_exits}")
    if not args.no_reid and tracker.reid:
        stats = tracker.reid.get_stats()
        logger.info(f"  ReID unique     : {stats['total_unique']}")
        logger.info(f"  ReID duplicates : {stats['total_duplicate']}")
    logger.info("  Smart Count Tramway — Stopped.")
    logger.info("─" * 55)


if __name__ == "__main__":
    main()
