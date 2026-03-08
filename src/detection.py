"""
detection.py — Smart Count Tramway
===================================
Handles all inference using YOLOv8 for:
  1. Person detection  (COCO class_id = 0)
  2. Face detection    (OpenCV Haar cascade — ultra-lightweight for edge)
  3. "At the edge" face blurring for GDPR/privacy compliance

The model path is fully configurable so weights can be swapped
(e.g., swap yolov8n.pt → yolov8m.pt for more accuracy) without
touching any logic code.

Author  : Smart Count Tramway Team
Hardware: AMD Ryzen (dev) | Raspberry Pi 4 / Jetson Nano (prod)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Represents a single detected person bounding box."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                  # Confidence score in [0, 1]
    class_id: int                      # COCO class id  (0 = person)


# ─────────────────────────────────────────────────────────────────────────────
# Detector Class
# ─────────────────────────────────────────────────────────────────────────────

class PersonDetector:
    """
    Wraps YOLOv8 inference for person detection and integrates
    at-the-edge face anonymisation.

    Parameters
    ----------
    model_path : str
        Path to YOLOv8 .pt weights file.
        Default → 'models/yolov8n.pt'  (nano: fastest, best for edge devices)
        Swap  →  'models/yolov8m.pt'  for higher accuracy on a workstation.
    confidence_threshold : float
        Minimum score to keep a detection.  0.50 recommended.
    device : str | None
        'cpu', 'cuda', or 'mps'. Auto-detected when None.
    """

    PERSON_CLASS_ID = 0   # COCO dataset label id for "person"

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:

        # ── Device selection ──────────────────────────────────────────────
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[Detector] Running on device : {self.device}")

        # ── Load YOLOv8 person-detection model ────────────────────────────
        # ultralytics.YOLO accepts any .pt file, so swapping weights is
        # as simple as changing the model_path argument at construction time.
        logger.info(f"[Detector] Loading model from : {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

        # ── Load face cascade for privacy blurring ────────────────────────
        # OpenCV's Haar cascade is ~600 KB and runs fast even on a Pi 4.
        # No GPU required — ideal for the "at the edge" privacy requirement.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            logger.warning("[Detector] Haar cascade not found — face blurring disabled.")
            self.face_cascade = None
        else:
            logger.info("[Detector] Face anonymisation cascade loaded ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # Public Methods
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV  (H × W × 3).

        Returns
        -------
        List[Detection]
            Every person bounding box above the confidence threshold.
        """
        results = self.model(
            frame,
            verbose=False,                       # suppress per-frame console spam
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],      # only detect persons — faster
        )

        detections: List[Detection] = []

        # results[0].boxes holds all detections for the first (and only) image
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append(
                Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls)
            )

        return detections

    def blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect all faces and replace them with Gaussian blur.

        Privacy principle: blurring is applied BEFORE any data is stored
        or sent over the network, so raw face pixels never leave the device.
        This satisfies the "anonymisation at the edge" requirement in §3.2 of
        the Cahier des Charges.

        The method modifies the frame IN-PLACE and also returns it so it
        can be chained in a single-line pipeline.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (modified in-place).

        Returns
        -------
        np.ndarray
            Same frame with all detected faces blurred.
        """
        if self.face_cascade is None:
            return frame

        # Haar cascade works on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # image pyramid step: 1.1 = 10 % reduction each level
            minNeighbors=5,    # higher → fewer false positives
            minSize=(30, 30),  # ignore tiny faces (noise / distant passengers)
        )

        for (fx, fy, fw, fh) in faces:
            face_roi = frame[fy : fy + fh, fx : fx + fw]

            # (99, 99) kernel + sigmaX=30 → strong, practically irreversible blur.
            # Kernel size MUST be odd — this is a hard OpenCV requirement.
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[fy : fy + fh, fx : fx + fw] = blurred_face

        return frame

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Overlay bounding boxes on the frame.

        ⚠️  For DEBUG / DEVELOPMENT ONLY — do NOT enable in production.
        In production the face-blurred frame is displayed, not raw bboxes.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {det.confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return frame
