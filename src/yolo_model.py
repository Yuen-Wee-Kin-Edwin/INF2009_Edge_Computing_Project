# File: src/yolo_model.py
from ultralytics import YOLO
import os
import cv2
import numpy as np


class Detector:
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        # Load a YOLO26m PyTorch model
        # For the lower end used "yolo26n.pt"
        # self.model = YOLO("models/yolo26m.pt")
        self.model = YOLO("models/yolo26n.pt")

    def get_model(self) -> YOLO:
        return self.model

    def detect_frame(self, frame_bytes: bytes, annotate: bool = True):
        """
        Run YOLO detection on a single camera frame.

        Args:
            frame_bytes (bytes): JPEG-encoded frame from camera.
            annotate (bool): Whether to return an annotated frame with bounding boxes.

        Returns:
            results (ultralytics.engine.results.Results): YOLO detection results object.
            annotated_frame (np.ndarray | None): OpenCV frame with bounding boxes (if annotate=True).
        """
        # Convert JPEG bytes to OpenCV image
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None

        # Resize to smaller resolution for faster inference.
        frame_small = cv2.resize(frame, (640, 360))
        # Run YOLO detection
        results = self.model(frame_small)  # Returns list of Results objects.

        # Annotate frame if requested.
        annotated_frame = results[0].plot() if annotate else None
        return results[0], annotated_frame
