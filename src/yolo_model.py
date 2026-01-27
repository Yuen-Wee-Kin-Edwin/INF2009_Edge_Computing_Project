# File: src/yolo_model.py
from ultralytics import YOLO
import os
import cv2
import numpy as np
from face_recogniser import FaceRecogniser


class Detector:
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        # Load a YOLO26m PyTorch model
        # For the lower end used "yolo26n.pt"
        # self.model = YOLO("models/yolo26m.pt")
        self.model = YOLO("models/yolo26n.pt")

        # Initialise face recogniser
        self.face_recogniser = FaceRecogniser("known_faces")

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
            face_results
        """
        # Convert JPEG bytes to OpenCV image
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None

        # Resize to smaller resolution for faster inference.
        frame_small = cv2.resize(frame, (640, 360))

        # Run YOLO detection
        yolo_results = self.model(frame_small)  # Returns list of Results objects.

        # Annotate frame if requested.
        annotated_frame = yolo_results[0].plot() if annotate else frame_small.copy()

        # Run face recognition on the same frame.
        face_results = self.face_recogniser.recognise(frame_small)

        # Fraw face boxes and labels.
        for face in face_results:
            left, top, right, bottom = face["box"]
            name = face["name"]
            confidence = face["confidence"]

            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            label = f"{name} ({confidence:.2f})"
            cv2.putText(
                annotated_frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return yolo_results[0], annotated_frame, face_results
