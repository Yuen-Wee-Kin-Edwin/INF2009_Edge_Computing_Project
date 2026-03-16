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
        self.face_recogniser = FaceRecogniser()

    def get_model(self) -> YOLO:
        return self.model

    def detect_frame(self, frame: np.ndarray, annotate: bool = True):
        """
        Run YOLO detection on a single camera frame matrix.

        Args:
            frame (np.ndarray): OpenCV image matrix.
            annotate (bool): Whether to return an annotated frame with bounding boxes.

        Returns:
            results (ultralytics.engine.results.Results): YOLO detection results object.
            annotated_frame (np.ndarray | None): OpenCV frame with bounding boxes (if annotate=True).
            face_results (list): List of recognised faces and coordinates.
        """
        # Validate the incoming matrix
        if frame is None or not isinstance(frame, np.ndarray):
            print("[YOLO] Error: Invalid frame matrix passed to detector.")
            return None, None, None

        # Resize to smaller resolution for faster inference.
        frame_small = cv2.resize(frame, (640, 360))

        # Run YOLO detection
        yolo_results = self.model(frame_small)  # Returns list of Results objects.

        # Annotate frame if requested.
        annotated_frame = yolo_results[0].plot() if annotate else frame_small.copy()

        person_detected = False
        face_results = []

        # Parse YOLO results to verify if a 'person' (class 0) is in the frame.
        for box in yolo_results[0].boxes:
            if int(box.cls[0]) == 0:
                person_detected = True
                break

        # Only trigger facial recognition if a human is present.
        if person_detected:
            # Run face recognition on the same frame.
            face_results = self.face_recogniser.recognise(frame_small)

            # Draw custom face boxes and labels over the YOLO annotations.
            for face in face_results:
                left, top, right, bottom = face["box"]
                name = face["name"]
                confidence = face["confidence"]

                # Use red for unknown instruders, green for recognised faces
                colour = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), colour, 2)

                label = f"{name} ({confidence:.2f})"
                cv2.putText(
                    annotated_frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                )

        else:
            # Return a specific flag if YOLO did not see a person, so app.py known to ignore it.
            return yolo_results[0], annotated_frame, "NO_PERSON"

        return yolo_results[0], annotated_frame, face_results
