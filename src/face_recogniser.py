import face_recognition
import numpy as np
import os
import cv2


class FaceRecogniser:
    def __init__(self, known_faces_dir="known_faces"):
        """
        Load known faces from disk and build embedding database.
        Each image filename should be the person's name.
        Example: alice.jpg, bob.png
        """
        self.known_encodings = []
        self.known_names = []

        # Resolve absolute path relative to this file (src directory)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        known_faces_dir = os.path.join(base_dir, known_faces_dir)

        # Create directory if it does not exist.
        os.makedirs(known_faces_dir, exist_ok=True)

        for file in os.listdir(known_faces_dir):
            # Skip non-image files.
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            name = os.path.splitext(file)[0]
            path = os.path.join(known_faces_dir, file)

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            # Skip images where no face was detected
            if len(encodings) == 0:
                print(f"[WARN] No face detected in {file}")
                continue

            self.known_encodings.append(encodings[0])
            self.known_names.append(name)

        print(f"[INFO] Loaded {len(self.known_names)} known faces")

    def recognise(self, frame_bgr):
        """
        Detect and recognise faces in a single OpenCV frame.

        Returns:
            List of dicts with name, confidence, and bounding box.
        """
        # Convert OpenCV BGR -> RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        results = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=0.5
            )

            name = "Unknown"
            confidence = 0.0

            if True in matches:
                idx = matches.index(True)
                name = self.known_names[idx]

                distances = face_recognition.face_distance(
                    self.known_encodings, encoding
                )
                confidence = float(1.0 - distances[idx])

            results.append(
                {
                    "name": name,
                    "confidence": round(confidence, 3),
                    "box": (left, top, right, bottom),
                }
            )

        return results
