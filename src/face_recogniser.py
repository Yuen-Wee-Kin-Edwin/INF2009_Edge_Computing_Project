import json

import face_recognition
import numpy as np
import os
import cv2

from db import Database


class FaceRecogniser:
    def __init__(self):
        """
        Initialises the recogniser and loads all known facial encodings
        directly from the SQLite database.
        """
        self.known_encodings = []
        self.known_names = []
        self._load_encodings_from_db()

    def _load_encodings_from_db(self):
        """
        Connects to the database, retrieves the serialised JSON encodings,
        and reconstructs them into NumPy arrays for mathematical comparison.
        """
        db = Database()
        try:
            with db.connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, encoding FROM authorised_faces")
                rows = cursor.fetchall()

                for row in rows:
                    name = row["name"]
                    # Parse the JSON string back into a standard Python list
                    encoding_list = json.loads(row["encoding"])
                    # Convert the list back into a NumPy array required by the face_recognition library
                    encoding_array = np.array(encoding_list)

                    self.known_names.append(name)
                    self.known_encodings.append(encoding_array)

            print(
                f"[INFO] Successfully loaded {len(self.known_names)} known face encodings from the database."
            )
        except Exception as e:
            print(f"[ERROR] Critical failure loading encodings from database: {e}")

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

        # If the database is empty, all detected faces are immediately flagged as Unknown
        if not self.known_encodings:
            for top, right, bottom, left in face_locations:
                results.append(
                    {
                        "name": "Unknown",
                        "confidence": 0.0,
                        "box": (left, top, right, bottom),
                    }
                )
            return results

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Compare the live face encoding against all database encodings
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=0.5
            )

            name = "Unknown"
            confidence = 0.0

            if True in matches:
                # Find the index of the first match to retrieve the corresponding name
                idx = matches.index(True)
                name = self.known_names[idx]

                # Calculate the mathematical distance to derive a confidence percentage
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

    def reload_database(self):
        """
        Clears the current arrays and reloads them from the database.
        Call this method whenever a new user is registered to update the system
        without requiring a full application restart.
        """
        print("[INFO] Reloading facial encodings from database...")
        self.known_encodings.clear()
        self.known_names.clear()
        self._load_encodings_from_db()
