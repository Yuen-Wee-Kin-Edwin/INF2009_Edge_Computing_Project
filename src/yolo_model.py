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


# # # --- Download a sample image ---
# os.makedirs("test_images", exist_ok=True)
# sample_image_path = "test_images/sample.jpg"

# if not os.path.exists(sample_image_path):
#     url = "https://ultralytics.com/images/bus.jpg"
#     print("Downloading sample image...")
#     urllib.request.urlretrieve(url, sample_image_path)
#     print("Download complete.")

# # --- Load YOLO model ---
# det = Detector()
# model = det.get_model()

# # --- Run prediction ---
# results_list = model.predict(sample_image_path)

# # --- Print detected objects ---
# print("\nDetections:")
# for result in results_list:
#     for box in result.boxes:
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         xyxy = box.xyxy[0].tolist()
#         print(f"Class {cls_id}, Confidence {conf:.2f}, Box {xyxy}")

# # --- Save annotated images manually ---
# os.makedirs("outputs", exist_ok=True)
# for i, result in enumerate(results_list):
#     annotated_img = result.plot()  # returns annotated image as numpy array
#     output_path = os.path.join("outputs", f"annotated_{i}.jpg")
#     cv2.imwrite(output_path, annotated_img)
#     print(f"Saved annotated image to {output_path}")

# print("\n? Test complete. Annotated images saved in 'outputs' folder.")
