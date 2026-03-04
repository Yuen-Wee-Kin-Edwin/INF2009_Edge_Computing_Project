#!/usr/bin/env python3
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import os
from datetime import datetime
import numpy as np
import ai_edge_litert.interpreter as tflite

# Load MobileNet
MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
PERSON_CLASS_INDEX = 15 # ImageNet 'person'

# In the standard COCO dataset, 'person' is class 0
PERSON_CLASS_INDEX = 0 
MIN_CONFIDENCE = 0.60  # Require 60% confidence to trigger a save

MAX_SNAPSHOTS = 5
CAMERA_INDEX = 0
ROI_COORDS = (0, 720, 0, 1280) # y1, y2, x1, x2
CAPTURE_INTERVAL = 5

# ------------------------------
# Initialise Environment
# ------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Initialise the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dynamically extract the required input resolution (typically 300x300 for SSD)
INPUT_HEIGHT = input_details[0]['shape'][1]
INPUT_WIDTH = input_details[0]['shape'][2]

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting continuous object detection. Press Ctrl+C to stop.")

# ------------------------------
# Execution Loop
# ------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # 1. Crop Region of Interest.
        y1, y2, x1, x2 = ROI_COORDS
        roi = frame[y1:y2, x1:x2]

        # 2. Pre-process for MobileNet SSD.
        roi_resized = cv2.resize(roi, (INPUT_WIDTH, INPUT_HEIGHT))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(roi_rgb, axis=0)

        # Normalise only if the model specifically demands float32.
        # The downloaded quantized model expects uint8, so this is bypassed.
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # 3. Execute Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 4. Parse the Multiple Output Tensors
        # Output 0: Bounding box coordinates [ymin, xmin, ymax, xmax]
        # Output 1: Class indices
        # Output 2: Confidence scores
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        person_detected = False

        # Iterate through the detections to find a confident person match
        for i in range(len(scores)):
            if scores[i] > MIN_CONFIDENCE and int(classes[i]) == PERSON_CLASS_INDEX:
                person_detected = True
                confidence_pct = scores[i] * 100
                print(f"[{datetime.now()}] Person detected! Confidence: {confidence_pct:.1f}%")
                # Optional: Extract bounding box coordinates here if you wish 
                # ymin, xmin, ymax, xmax = boxes[i]
                break

        # 5. Handle Detections
        if person_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cam1_snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)

            cv2.imwrite(filepath, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Snapshot saved as {filepath}")

            snapshots = sorted(
                [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
                key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x))
            )

            while len(snapshots) > MAX_SNAPSHOTS:
                oldest = snapshots.pop(0)
                os.remove(os.path.join(SNAPSHOT_DIR, oldest))
                print(f"Deleted oldest snapshot: {oldest}")
        else:
            print(f"[{datetime.now()}] Clear. No person detected.")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping capture...")

finally:
    cap.release()
    cv2.destroyAllWindows()
