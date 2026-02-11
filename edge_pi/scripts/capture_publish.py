#!/usr/bin/env python3
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import os
from datetime import datetime

# Load TFLite MobileNet
net = cv2.dnn.readNetFromTensorflow('mobilenet_v2_1.0_224.tflite')
PERSON_CLASS_INDEX = 15 # ImageNet 'person' class.


# ------------------------------
# Configuration
# -----------------------------
BROKER_IP = "<MAIN_PI_IP>" # Replace with main Pi's IP
TOPIC = "edge-cam1/snapshot"
MAX_SNAPSHOTS = 5
CAMERA_INDEX = 0 # Default webcam.
ROI_COORDS = (100, 600, 200, 1000) # y1, y2, x1, x2
MOBILENET_INPUT_SIZE = (224, 224) # MobileNet input.
CAPTURE_INTERVAL = 5 # Seconds between snapshots

# ------------------------------
# Folder for snapshots.
# ------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
os.makedirs(SNAPSHOT_DIR, exist_ok=True) # Create folder if missing.

# Open the default webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set 720p resolution (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting continuous capture. Press Ctrl+C to stop.")

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # --------------------------------------
        # Preprocessing Pipeline
        # --------------------------------------

        # 1. Crop ROI (restricted area)
        y1, y2, x1, x2 = ROI_COORDS
        roi = frame[y1:y2, x1:x2]

        # 2. Resize for YOLO26n and MobileNet.
        roi_resized = cv2.resize(roi, MOBILENET_INPUT_SIZE)

        # Convert BGR -> RGB for MobileNet
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------
        blob = cv2.dnn.blobFromImage(roi_rgb, 1/255.0, MOBILENET_INPUT_SIZE, swapRB=True)
        net.setInput(blob)
        output = net.forward()
        top_class = np.argmax(output)

        if top_class == PERSON_CLASS_INDEX:
            print(f"[{datetime.now()}] Person detected!")

            # --------------------------------------
            # 3. Save snapshot locally.
            # --------------------------------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cam1_snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)

            cv2.imwrite(filepath, roi_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Snapshot saved as {filepath}")

            # Manage snapshot folder.
            snapshots = sorted(
                [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
                key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x))
            )
            while len(snapshots) > MAX_SNAPSHOTS:
                oldest = snapshots.pop(0)
                os.remove(os.path.join(SNAPSHOT_DIR, oldest))
                print(f"Deleted oldest snapshot: {oldest}")
        else:
            print(f"[{datetime.now()}] No person detected, frame discarded.")

        # Wait for next capture.
        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping capture...")

finally:
    cap.release()
