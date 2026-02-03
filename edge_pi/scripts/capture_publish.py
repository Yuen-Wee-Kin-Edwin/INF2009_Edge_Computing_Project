#!/usr/bin/env python3
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import os
from datetime import datetime

# ------------------------------
# Configuration
# -----------------------------
BROKER_IP = "<MAIN_PI_IP>" # Replace with main Pi's IP
TOPIC = "edge-cam1/snapshot"
CAPTURE_INTERVAL = 5 # Seconds between snapshots
MAX_SNAPSHOTS = 5

# ------------------------------
# Folder for snapshots.
# ------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
os.makedirs(SNAPSHOT_DIR, exist_ok=True) # Create folder if missing.

# Open the default webcam
cap = cv2.VideoCapture(0)
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
        if ret:
            # Timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cam1_snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)

            # Save the snapshot.
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Snapshot saved as {filepath}")

            # Manage snapshot folder.
            snapshots = sorted(
                [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
                key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x))
            )

            # Delete oldest if more than MAX_SNAPSHOTS
            while len(snapshots) > MAX_SNAPSHOTS:
                oldest = snapshots.pop(0)
                os.remove(os.path.join(SNAPSHOT_DIR, oldest))
                print(f"Deleted oldest snapshot: {oldest}")

        else:
            print("Error: Failed to capture frame.")
        
        # Wait for next capture
        time.sleep(CAPTURE_INTERVAL)


except KeyboardInterrupt:
    print("\nStopping capture...")

finally:
    cap.release()
