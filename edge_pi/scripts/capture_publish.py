#!/usr/bin/env python3
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import os
import socket
import json
from datetime import datetime
import numpy as np
import ai_edge_litert.interpreter as tflite

# Load MobileNet
MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
# In the standard COCO dataset, 'person' is class 0
PERSON_CLASS_INDEX = 0 
MIN_CONFIDENCE = 0.60  # Require 60% confidence to trigger a save

# Camera & Capture
MAX_SNAPSHOTS = 5
CAMERA_INDEX = 0
ROI_COORDS = (0, 720, 0, 1280) # y1, y2, x1, x2
CAPTURE_INTERVAL = 5

# Dynamically acquire the device hostname (e.g., 'edge-camera-01')
CLIENT_ID = socket.gethostname()

# Define the location/lab. Ideally, load this from an envrionment variable
# to keep the script entirely generic across all devices.
LOCATION = "sit"
LAB_ID = os.environ.get("LAB_ID", "lab_default")

# Construct the scalable, strictly formatted MQTT topic.
# Example output: sit/lab_default/edge-camera-01/vision/person
MQTT_TOPIC = f"{LOCATION}/{LAB_ID}/{CLIENT_ID}/vision/person"

# MQTT Settings
MQTT_BROKER = "192.168.137.98"
MQTT_PORT = 1883
MQTT_USER = "edwin"
MQTT_PASS = "password"

# Connection Callbacks
def on_connect(client, userdata, flags, reason_code, properties):
    """Callback triggered when the edge connects to the hub."""
    if reason_code == 0:
        print(f"Edge successfully connected to the hub at {MQTT_BROKER}")
    else:
        print(f"Connection to hub failed with return code {reason_code}")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    """Callback triggered on unexpected disconnections."""
    print("Warning: Unexpected disconnection from hub. Reconnecting...")

# ------------------------------
# Initialise Connection
# ------------------------------
# Instantiate the client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Set authentication credentials
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)

# Bind the callbacks to the client
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

print(f"Attempting to connect to MQTT hub at {MQTT_BROKER}...")
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # loop_start() spawns a daemon thread.
    # It manages ping requests, network traffic, and automatic reconnections in the background
    mqtt_client.loop_start()

except Exception as e:
    print(f"Critical error: Could not establish initial connection to the hub. {e}")
    exit(1)

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
            filename = f"{CLIENT_ID}_snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)

            # Save to disk locally
            cv2.imwrite(filepath, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Snapshot saved as {filepath}")

            # Manage local snapshot retention
            snapshots = sorted(
                [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
                key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x))
            )

            while len(snapshots) > MAX_SNAPSHOTS:
                oldest = snapshots.pop(0)
                os.remove(os.path.join(SNAPSHOT_DIR, oldest))
                print(f"Deleted oldest snapshot: {oldest}")

            # 6. Construct and Publish MQTT Payload.
            # Encode image matrix directly to a JPEG buffer in memory.
            success, buffer = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if success:
                # Convert the binary buffer to a base64 string.
                b64_string = base64.b64encode(buffer).decode('utf-8')

                # Construct a structured JSON payload with relevant metadata.
                payload = {
                        "camera_id": CLIENT_ID,
                        "location": LOCATION,
                        "lab_id": LAB_ID,
                        "timestamp": timestamp,
                        "confidence": float(confidence_pct),
                        "image": b64_string
                }

                # Serialise the dictionary to a JSON string and publish
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                print(f"Published detection payload to {MQTT_TOPIC}")
            else:
                print("Error: Failed to encode image to memory buffer.")
        else:
            print(f"[{datetime.now()}] Clear. No person detected.")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping capture...")

finally:
    cap.release()
    # Stop the background network thread and sever the connection cleanly.
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
