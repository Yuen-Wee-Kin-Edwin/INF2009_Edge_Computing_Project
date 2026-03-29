#!/usr/bin/env python3
import cv2
import base64
import paho.mqtt.client as mqtt
import time
import os
import socket
import json
from datetime import datetime, timezone
import numpy as np
import ai_edge_litert.interpreter as tflite
from collections import deque
import threading
import queue
import subprocess

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

# The topic the edge device will listen to for commands
COMMAND_TOPIC = f"{LOCATION}/{LAB_ID}/{CLIENT_ID}/command"

# Global state flag to control the capture loop
camera_active = True

# MQTT Settings
MQTT_BROKER_DNS = "edwinpi.local"
MQTT_BROKER_FALLBACK_IP = "192.168.137.98"
MQTT_PORT = 1883
MQTT_USER = "edwin"
MQTT_PASS = "password"

# Initialise data structures for concurrency and memory management.
snapshot_tracker = deque(maxlen=MAX_SNAPSHOTS)
payload_queue = queue.Queue(maxsize=10)

# Network Callbacks & Workers
def on_connect(client, userdata, flags, reason_code, properties):
    """Callback triggered when the edge connects to the hub."""
    if reason_code == 0:
        print(f"Edge successfully connected to the hub at {MQTT_BROKER_DNS}")
        # Subscribe to the command topic immediately upon connection
        client.subscribe(COMMAND_TOPIC, qos=1)
        print(f"Subscribed to command topic: {COMMAND_TOPIC}")
    else:
        print(f"Connection to hub failed with return code {reason_code}")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    """Callback triggered on unexpected disconnections."""
    print("Warning: Unexpected disconnection from hub. Reconnecting...")

def on_publish(client, userdata, mid, reason_code=None, properties=None):
    """Verifies Qos 1 delivery."""
    print(f"[{datetime.now()}] Message ID {mid} acknowledged by broker (QoS 1).")

def on_message(client, userdata, msg):
    """Callback triggered when a command is received from the Pi 5."""
    global camera_active
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        action = payload.get("action")

        if action == "activate":
            if not camera_active:
                camera_active = True
                print(f"[{datetime.now()}] Camera activated via remote command.")
        elif action == "deactivate":
            if camera_active:
                camera_active = False
                print(f"[{datetime.now()}] Camera deactivated via remote command. Entering standby.")
        else:
            print(f"Warning: Unrecognised command received: {action}")
    
    except json.JSONDecodeError:
        print("Error: Received malformed JSON command.")
    except Exception as e:
        print(f"Error processing command: {e}")

def mqtt_worker_thread():
    """
    Background thread to process heavy encoding and network transmissions.
    Prevents the main camera thread from blocking.
    """
    while True:
        try:
            item = payload_queue.get()
            if item is None:
                break # Sentinel value to terminate thread.

            roi, metadata = item

            # Offload the heavy JPEG and Base64 encoding to this thread.
            success, buffer = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if success:
                b64_string = base64.b64encode(buffer).decode("utf-8")
                metadata["image"] = b64_string

                # Publish with QoS 1
                mqtt_client.publish(MQTT_TOPIC, json.dumps(metadata), qos=1)
            else:
                print("Error: Failed to encode image in worker thread.")

            payload_queue.task_done()
        except Exception as e:
            print(f"Worker thread error: {e}")


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
mqtt_client.on_publish = on_publish
mqtt_client.on_message = on_message

def get_cpu_temp():
    """
    Queries the Raspberry Pi hardware for the current CPU core temperature.
    Returns a float representing the temperature in Celsius.
    """
    try:
        # Executes the native Broadcom VideoCore command
        result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
        # Parses the output string (e.g., "temp=55.0'C") to extract the float.
        temp_str = result.stdout.replace("temp=", "").replace("'C\n", "")
        return float(temp_str)
    except Exception as e:
        print(f"Hardware querying error: {e}")
        return 0.0


print(f"Attempting to connect to MQTT hub at {MQTT_BROKER_DNS}...")
try:
    mqtt_client.connect(MQTT_BROKER_DNS, MQTT_PORT, 60)
    print(f"Connected to hub via DNS: {MQTT_BROKER_DNS}")

    # loop_start() spawns a daemon thread.
    # It manages ping requests, network traffic, and automatic reconnections in the background
    mqtt_client.loop_start()

except socket.gaierror:
    # socket.gaierror is thrown if the hostname cannot be resolved.
    print(f"Warning: DNS resolution for {MQTT_BROKER_DNS} failed. Falling back to static IP.")
    try:
        mqtt_client.connect(MQTT_BROKER_FALLBACK_IP, MQTT_PORT, 60)
        print(f"Connected to hub via fallback IP: {MQTT_BROKER_FALLBACK_IP}")
        mqtt_client.loop_start()
    except Exception as e:
        print(f"Critical error: Fallback connection failed. {e}")
        exit(1)

except Exception as e:
    print(f"Critical error: Could not establish initial connection to the hub. {e}")
    exit(1)

# Start the background worker thread.
worker = threading.Thread(target=mqtt_worker_thread, daemon=True)
worker.start()

# ------------------------------
# Initialise Environment
# ------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Pre-populate deque with existing files to respect MAX_SNAPSHOTS on reboot.
existing_files = sorted(
    [os.path.join(SNAPSHOT_DIR, f) for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
    key=os.path.getmtime
)
for f in existing_files[-MAX_SNAPSHOTS:]:
    snapshot_tracker.append(f)

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
# Execution Loop with PASO Profiling
# ------------------------------
try:
    while True:
        # Check the state flag before doing any heavy lifting
        if not camera_active:
            time.sleep(1) # Idle in standby mode to save CPU cycles
            continue

        # Start master timer
        t_start = time.perf_counter()

        # --- Phase 1: Camera I/O Profiling ---
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue
        t_capture = time.perf_counter()

        # --- Phase 2: Pre-processing Profiling ---
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
        t_preprocess = time.perf_counter()

        # --- Phase 3: AI Inference Profiling ---
        # 3. Execute Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        t_inference = time.perf_counter()

        # --- Phase 4: Post-processing Profiling ---
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
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{CLIENT_ID}_snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)

            # Efficient local storage management via deque.
            if len(snapshot_tracker) == MAX_SNAPSHOTS:
                oldest_file = snapshot_tracker.popleft()
                try:
                    os.remove(oldest_file)
                except OSError as e:
                    print(f"Warning: Could not delete {oldest_file}. {e}")

            # Save to disk locally
            cv2.imwrite(filepath, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            snapshot_tracker.append(filepath)

            # Construct metadata and pass to the queue (non-blocking)
            payload_metadata = {
                "camera_id": CLIENT_ID,
                "location": LOCATION,
                "lab_id": LAB_ID,
                "timestamp": timestamp,
                "confidence": float(confidence_pct)
            }

            if not payload_queue.full():
                # Pass a copy of the ROI to prevent it being overwritten by the next frame
                payload_queue.put((roi.copy(), payload_metadata))
            else:
                print("Warning: Network queue is full. Dropping payload to maintain framerate.")

        else:
            print(f"[{datetime.now()}] Clear. No person detected.")
        
        # --- Output PASO Profiling Metrics ---
        # Convert seconds to milliseconds for granular analysis
        t_postprocess = time.perf_counter()
        time_capture = (t_capture - t_start) * 1000
        time_prep = (t_preprocess - t_capture) * 1000
        time_inf = (t_inference - t_preprocess) * 1000
        time_post = (t_postprocess - t_inference) * 1000
        time_total = (t_postprocess - t_start) * 1000
        current_temp = get_cpu_temp()

        print(f"\n--- Profiling Report ---")
        print(f"Camera I/O: {time_capture:.1f} ms")
        print(f"Pre-processing: {time_prep:.1f} ms")
        print(f"AI Inference: {time_inf:.1f} ms")
        print(f"Post-processing: {time_post:.1f} ms")
        print(f"Total Loop: {time_total:.1f} ms")
        print(f"Core Temp: {current_temp} Celsius")
        print(f"--------------------------\n")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping capture...")

finally:
    cap.release()
    # Safely terminate the background thread and network client.
    payload_queue.put(None)
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
