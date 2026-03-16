# File: src/app.py
import os
import json
import base64
import numpy as np
import paho.mqtt.client as mqtt
from flask import (
    Flask,
    jsonify,
    render_template,
    redirect,
    url_for,
    send_from_directory,
)
import time
import cv2
import threading

from db import Database
from entities.camera import Camera
from entities.camera_manager import CameraManager
from yolo_model import Detector

# -------------------------
# Directory Configuration
# -------------------------
# Define and create the non-compliance evidence directory
NON_COMPLIANCE_DIR = os.path.join(os.path.dirname(__file__), "non_compliance")
os.makedirs(NON_COMPLIANCE_DIR, exist_ok=True)

# Directory to save known faces
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -------------------------
# Initialize YOLO Detector
# -------------------------
detector = Detector()

# Create the Flask application instance.
app = Flask(__name__)

# -------------------------
# Application State (Stub)
# -------------------------

APP_START_TIME = time.time()

# These will later be populated by MQTT, vision, and sensor pipelines.
SYSTEM_STATUS = {
    "mqtt_connected": False,
    "camera_online": False,
    "mmwave_online": False,
    "last_alert": None,
}

LATEST_DETECTION = {
    "human_detected": False,
    "confidence": 0.0,
    "timestamp": None,
    "source": None,  # camera_id / sensor_id
}

EVENT_LOGS = []  # Will later be persisted to disk or database.

# -------------------------
# MQTT Test Integration
# -------------------------
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_USER = "edwin"
MQTT_PASS = "password"
MQTT_TOPIC = "sit/+/+/vision/person"


def on_connect(client, userdata, flags, reason_code, properties):
    """Callback for broker connection."""
    if reason_code == 0:
        print(f"[MQTT] Successfully connected to broker at {MQTT_BROKER}")
        # Explicitly request Qos 1 to prevent broker delivery downgrades
        client.subscribe(MQTT_TOPIC, qos=1)
        SYSTEM_STATUS["mqtt_connected"] = True
        print(f"[MQTT] Subscribed to topic: {MQTT_TOPIC} with QoS 1")
    else:
        print(f"[MQTT] Connection failed with code {reason_code}")


def on_message(client, userdata, msg):
    """Callback for receiving and parsing the payload."""
    try:
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)

        # Extract all necessary metadata for the database.
        camera_id = data.get("camera_id", "unknown_edge")
        location = data.get("location", "sit")
        lab_id = data.get("lab_id", "unknown_lab")
        confidence = data.get("confidence", 0.0)
        timestamp = data.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
        b64_image = data.get("image", "")

        print(f"[MQTT] Payload received from {camera_id}. Confidence: {confidence}%")

        # Verify the base64 string is present and not a placeholder.
        if b64_image and not b64_image.startswith("<"):
            # Decode the base64 string to binary.
            image_bytes = base64.b64decode(b64_image)

            # Conver the binary array to an OpenCV matrix
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # 1. Pass matrix to the combined YOLO/Face pipeline.
                # This returns the frame ALREADY annotated with YOLO boxes and Face Recognition names.
                results, annotated_frame, face_results = detector.detect_frame(img)

                # Guard clause: Drop the frame to save disk space if no human is present.
                if face_results == "NO_PERSON":
                    print("[VISION] YOLO detected no personnel. Discarding frame.")
                    return

                # Update API State for testing only after confirming a person is present.
                LATEST_DETECTION["source"] = camera_id
                LATEST_DETECTION["confidence"] = confidence
                LATEST_DETECTION["timestamp"] = timestamp
                LATEST_DETECTION["human_detected"] = True

                if annotated_frame is not None:
                    # 2. Update the global frame for the Flask dashboard stream.
                    global latest_frame
                    # Acquire the lock before modifying the variable.
                    with frame_lock:
                        latest_frame = annotated_frame

                    # 3. Save only the annotated frame as evidence.
                    filename = f"incident_{camera_id}_{timestamp}.jpg"
                    filepath = os.path.join(NON_COMPLIANCE_DIR, filename)
                    cv2.imwrite(filepath, annotated_frame)

                    # Insert the metadata and filename pointer into SQLite.
                    db.insert_snapshot(
                        camera_id=camera_id,
                        location=location,
                        lab_id=lab_id,
                        timestamp=timestamp,
                        confidence=confidence,
                        filename=filename,
                    )
                    print(f"[DB] Logged incident {filename} to database.")

                else:
                    print(f"[MQTT] Warning: YOLO returned an empty frame.")

            else:
                print("[MQTT] Error: cv2 failed to decode the image matrix.")
        else:
            print(
                "[MQTT] Warning: Payload did not contain a valid base64 image string."
            )

    except json.JSONDecodeError:
        print("[MQTT] Error: Received malformed JSON payload.")
    except Exception as e:
        print(f"[MQTT] Unexpected error during message processing: {e}")


# Initialise MQTT Thread
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

print("[SYSTEM] Starting MQTT validation thread...")
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


# Initialize camera (edge simulation)
# camera = Camera()
cm = CameraManager()
cm.add_camera("cam1", source="/dev/video0")  # Use local webcam as "cam1"

# Instantiate the database wrapper for local use in this module.
db = Database()

# Initialise the thread lock globally.
frame_lock = threading.Lock()
latest_frame = None


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Dashboard landing page.
    """
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Dashboard landing page.
    """
    return render_template("dashboard.html")


@app.route("/api/capture_face/<name>", methods=["POST"])
def capture_face(name):
    """
    Capture the current frame from the camera and save it as a known face.
    The user provides 'name' in the URL.
    """
    global latest_frame

    if latest_frame is None:
        return jsonify({"status": "error", "message": "No frame available yet."}), 400

    # Optional: convert name to lowercase.
    name = name.strip().lower()
    if not name:
        return jsonify({"status": "error", "message": "Name cannot be empty."}), 400

    # Save full frame (or crop face later)
    filename = f"{name}.jpg"
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, latest_frame)

    return jsonify(
        {
            "status": "ok",
            "message": f"Face saved as {filename}. Restart app to activate.",
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    """
    Machine-readable health endpoint.
    Used for uptime monitoring and watchdog checks.
    """
    uptime_seconds = int(time.time() - APP_START_TIME)

    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "service": "edge-lab-monitor",
        }
    )


@app.route("/api/status", methods=["GET"])
def system_status():
    """
    Returns current system connectivity and sensor status.
    """
    return jsonify(SYSTEM_STATUS)


@app.route("/api/detection/latest", methods=["GET"])
def latest_detection():
    """
    Returns the most recent detection result.
    """
    return jsonify(LATEST_DETECTION)


@app.route("/api/events", methods=["GET"])
def event_logs():
    """
    Returns historical event logs fetched directly from the SQLite database.
    """
    # Fetch the 50 most recent events.
    recent_events = db.get_recent_events(limit=50)

    return jsonify(
        {
            "count": len(recent_events),
            "events": recent_events,
        }
    )


@app.route("/api/project", methods=["GET"])
def project_info():
    """
    Returns project background, objectives, and equipment summary.
    """
    return jsonify(
        {
            "title": "Edge-Based Laboratory Monitoring System",
            "purpose": (
                "To prevent accidents, unauthorised access, and equipment damage "
                "by implementing automated real-time monitoring in restricted "
                "laboratory environments."
            ),
            "key_technologies": [
                "Raspberry Pi Edge Computing",
                "YOLO Human Detection (YOLO11-nano)",
                "mmWave Motion Sensor",
                "MQTT Messaging",
                "Computer Vision",
            ],
            "objectives": [
                "Real-time detection of unauthorised human entry.",
                "Multi-sensor validation using vision and motion sensing.",
                "Low-latency edge processing without cloud dependency.",
                "Automated alert delivery via MQTT.",
                "Performance evaluation of latency and accuracy.",
            ],
            "expected_outcomes": [
                "Functional Raspberry Pi prototype.",
                "Real-time alerting system.",
                "Performance metrics and evaluation.",
                "Live monitoring dashboard.",
                "Full technical documentation.",
            ],
            "equipment": [
                "3 x Raspberry Pi",
                "1 x ESP32",
                "3 x Webcams",
                "1 x mmWave Motion Sensor",
            ],
        }
    )


def run_flask():
    app.run(
        host="0.0.0.0",  # Allows access from other devices on the network
        port=5000,
        debug=True,
        use_reloader=False,  # Must be disabled when running in threads
    )
