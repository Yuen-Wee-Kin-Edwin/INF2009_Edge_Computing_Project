# File: src/app.py
import os
from flask import Flask, Response, jsonify, render_template, redirect, url_for
import time
import cv2
import threading

from entities.camera import Camera
from entities.camera_manager import CameraManager
from yolo_model import Detector

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

# Initialize camera (edge simulation)
# camera = Camera()
cm = CameraManager()
cm.add_camera("cam1", source="/dev/video0")  # Use local webcam as "cam1"


# def generate_frames():
#     """Flask generator to stream MJPEG frames."""
#     while True:
#         frame_bytes = camera.get_frame_bytes()
#         if frame_bytes is None:
#             continue
#         yield (
#             b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
#         )


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


# Global variables for latest annotated frame.
latest_frame = None


def detection_loop(camera_id):
    """
    Continuously fetch frames from camera, run YOLO detection, and
    store the latest annotated frame for streaming.
    """
    global latest_frame
    while True:
        frame_bytes = cm.get_frame(camera_id)
        if frame_bytes is None:
            continue

        # Run YOLO detection and annotate frame.
        _, annotated_frame, face_results = detector.detect_frame(frame_bytes)

        if annotated_frame is not None:
            latest_frame = annotated_frame

        if face_results:
            print("[INFO] Face results:", face_results)


# Start the detection thread.
threading.Thread(target=detection_loop, args=("cam1",), daemon=True).start()

STREAM_FPS = 15  # target FPS


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    """
    Stream the latest YOLO-annotated frame at a stable rate.
    """

    def generate():
        global latest_frame
        while True:
            if latest_frame is None:
                continue  # Wait until first annoated frame is ready.

            # Encode frame to JPEG.
            ret, buffer = cv2.imencode(".jpg", latest_frame)
            if not ret:
                continue

            # Stream as MJPEG
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

            time.sleep(1 / STREAM_FPS)  # Pace streaming to ~15 FPS.

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Directory to save known faces
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


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
    Returns historical event logs (stub).
    """
    return jsonify(
        {
            "count": len(EVENT_LOGS),
            "events": EVENT_LOGS[-50:],  # Limit output size
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
