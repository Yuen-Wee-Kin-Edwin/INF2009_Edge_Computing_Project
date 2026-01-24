# File: src/app.py
from flask import Flask, Response, jsonify, render_template, redirect, url_for
import time

from entities.camera import Camera
from entities.camera_manager import CameraManager

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


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    """Stream frames for a specific camera."""

    def generate():
        while True:
            frame_bytes = cm.get_frame(camera_id=camera_id)
            if frame_bytes is None:
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
