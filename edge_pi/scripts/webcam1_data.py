import json
import time
import os
import cv2
import paho.mqtt.client as mqtt
from datetime import datetime

# ---------------- MQTT CONFIG ----------------
BROKER_IP = "<MAIN_PI_IP>" # Replace with main Pi's IP
BROKER_PORT = 1883
WAKE_COMMAND_TOPIC = "system/wake_pi"

# ---------------- SNAPSHOT CONFIG ----------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'snapshot')
MAX_SNAPSHOTS = 50                   # Keep only the last 50 snapshots
CAPTURE_INTERVAL = 2                  # Seconds between snapshots

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ---------------- GLOBAL STATE ----------------
capture_active = False  # Flag to control continuous capture

# ---------------- MQTT CALLBACKS ----------------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(WAKE_COMMAND_TOPIC)
        print("✅ Pi Zero continuous capture ready")
    else:
        print(f"❌ MQTT connection failed with code {rc}")

def on_message(client, userdata, msg):
    global capture_active

    data = json.loads(msg.payload.decode())

    command = data.get("command")
    reason = data.get("reason", "unknown")

    if command == "wake":
        print(f"🚀 Waking up Pi Zero 2W - Reason: {reason}")
        if not capture_active:
            threading.Thread(target=continuous_capture, daemon=True).start()

    elif command == "sleep":
        print(f"💤 Sleep command received - Reason: {reason}")
        capture_active = False  # Stop continuous capture

# ---------------- CONTINUOUS CAPTURE FUNCTION ----------------
def continuous_capture():
    global capture_active
    capture_active = True
    print("📸 Starting continuous capture. Press Ctrl+C to stop.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed to open")
        capture_active = False
        return

    try:
        while capture_active:
            ret, frame = cap.read()
            if ret:
                # Timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cam1_snapshot_{timestamp}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)

                # Save snapshot
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"📷 Snapshot saved as {filepath}")

                # Manage snapshots folder
                snapshots = sorted(
                    [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".jpg")],
                    key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x))
                )
                while len(snapshots) > MAX_SNAPSHOTS:
                    oldest = snapshots.pop(0)
                    os.remove(os.path.join(SNAPSHOT_DIR, oldest))
                    print(f"🗑 Deleted oldest snapshot: {oldest}")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Continuous capture stopped by user")
    finally:
        cap.release()
        capture_active = False

# ---------------- MAIN ----------------
import threading

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, BROKER_PORT, 60)
client.loop_start()

print("⏰ Pi Zero Wake Receiver running... Waiting for wake command.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down MQTT client")
    capture_active = False  # Stop capture if running
    client.loop_stop()
    client.disconnect()