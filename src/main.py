# File: src/main.py
import threading
import time

from app import run_flask


def monitoring_loop():
    """
    Background monitoring loop.

    This function runs continuously in a daemon thread.
    Sensor polling, logging, or hardware monitoring logic
    should be placed here.
    """
    while True:
        # TODO: Replace with real monitoring logic
        time.sleep(0.1)


# Raspberry Pi 5 Model B Rev 1.1

if __name__ == "__main__":
    # Run monitoring in a background thread.
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    print("Monitoring thread started.")

    # Start Flask dashboard (can also show live webcam feed + sensor data)
    print("Starting Flask dashboard...")
    run_flask()
