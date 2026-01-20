import threading
import time


def monitoring_loop():
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    # Run monitoring in a background thread.
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()

    # Start Flask dashboard (can also show live webcam feed + sensor data)
    print("Running!")
