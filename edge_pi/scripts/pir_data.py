import RPi.GPIO as GPIO
import time
import json
import paho.mqtt.client as mqtt
from datetime import datetime

# MQTT Broker
BROKER_IP = "<<MAIN_PI_IP>>"
BROKER_PORT = 1883
WAKE_PI_TOPIC = "system/wake_pi"

# PIR Sensor Configuration
PIR_PIN = 4  # GPIO pin for PIR sensor
DEBOUNCE_TIME = 2  # seconds

# Setup PIR
def setup_pir_sensor():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN)
    print(f"👁️ PIR sensor initialized on GPIO{PIR_PIN}")

# Function to wake camera Pi
def wake_pi_zero(client, reason):
    wake_payload = {
        "command": "wake",
        "target_pi": "webcam1_pi",
        "source": "pir_node",
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "action": "capture_image"
    }
    client.publish(WAKE_PI_TOPIC, json.dumps(wake_payload))
    print(f"🔔 Wake command sent - Reason: {reason}")

# Monitor PIR sensor
def monitor_pir_sensor(client):
    last_trigger = 0
    print("👁️ PIR sensor monitoring started...")
    while True:
        try:
            if GPIO.input(PIR_PIN):
                current_time = time.time()
                if current_time - last_trigger > DEBOUNCE_TIME:
                    print("🚨 Motion detected!")
                    last_trigger = current_time
                    wake_pi_zero(client, "motion_detected")
                    time.sleep(0.5)  # short delay after trigger
            time.sleep(0.1)  # prevent CPU overuse
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ PIR monitoring error: {e}")

# MQTT setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "PiZero_PIR_Monitor")
client.connect(BROKER_IP, BROKER_PORT, 60)
client.loop_start()

# Setup and start monitoring
setup_pir_sensor()
try:
    monitor_pir_sensor(client)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
finally:
    GPIO.cleanup()
    client.loop_stop()
    client.disconnect()
    print("✅ Cleanup complete")
