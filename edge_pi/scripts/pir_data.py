import time
import json
import threading
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# ===== CONFIGURATION =====
BROKER_IP = "<MAIN_PI_IP>" # Replace with main Pi's IP
BROKER_PORT = 1883
MQTT_TOPIC = "system/wake_pi"

PIR_PIN = 4
MIN_TRIGGER_INTERVAL = 0.5   # seconds debounce
STUCK_LOW_THRESHOLD = 10     # seconds to publish normal sleep
FAILURE_TIMEOUT = 30         # seconds of continuous sleep to trigger PIR failure

# ===== MQTT SETUP =====
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect_async(BROKER_IP, BROKER_PORT, 60)
client.loop_start()

# ===== TRY PIGPIO =====
try:
    import pigpio
    pi = pigpio.pi()
    USE_INTERRUPTS = pi.connected
    if USE_INTERRUPTS:
        pi.set_mode(PIR_PIN, pigpio.INPUT)
        pi.set_pull_up_down(PIR_PIN, pigpio.PUD_DOWN)
        print("✅ pigpio connected - using interrupt mode")
    else:
        pi = None
        print("⚠️ pigpio not connected - will use polling")
except (ImportError, ModuleNotFoundError):
    pi = None
    USE_INTERRUPTS = False
    print("⚠️ pigpio not installed - will use polling")

# ===== GLOBAL STATE =====
last_trigger_time = 0
sleep_published_at = None      # timestamp when sleep was last published
failure_wakeup_done = False    # flag to prevent repeated failure wakeups
stats = {"count": 0, "min_latency": float('inf'), "max_latency": 0}

# ===== MQTT HELPERS =====
def send_wake_message(latency_us=0, reason="pir_triggered"):
    payload = {
        "command": "wake",
        "source": "pir_node",
        "timestamp": time.time(),
        "latency_us": latency_us,
        "reason": reason
    }
    client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
    print(f"🚨 Published wake ({reason})")

def send_wake_upon_failure():
    send_wake_message(reason="pir_disconnected")

def publish_sleep():
    global sleep_published_at, failure_wakeup_done
    payload = {
        "command": "sleep",
        "source": "pir_node",
        "timestamp": time.time(),
        "reason": "pir_inactive"
    }
    client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
    sleep_published_at = time.time()
    failure_wakeup_done = False
    print("💤 Published sleep command")

# ===== INTERRUPT CALLBACK =====
def motion_detected(gpio, level, tick):
    global last_trigger_time, sleep_published_at, failure_wakeup_done
    if level != 1:  # Rising edge only
        return

    now = time.time()
    if now - last_trigger_time < MIN_TRIGGER_INTERVAL:
        return

    last_trigger_time = now
    latency_us = pigpio.tickDiff(tick, pi.get_current_tick()) if pi else 0

    stats["count"] += 1
    stats["min_latency"] = min(stats["min_latency"], latency_us)
    stats["max_latency"] = max(stats["max_latency"], latency_us)

    print(f"🚨 Motion #{stats['count']} | Latency: {latency_us} µs")
    send_wake_message(latency_us)
    # reset sleep/failure flags
    sleep_published_at = None
    failure_wakeup_done = False

# ===== LOW-STUCK CHECKER =====
def low_stuck_checker():
    global sleep_published_at, failure_wakeup_done, last_trigger_time
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    while True:
        pin_value = GPIO.input(PIR_PIN)
        now = time.time()

        if pin_value == 0:
            # Publish normal sleep if not already done
            if sleep_published_at is None and now - last_trigger_time >= STUCK_LOW_THRESHOLD:
                publish_sleep()

            # Trigger failure wake only once if sleep exceeds FAILURE_TIMEOUT
            if sleep_published_at and not failure_wakeup_done and (now - sleep_published_at >= FAILURE_TIMEOUT):
                send_wake_upon_failure()
                failure_wakeup_done = True
        else:
            # Reset timers if motion detected
            sleep_published_at = None
            failure_wakeup_done = False

        time.sleep(0.1)

# ===== POLLING MODE =====
def polling_mode():
    global last_trigger_time, sleep_published_at, failure_wakeup_done
    print("🔄 Polling mode activated")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    while True:
        pin_value = GPIO.input(PIR_PIN)
        now = time.time()

        # Motion detected
        if pin_value == 1 and now - last_trigger_time >= MIN_TRIGGER_INTERVAL:
            print("🚨 Motion detected (polling)")
            send_wake_message()
            last_trigger_time = now
            sleep_published_at = None
            failure_wakeup_done = False
            time.sleep(0.5)

        # LOW-stuck detection for sleep/failure
        elif pin_value == 0:
            if sleep_published_at is None and now - last_trigger_time >= STUCK_LOW_THRESHOLD:
                print(f"⚠️ PIR pin LOW for {STUCK_LOW_THRESHOLD}s. Publishing sleep...")
                publish_sleep()

            if sleep_published_at and not failure_wakeup_done and (now - sleep_published_at >= FAILURE_TIMEOUT):
                send_wake_upon_failure()
                failure_wakeup_done = True

        time.sleep(0.1)

# ===== START MONITORING =====
if USE_INTERRUPTS:
    pi.callback(PIR_PIN, pigpio.RISING_EDGE, motion_detected)
    threading.Thread(target=low_stuck_checker, daemon=True).start()
    print("👁️ Monitoring PIR sensor with interrupts")
else:
    threading.Thread(target=polling_mode, daemon=True).start()
    print("👁️ Monitoring PIR sensor with polling")

# ===== MAIN LOOP =====
print("Press Ctrl+C to exit\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    publish_sleep()
finally:
    if USE_INTERRUPTS and pi:
        pi.stop()
    client.loop_stop()
    client.disconnect()
    print("✅ Cleanup complete")