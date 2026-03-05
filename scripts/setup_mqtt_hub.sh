#!/bin/bash

# -------------------------------------------------------------
# Script: setup_mqtt_hub.sh
# Purpose: Idempotent MQTT setup for Raspberry Pi hub
# Author: Edge-hub setup
# -------------------------------------------------------------

CONF_FILE="/etc/mosquitto/mosquitto.conf"
PASSWD_FILE="/etc/mosquitto/passwd"

echo "=== Updating system packages ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Mosquitto broker and client utilities ==="
sudo apt install -y mosquitto mosquitto-clients

echo "=== Fixing permissions for Mosquitto data directories ==="
sudo chown -R mosquitto:mosquitto /var/lib/mosquitto
sudo chown -R mosquitto:mosquitto /var/log/mosquitto
sudo chown -R mosquitto:mosquitto /run/mosquitto

# Ensure password file exists securely
if [ ! -f "$PASSWD_FILE" ]; then
    sudo touch "$PASSWD_FILE"
fi

# Set strict permissions initially to satisfy utility pre-checks
sudo chown root:root "$PASSWD_FILE"
sudo chmod 0700 "$PASSWD_FILE"

echo "=== Enabling Mosquitto to start at boot ==="
sudo systemctl enable mosquitto

# --- Write clean main configuration ---
echo "=== Writing Mosquitto configuration ==="
sudo tee "$CONF_FILE" > /dev/null <<EOF
# Mosquitto main configuration

pid_file /run/mosquitto/mosquitto.pid

# Persistence
persistence true
persistence_location /var/lib/mosquitto/

# Logging
log_dest file /var/log/mosquitto/mosquitto.log

# Listener and authentication
listener 1883 0.0.0.0
allow_anonymous false
password_file $PASSWD_FILE

# Include additional configurations
include_dir /etc/mosquitto/conf.d
EOF

echo "=== Mosquitto configuration written ==="

read -p "Do you want to enable MQTT authentication? (y/n) " auth_choice

if [ "$auth_choice" = "y" ]; then
    echo "=== Adding/updating MQTT user ==="
    read -p "Enter MQTT username to add/update: " mqtt_user
    read -s -p "Enter password for $mqtt_user: " mqtt_pass
    echo
    
    # Ensure strict permissions before running the utility to suppress warnings
    sudo chown root:root "$PASSWD_FILE"
    sudo chmod 0700 "$PASSWD_FILE"
    
    sudo mosquitto_passwd -b "$PASSWD_FILE" "$mqtt_user" "$mqtt_pass"

    # Apply daemon-friendly ownership and permissions for the service runtime
    sudo chown root:mosquitto "$PASSWD_FILE"
    sudo chmod 0640 "$PASSWD_FILE"
    
    echo "User $mqtt_user added/updated."
else
    echo "=== Disabling authentication (allow anonymous clients) ==="
    # Modify configuration to allow anonymous
    sudo sed -i "s/^allow_anonymous false/allow_anonymous true/" "$CONF_FILE"
fi

echo "=== Restarting Mosquitto to apply changes ==="
sudo systemctl restart mosquitto
sudo systemctl status mosquitto | grep "Active:"

echo "=== MQTT Broker setup complete ==="
echo "Broker is running on port 1883"
echo "Test subscriber:"
echo "  mosquitto_sub -h localhost -t 'test/topic' -u <user> -P <password>'"
echo "Test publisher:"
echo "  mosquitto_pub -h localhost -t 'test/topic' -u <user> -P <password> -m 'Hello'"