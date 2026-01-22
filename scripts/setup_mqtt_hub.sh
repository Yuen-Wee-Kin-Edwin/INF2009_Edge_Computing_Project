#!/bin/bash

# -------------------------------------------------------------
# Script: setup_mqtt_hub_final.sh
# Purpose: Idempotent MQTT setup for Raspberry Pi hub
# Author: Edge-hub setup
# -------------------------------------------------------------

echo "=== Updating system packages ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Mosquitto broker and client utilities ==="
sudo apt install -y mosquitto mosquitto-clients

echo "=== Fixing permissions for Mosquitto data directories ==="
# Ensure Mosquitto can write to persistent and runtime directories
sudo chown -R mosquitto:mosquitto /var/lib/mosquitto
sudo chown -R mosquitto:mosquitto /var/log/mosquitto
sudo chown -R mosquitto:mosquitto /run/mosquitto
sudo chown mosquitto:mosquitto /etc/mosquitto/passwd
sudo chmod 600 /etc/mosquitto/passwd


echo "=== Enabling Mosquitto to start at boot ==="
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

echo "=== Checking Mosquitto status ==="
sudo systemctl status mosquitto | grep "Active:"

CONF_FILE="/etc/mosquitto/mosquitto.conf"

read -p "Do you want to enable MQTT authentication? (y/n) " auth_choice

if [ "$auth_choice" = "y" ]; then
    echo "=== Adding/updating MQTT user ==="

    # Make sure allow_anonymous is false
    if ! grep -q "^allow_anonymous false" "$CONF_FILE"; then
        # Remove allow_anonymous true if present
        sudo sed -i "/^allow_anonymous true/d" "$CONF_FILE"
        echo "allow_anonymous false" | sudo tee -a "$CONF_FILE"
    fi

    # Make sure password_file directive exists
    if ! grep -q "^password_file /etc/mosquitto/passwd" "$CONF_FILE"; then
        echo "password_file /etc/mosquitto/passwd" | sudo tee -a "$CONF_FILE"
    fi

    # Create passwd file if missing
    if [ ! -f /etc/mosquitto/passwd ]; then
        sudo touch /etc/mosquitto/passwd
    fi

    # Add or update user without removing other users
    read -p "Enter MQTT username to add/update: " mqtt_user
    read -s -p "Enter password for $mqtt_user: " mqtt_pass
    echo
    sudo mosquitto_passwd -b /etc/mosquitto/passwd "$mqtt_user" "$mqtt_pass"
    echo "User $mqtt_user added/updated."

    # Restart Mosquitto to apply auth
    sudo systemctl restart mosquitto
    echo "=== Authentication enabled ==="

else
    echo "=== Disabling authentication (anonymous access allowed) ==="
    # Remove allow_anonymous false if exists
    sudo sed -i "/^allow_anonymous false/d" "$CONF_FILE"
    # Add allow_anonymous true if missing
    if ! grep -q "^allow_anonymous true" "$CONF_FILE"; then
        echo "allow_anonymous true" | sudo tee -a "$CONF_FILE"
    fi
    # Restart broker
    sudo systemctl restart mosquitto
    echo "=== Authentication disabled. Users still remain in passwd file ==="
fi

echo "=== MQTT Broker setup complete ==="
echo "Broker is running on port 1883"
echo "Test subscriber:"
echo "  mosquitto_sub -h localhost -t 'test/topic'"
echo "Test publisher:"
echo "  mosquitto_pub -h localhost -t 'test/topic' -m 'Hello'"
