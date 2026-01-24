#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating package lists..."
sudo apt update

echo "Installing core build tools and CMake..."
sudo apt install -y cmake build-essential libboost-all-dev libgtk-3-dev

echo "Installing linear algebra libraries..."
sudo apt install -y libopenblas-dev liblapack-dev

echo "Installing Python headers..."
sudo apt install -y python3-dev

echo "Installing SQLite..."
sudo apt install -y sqlite3

echo "All dependencies installed successfully!"
