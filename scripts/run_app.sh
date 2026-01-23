#!/bin/bash
# run_app.sh
# This script activates the virtual environment and runs main.py
# with automatic restart on any file change in src/.

# ---- Configuration ----
VENV_PATH=".venv"          # Path to your virtual environment
SRC_PATH="src"             # Path to source folder

# ---- Activate virtual environment ----
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: virtual environment not found at $VENV_PATH"
    exit 1
fi

# ---- Run main.py with watchmedo ----
echo "Starting main.py with automatic restart on file changes..."
watchmedo auto-restart \
    --patterns="*.py;*.html;*.css;*.js" \
    --recursive \
    -- python "$SRC_PATH/main.py"
