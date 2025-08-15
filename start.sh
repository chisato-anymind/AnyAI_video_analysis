#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "Activating virtual environment..."
# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Check if activation was successful
if ! command -v python &> /dev/null
then
    echo "Could not activate the virtual environment. Please re-create it."
    exit 1
fi

echo "Starting the web server..."
# Run the server script from its new location in the src/ directory
python "$SCRIPT_DIR/src/server.py"

# Deactivate the virtual environment when the server stops
deactivate