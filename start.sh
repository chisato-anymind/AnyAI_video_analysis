#!/bin/bash

# This script starts the web server using the correct Python from the virtual environment.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate the virtual environment and run the server directly
echo "Starting the web server..."
"$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/server.py"
