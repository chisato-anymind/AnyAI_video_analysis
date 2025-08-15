#!/bin/bash

# --- Change to the script's directory ---
# This ensures that all relative paths in the Python script work correctly,
# making the app portable.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo "----------------------------------------------------"
echo " AnyAI Video Analysis Server"
echo "----------------------------------------------------"
echo "Working Directory: $(pwd)"
echo ""

# --- Activate Python Virtual Environment ---
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! ERROR: Virtual environment not found at '$VENV_PATH'"
    echo "!!! Please run the setup instructions from the README file."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # Wait for user to press Enter before exiting
    read -p "Press Enter to exit..."
    exit 1
fi

# --- Function to open browser ---
open_browser() {
    # Give the server a moment to start up
    sleep 2
    echo "Opening the application in your default browser..."
    open http://127.0.0.1:5001
}

# --- Start the Server ---
echo ""
echo "Starting the web server..."
echo "To stop the server, simply close this terminal window."
echo "----------------------------------------------------"

# Open the browser in the background
open_browser &

# Run the server in the foreground
python "$SCRIPT_DIR/src/server.py"

# This part of the script will run if the server stops for any reason.
echo ""
echo "----------------------------------------------------"
echo "Server has been stopped."
read -p "Press Enter to close this window..."
