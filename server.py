import os
import sys
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv

# --- Suppress the specific NotOpenSSLWarning ---
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass
# ---------------------------------------------

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

def find_client_secrets_file():
    """Finds the first .json file in the root directory."""
    for f in Path(".").glob("*.json"):
        # A simple check to exclude the token file
        if "token.json" not in f.name:
            return f
    return None

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """Receives form data and runs the analysis script as a subprocess."""
    data = request.json
    
    # --- Find required files ---
    client_secrets = find_client_secrets_file()
    if not client_secrets:
        return jsonify({"error": "Could not find a client_secrets .json file in the project directory."}), 400
    
    video_dir = Path("./local_video")
    if not video_dir.is_dir():
        return jsonify({"error": "The 'local_video' directory was not found."}), 400

    # Clean up any previous stop signal before starting a new run
    stop_signal_file = Path("stop_signal.txt")
    if stop_signal_file.exists():
        stop_signal_file.unlink()

    # --- Build the command for the subprocess ---
    command = [
        sys.executable, # Use the same python interpreter that's running flask
        "anyai_video.py",
        "--client-secrets", str(client_secrets),
        "--base-dir", str(video_dir),
        "--sheet-url", data.get('sheet_url', ''),
        "--source-sheet", data.get('source_sheet_name', 'Sheet1'),
        "--video-col", data.get('video_col_letter', 'B'),
        "--output-col", data.get('output_col_letter', 'G'),
        "--start-row", str(data.get('start_row', 1)),
        "--model", data.get('model_name', 'gemini-1.5-flash'),
        "--prompt-file", "prompt.txt",
        "--workers", str(data.get('workers', 5))
    ]
    
    # Add optional arguments if they exist
    if data.get('output_sheet_name'):
        command.extend(["--output-sheet", data['output_sheet_name']])
    if data.get('end_row'):
        command.extend(["--end-row", str(data['end_row'])])

    def generate_output():
        """Starts the subprocess and yields its output line by line."""
        try:
            # Using Popen to stream output in real-time
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1 # Line-buffered
            )
            
            # Yield each line from the subprocess's stdout
            for line in iter(process.stdout.readline, ''):
                yield f"data: {line}\n\n"
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                yield f"data: -> ERROR: Script finished with exit code {return_code}.\n\n"
            else:
                yield f"data: -> SCRIPT FINISHED SUCCESSFULLY.\n\n"

        except Exception as e:
            yield f"data: -> FATAL SERVER ERROR: {e}\n\n"

    # Return a streaming response
    return Response(generate_output(), mimetype='text/event-stream')

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis():
    """Creates a signal file that the analysis script will detect."""
    try:
        with open("stop_signal.txt", "w") as f:
            f.write("stop")
        print("-> Stop signal received. File created.", file=sys.stderr)
        return jsonify({"status": "success", "message": "Stop signal sent."}), 200
    except Exception as e:
        print(f"-> ERROR: Could not create stop signal file: {e}", file=sys.stderr)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("-> Starting web server...")
    print("-> To use the dashboard, open your browser to http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
