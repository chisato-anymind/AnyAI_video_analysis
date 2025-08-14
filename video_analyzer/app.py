
import configparser
import logging
import queue
import threading
import os
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from analyzer_engine import run_analysis

app = Flask(__name__)

# --- Define base directory and video folder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, 'local_video')
os.makedirs(VIDEO_FOLDER, exist_ok=True) # Create the folder if it doesn't exist

# --- In-memory queue for log streaming and a stop event ---
log_queue = queue.Queue()
stop_event = threading.Event()
analysis_thread = None

# --- Basic Logging Setup ---
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# --- Configuration Handling ---
CONFIG_FILE = 'config.ini'

def get_api_key():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config.get('GEMINI', 'API_KEY', fallback=None)

def set_api_key(key):
    config = configparser.ConfigParser()
    config['GEMINI'] = {'API_KEY': key}
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

# --- Flask Routes ---
@app.route('/')
def index():
    api_key = get_api_key()
    api_key_saved = api_key and api_key != 'YOUR_API_KEY_HERE'
    return render_template('index.html', api_key_saved=api_key_saved)

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        api_key = request.form['api_key']
        set_api_key(api_key)
        return 'API Key saved! <a href="/">Go back</a>.'
    return '''
        <title>API Key Setup</title>
        <h1>API Key Setup</h1>
        <p>Please enter your Google AI (Gemini) API key. This is a one-time setup.</p>
        <form method="post">
            <input type="text" name="api_key" size="50" required>
            <button type="submit">Save</button>
        </form>
    '''

@app.route('/run-analysis', methods=['POST'])
def start_analysis_route():
    global analysis_thread
    if analysis_thread and analysis_thread.is_alive():
        return jsonify({"error": "An analysis is already running."}), 400

    params = request.json
    params['video_folder'] = VIDEO_FOLDER # Add the video folder path automatically

    api_key = get_api_key()

    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return jsonify({"error": "API Key not configured. Please go to /setup"}), 400

    stop_event.clear()
    analysis_thread = threading.Thread(
        target=run_analysis,
        args=(params, api_key, logger, stop_event)
    )
    analysis_thread.start()
    return jsonify({"message": "Analysis started successfully."})

@app.route('/stream-logs')
def stream_logs():
    def generate():
        while True:
            try:
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                if analysis_thread and not analysis_thread.is_alive():
                    yield "event: complete\ndata: Process finished.\n\n"
                    break
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis_route():
    if analysis_thread and analysis_thread.is_alive():
        stop_event.set()
        return jsonify({"message": "Stop signal sent."})
    return jsonify({"error": "No analysis is currently running."}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
