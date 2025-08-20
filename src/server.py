#!/usr/bin/env python3
"""
A self-contained Flask web server that directly handles video analysis 
tasks from a Google Sheet using the Gemini API. It runs the analysis in a 
background thread to avoid blocking the web server.
"""

# --- Standard Library Imports ---
import os
import sys
import threading
import time
import traceback
from pathlib import Path
import concurrent.futures
from typing import Optional
import mimetypes
from queue import Empty, Queue
import uuid

# --- Third-Party Imports ---
import webview
from waitress import serve
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv

# --- Google & Gemini API Imports ---
try:
    import google.generativeai as genai
    import gspread
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError as e:
    sys.exit(f"FATAL: A required library is missing: {e}. Please run 'pip install -r requirements.txt'")

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# --- Path Helper for Bundled App ---
# ==============================================================================

def get_base_path():
    """Get the base path for the app, whether running from source or bundled."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).parent.parent

BASE_DIR = get_base_path()

# ==============================================================================
# --- Flask App Initialization ---
# ==============================================================================

app = Flask(__name__, 
            template_folder=BASE_DIR / 'templates', 
            static_folder=BASE_DIR / 'static')

# --- Global State for Threading ---
analysis_thread = None
stop_event = threading.Event()
log_queue = Queue()

# --- Google API Configuration ---
SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive", # Use full drive scope for robustness
]
MAX_CELL_LEN = 45000

# ==============================================================================
# --- Logging and Helper Functions ---
# ==============================================================================

def log_message(message: str, is_error: bool = False):
    """Logs a message to the shared queue."""
    log_queue.put(message)
    print(message, file=sys.stderr if is_error else sys.stdout)

def col_to_num(col_str: str) -> Optional[int]:
    if not col_str or not isinstance(col_str, str): return None
    num = 0
    for char in col_str.upper():
        if not 'A' <= char <= 'Z': return None
        num = num * 26 + (ord(char) - ord('A')) + 1
    return num

def extract_sheet_id_from_url(url: str) -> str:
    if "/spreadsheets/d/" in url:
        return url.split("/spreadsheets/d/")[1].split("/")[0]
    return url

def extract_drive_file_id_from_url(url: str) -> Optional[str]:
    if "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    return None

# ==============================================================================
# --- Core Google API and Gemini Logic (Runs in Threads) ---
# ==============================================================================

def get_google_creds(client_secrets_file: str) -> Optional[Credentials]:
    creds = None
    token_path = BASE_DIR / "credentials/token.json"
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SHEET_SCOPES)
        except Exception as e:
            log_message(f"WARNING: Could not load token.json: {e}. Re-authenticating.", is_error=True)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log_message(f"WARNING: Failed to refresh token, will re-authenticate: {e}", is_error=True)
                if token_path.exists(): token_path.unlink()
                creds = None
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SHEET_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                log_message(f"FATAL: Failed to run auth flow from '{client_secrets_file}': {e}", is_error=True)
                return None
        try:
            token_path.parent.mkdir(exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            log_message(f"WARNING: Could not write token to {token_path}: {e}", is_error=True)
    return creds

def download_drive_file(drive_service, file_id: str, temp_dir: Path) -> Path:
    try:
        file_metadata = drive_service.files().get(fileId=file_id, fields='name').execute()
        file_name = file_metadata.get('name', f"unknown_file_{file_id}")
        safe_filename = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in ('.', '_', '-')]).rstrip()
        local_path = temp_dir / safe_filename
        request = drive_service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id} from Drive: {e}")

def upload_video_and_wait(video_path: Path, max_wait_secs: int):
    video_file = genai.upload_file(path=video_path, mime_type=mimetypes.guess_type(video_path)[0] or "video/mp4")
    waited_time = 0
    while video_file.state.name == "PROCESSING":
        if waited_time >= max_wait_secs:
            try: genai.delete_file(video_file.name)
            except Exception: pass
            raise TimeoutError(f"Timeout waiting for file '{video_file.name}'")
        time.sleep(10)
        waited_time += 10
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError(f"File upload failed for '{video_file.name}'")
    return video_file

def analyze_with_gemini(model, video_file, prompt_text: str) -> str:
    response = model.generate_content([prompt_text, video_file], request_options={'timeout': 600})
    if not response.candidates:
        return "SKIPPED - Analysis returned no candidates."
    
    candidate = response.candidates[0]
    if candidate.finish_reason == 1: # STOP
        # The response structure is now candidate.content.parts
        return "".join(part.text for part in candidate.content.parts).strip()
    else:
        return f"SKIPPED - Analysis stopped. Reason: {candidate.finish_reason.name}"

def process_video_task(task_info: dict, config: dict, api_key: str):
    if stop_event.is_set(): return (task_info['row'], "SKIPPED - Operation cancelled")
    row_idx = task_info['row']
    video_url = task_info['url']
    log_message(f"-> Processing Row {row_idx}: {video_url}")
    gemini_file, local_video_path = None, None
    temp_dir = BASE_DIR / "temp_video_downloads"
    try:
        creds = get_google_creds(config["client_secrets"])
        if not creds: raise RuntimeError("Worker failed to get Google credentials.")
        drive_service = build('drive', 'v3', credentials=creds)
        
        if not api_key: raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config["model"])
        
        prompt_text = (BASE_DIR / config["prompt_file"])
        if not prompt_text.is_file(): raise FileNotFoundError(f"Prompt file not found at {prompt_text}")
        prompt_text = prompt_text.read_text(encoding="utf-8")
        
        file_id = extract_drive_file_id_from_url(video_url)
        if not file_id: return (row_idx, "SKIPPED - Invalid Google Drive URL")
        
        local_video_path = download_drive_file(drive_service, file_id, temp_dir)
        gemini_file = upload_video_and_wait(local_video_path, config["max_wait"])
        result_text = analyze_with_gemini(model, gemini_file, prompt_text)
        return (row_idx, result_text)
    except Exception as e:
        error_message = f"ERROR: {type(e).__name__}: {e}"
        log_message(error_message, is_error=True)
        return (row_idx, error_message)
    finally:
        if gemini_file:
            try: genai.delete_file(gemini_file.name)
            except Exception: pass
        if local_video_path and local_video_path.exists():
            try: local_video_path.unlink() 
            except Exception: pass

# ==============================================================================
# --- Main Background Thread Logic ---
# ==============================================================================

def analysis_main_logic(config: dict):
    try:
        log_message("--- Background Analysis Started ---")
        creds = get_google_creds(config["client_secrets"])
        if not creds: raise RuntimeError("Failed to get Google credentials.")
        
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(extract_sheet_id_from_url(config["sheet_url"]))
        source_ws = spreadsheet.worksheet(config["source_sheet"])
        output_ws = spreadsheet.worksheet(config["output_sheet"]) if config.get("output_sheet") else source_ws
        
        all_data = source_ws.get_all_values()
        tasks = []
        video_col_num = col_to_num(config["video_col"])
        output_col_num = col_to_num(config["output_col"])
        end_row = config.get("end_row") or len(all_data)

        for i, row in enumerate(all_data, start=1):
            if config["start_row"] <= i <= end_row:
                video_url = (row[video_col_num - 1] if len(row) >= video_col_num else "").strip()
                output_val = (row[output_col_num - 1] if len(row) >= output_col_num else "").strip()
                if video_url and not output_val and "drive.google.com" in video_url:
                    tasks.append({'row': i, 'url': video_url})
        
        if not tasks:
            log_message("-> No tasks to process.")
            return
        
        temp_dir = BASE_DIR / "temp_video_downloads"
        temp_dir.mkdir(exist_ok=True)
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            log_message("FATAL: GEMINI_API_KEY not found in environment.", is_error=True)
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            future_to_task = {executor.submit(process_video_task, task, config, gemini_api_key): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                if stop_event.is_set(): break
                row_idx, result_text = future.result()
                log_message(f"  -> Result for row {row_idx} received. Attempting to update sheet...")
                try:
                    output_ws.update(f'{config["output_col"]}{row_idx}', [[result_text]])
                    log_message(f"  -> Successfully updated sheet for row {row_idx}.")
                except Exception as e:
                    log_message(f"  -> ERROR: Failed to update sheet for row {row_idx}: {e}", is_error=True)

    except Exception as e:
        log_message(f"\n--- A FATAL ERROR occurred: {e} ---", is_error=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        log_message(f"\n--- Analysis Finished ---")
        log_queue.put("---PROCESS_COMPLETE---")

# ==============================================================================
# --- Flask Web Server Routes ---
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-analysis', methods=['POST'])
def run_analysis_route():
    global analysis_thread
    if analysis_thread and analysis_thread.is_alive():
        return jsonify({"error": "An analysis is already running."} ), 400

    data = request.json
    client_secrets_path = BASE_DIR / "credentials/client_secrets.json"
    if not client_secrets_path.is_file(): return jsonify({"error": "Could not find client_secrets.json"} ), 400
    
    try:
        config = {
            "sheet_url": data['sheet_url'], "source_sheet": data['source_sheet_name'],
            "output_sheet": data.get('output_sheet_name') or data['source_sheet_name'],
            "video_col": data['video_col_letter'], "output_col": data['output_col_letter'],
            "start_row": int(data['start_row']), "end_row": int(data['end_row']) if data.get('end_row') else None,
            "model": data['model_name'], "workers": int(data.get('workers', 5)),
            "max_wait": 900, "prompt_file": "config/prompt.txt",
            "client_secrets": str(client_secrets_path)
        }
    except (ValueError, TypeError, KeyError): 
        return jsonify({"error": "Invalid or missing form data."} ), 400
    
    stop_event.clear()
    analysis_thread = threading.Thread(target=analysis_main_logic, args=(config,))
    analysis_thread.start()
    return jsonify({"status": "success", "message": "Analysis started."} )

@app.route('/stream-logs')
def stream_logs():
    def generate():
        while True:
            try:
                message = log_queue.get(timeout=30)
                if message == "---PROCESS_COMPLETE---":
                    yield "event: complete\ndata: Analysis process finished.\n\n"
                    break
                yield f"data: {message}\n\n"
            except Empty:
                if analysis_thread and not analysis_thread.is_alive():
                    yield "event: error\ndata: Log stream connection lost.\n\n"
                    break
                else:
                    yield ": keep-alive\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis_route():
    stop_event.set()
    return jsonify({"status": "success", "message": "Stop signal sent."} )

# ==============================================================================
# --- Server and Application Startup ---
# ==============================================================================

def run_server():
    # Use waitress for a production-ready, stable server
    serve(app, host='127.0.0.1', port=5000)

if __name__ == '__main__':
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    webview.create_window(
        'AnyAI Video Analysis',
        'http://127.0.0.1:5000',
        width=1200,
        height=800,
        resizable=True
    )
    webview.start(gui='cocoa') # Force cocoa backend for stability