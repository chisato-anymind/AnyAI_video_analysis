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
import argparse
import concurrent.futures
from typing import Optional
import mimetypes
import multiprocessing

import uuid

# --- Flask and Environment Imports ---
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv

# --- Suppress common warnings ---
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

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
    # This will now clearly fail on startup if dependencies are missing
    sys.exit(f"FATAL: A required library is missing: {e}. Please run 'pip install -r requirements.txt'")

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# --- Global Configuration & State ---
# ==============================================================================

app = Flask(__name__, template_folder='../templates')

# --- Analysis Task State ---
# These variables manage the background processing thread
analysis_processes = {} # Dictionary to hold multiple analysis processes
log_buffer = []
log_lock = threading.Lock()
# Using a multiprocessing-safe queue for logs from child processes
log_queue = None

# --- Google API Configuration ---
SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
MAX_CELL_LEN = 45000

# ==============================================================================
# --- Logging and Helper Functions (from anyai_video.py) ---
# ==============================================================================

def log_message(message: str, is_error: bool = False, queue: Optional[multiprocessing.Queue] = None):
    """
    Logs a message to the console and to a queue if provided.
    This function can be called from the main process or child processes.
    """
    if queue:
        queue.put(message)
    else:
        # If no queue, we are in the main process, log to the global buffer
        with log_lock:
            log_buffer.append(message)
    
    if is_error:
        print(message, file=sys.stderr)
    else:
        print(message, file=sys.stdout)

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
# --- Core Google API and Gemini Logic (Child Process) ---
# ==============================================================================

def get_google_creds(client_secrets_file: str, log_q: multiprocessing.Queue) -> Optional[Credentials]:
    creds = None
    token_path = Path("credentials/token.json")
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SHEET_SCOPES)
        except Exception as e:
            log_message(f"WARNING: Could not load token.json: {e}. Re-authenticating.", is_error=True, queue=log_q)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                log_message("-> Refreshing expired credentials...", queue=log_q)
                creds.refresh(Request())
            except Exception as e:
                log_message(f"WARNING: Failed to refresh token, will re-authenticate: {e}", is_error=True, queue=log_q)
                if token_path.exists(): token_path.unlink()
                creds = None
        if not creds:
            log_message("-> Performing new user authentication (this may happen for each worker)...", queue=log_q)
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SHEET_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                log_message(f"FATAL: Failed to run authentication flow from '{client_secrets_file}': {e}", is_error=True, queue=log_q)
                return None
        try:
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            log_message(f"-> Credentials saved to {token_path}", queue=log_q)
        except Exception as e:
            log_message(f"WARNING: Could not write token to {token_path}: {e}", is_error=True, queue=log_q)
    return creds

RETRYABLE_API_ERRORS = (concurrent.futures.TimeoutError, HttpError)

@retry(retry=retry_if_exception_type(RETRYABLE_API_ERRORS), stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
def download_drive_file(drive_service, file_id: str, temp_dir: Path, log_q: multiprocessing.Queue) -> Path:
    try:
        log_message(f"   - [{file_id}] Downloading...", queue=log_q)
        file_metadata = drive_service.files().get(fileId=file_id, fields='name').execute()
        file_name = file_metadata.get('name', f"unknown_file_{file_id}")
        safe_filename = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in ('.', '_', '-')]).rstrip()
        local_path = temp_dir / safe_filename
        
        request = drive_service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        log_message(f"   - [{file_id}] Download complete: {local_path.name}", queue=log_q)
        return local_path
    except HttpError as e:
        if e.resp.status >= 500:
            log_message(f"   - [{file_id}] Retrying download due to server error (5xx): {e}", is_error=True, queue=log_q)
            raise
        raise RuntimeError(f"Non-retryable HTTP error downloading file {file_id}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id} from Drive: {e}")

@retry(retry=retry_if_exception_type(RETRYABLE_API_ERRORS), stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
def upload_video_and_wait(video_path: Path, max_wait_secs: int, log_q: multiprocessing.Queue):
    log_message(f"   - [{video_path.stem}] Uploading to Gemini API...", queue=log_q)
    video_file = genai.upload_file(
        path=video_path,
        mime_type=mimetypes.guess_type(video_path)[0] or "video/mp4"
    )
    waited_time = 0
    while video_file.state.name == "PROCESSING":
        if waited_time >= max_wait_secs:
            try: genai.delete_file(video_file.name)
            except Exception as e: log_message(f"   - Warning: Failed to clean up timed-out file {video_file.name}: {e}", is_error=True, queue=log_q)
            raise concurrent.futures.TimeoutError(f"Timeout waiting for file '{video_file.name}' after {max_wait_secs}s.")
        time.sleep(10)
        waited_time += 10
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError(f"File upload failed for '{video_file.name}'. Reason: {video_file.state}")
    log_message(f"   - [{video_path.stem}] Upload successful: {video_file.name}", queue=log_q)
    return video_file

@retry(retry=retry_if_exception_type(RETRYABLE_API_ERRORS), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30), reraise=True)
def analyze_with_gemini(model, video_file, prompt_text: str, log_q: multiprocessing.Queue) -> str:
    log_message(f"   - Analyzing {video_file.name} with Gemini...", queue=log_q)
    try:
        response = model.generate_content([prompt_text, video_file], request_options={'timeout': 600})
        return response.text.strip()
    except Exception as e:
        log_message(f"   - Error during Gemini analysis: {e}", is_error=True, queue=log_q)
        raise

def process_video_task_worker_wrapper(args):
    """Helper to unpack arguments for imap_unordered."""
    return process_video_task_worker(*args)

def process_video_task_worker(task_info: dict, config: dict, log_q: multiprocessing.Queue):
    """
    This function runs in a separate process.
    It handles one video from start to finish.
    """
    row_idx = task_info['row']
    video_url = task_info['url']
    
    log_message(f"-> Worker started for Row {row_idx}: {video_url}", queue=log_q)
    
    gemini_file, local_video_path = None, None
    temp_dir = Path("./temp_video_downloads")
    
    try:
        # --- Each worker needs its own credentials and services ---
        creds = get_google_creds(config["client_secrets"], log_q)
        if not creds:
            raise RuntimeError("Worker failed to get Google credentials.")
            
        drive_service = build('drive', 'v3', credentials=creds)
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found by worker.")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(config["model"])
        
        prompt_text = Path(config["prompt_file"]).read_text(encoding="utf-8")
        # --- End of worker setup ---

        file_id = extract_drive_file_id_from_url(video_url)
        if not file_id:
            return (row_idx, "SKIPPED - Invalid Google Drive URL")
        
        local_video_path = download_drive_file(drive_service, file_id, temp_dir, log_q)
        gemini_file = upload_video_and_wait(local_video_path, config["max_wait"], log_q)
        result_text = analyze_with_gemini(model, gemini_file, prompt_text, log_q)
        
        first_brace = result_text.find('{')
        last_brace = result_text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            result_text = result_text[first_brace:last_brace+1].strip()
        if len(result_text) > MAX_CELL_LEN:
            result_text = result_text[:MAX_CELL_LEN - 20] + "... [TRUNCATED]"
            log_message(f"   - [{row_idx}] Warning: Result was truncated.", queue=log_q)
        
        return (row_idx, result_text)

    except Exception as e:
        error_message = f"ERROR: {type(e).__name__}: {e}"
        log_message(f"   - ERROR on Row {row_idx}: {error_message}", is_error=True, queue=log_q)
        return (row_idx, error_message[:MAX_CELL_LEN])
    finally:
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
            except Exception: pass # Ignore cleanup errors
        if local_video_path and local_video_path.exists():
            try:
                local_video_path.unlink()
            except Exception: pass # Ignore cleanup errors
        log_message(f"-> Worker finished for Row {row_idx}", queue=log_q)

# ==============================================================================
# --- Main Background Thread Logic ---
# ==============================================================================

def analysis_main_logic(config: dict, log_q: multiprocessing.Queue):
    """The main entry point for the background analysis process."""
    try:
        log_message("--- Background Analysis Process Started ---", queue=log_q)
        
        # --- Initial Setup & Validation (in main process) ---
        log_message("[1] Validating configuration...", queue=log_q)
        if not Path(config["client_secrets"]).is_file():
            raise ValueError(f"Client secrets file not found at '{config['client_secrets']}'")
        if not Path(config["prompt_file"]).is_file():
            raise ValueError(f"Prompt file not found at '{config['prompt_file']}'")
        log_message("[+] Configuration valid.", queue=log_q)

        log_message("[2] Authenticating with Google in main process...", queue=log_q)
        creds = get_google_creds(config["client_secrets"], log_q)
        if not creds:
            raise RuntimeError("Failed to get Google credentials in main process.")
        log_message("[+] Google authentication successful.", queue=log_q)
        
        log_message("[3] Connecting to Google Sheets...", queue=log_q)
        gc = gspread.authorize(creds)
        sheet_id = extract_sheet_id_from_url(config["sheet_url"])
        spreadsheet = gc.open_by_key(sheet_id)
        source_ws = spreadsheet.worksheet(config["source_sheet"])
        output_ws = spreadsheet.worksheet(config["output_sheet"]) if config.get("output_sheet") else source_ws
        log_message("[+] Connected to Google Sheets.", queue=log_q)

        log_message("[4] Reading and filtering tasks from sheet...", queue=log_q)
        video_col_num = col_to_num(config["video_col"])
        output_col_num = col_to_num(config["output_col"])
        all_data = source_ws.get_all_values()
        end_row = config.get("end_row") or len(all_data)
        tasks = []
        for i, row in enumerate(all_data, start=1):
            if config["start_row"] <= i <= end_row:
                video_url = (row[video_col_num - 1] if len(row) >= video_col_num else "").strip()
                output_val = (row[output_col_num - 1] if len(row) >= output_col_num else "").strip()
                if video_url and not output_val and "drive.google.com" in video_url:
                    tasks.append({'row': i, 'url': video_url})
        
        if not tasks:
            log_message("-> No tasks to process. Exiting.", queue=log_q)
            return
        log_message(f"[+] Found {len(tasks)} tasks.", queue=log_q)

        log_message("[5] Starting multiprocessing pool...", queue=log_q)
        temp_dir = Path("./temp_video_downloads")
        temp_dir.mkdir(exist_ok=True)
        
        # Create a list of arguments for each worker
        worker_args = [(task, config, log_q) for task in tasks]

        # --- Process tasks and write results in batches ---
        BATCH_SIZE = 10  # Write to the sheet every 10 results
        update_requests = []
        tasks_processed = 0

        with multiprocessing.Pool(processes=config["workers"]) as pool:
            log_message(f"[+] Pool started with {config['workers']} processes. Processing {len(tasks)} tasks...", queue=log_q)
            
            # Use imap_unordered to get results as they are completed
            results_iterator = pool.imap_unordered(process_video_task_worker_wrapper, worker_args)
            
            for row_idx, result_text in results_iterator:
                tasks_processed += 1
                log_message(f"  -> Result received for row {row_idx} ({tasks_processed}/{len(tasks)} complete).", queue=log_q)
                
                update_requests.append({
                    'range': gspread.utils.rowcol_to_a1(row_idx, output_col_num),
                    'values': [[result_text]],
                })

                # If the batch is full, write to the sheet
                if len(update_requests) >= BATCH_SIZE:
                    log_message(f"  -> Writing batch of {len(update_requests)} results to the sheet...", queue=log_q)
                    try:
                        output_ws.batch_update(update_requests)
                        log_message("  -> Batch write successful.", queue=log_q)
                        update_requests = []  # Clear the batch
                    except Exception as e:
                        log_message(f"  -> WARNING: Failed to write batch to sheet: {e}", is_error=True, queue=log_q)

        log_message("[+] Multiprocessing pool finished.", queue=log_q)
        
        # Write any remaining results in the last batch
        if update_requests:
            log_message(f"-> Writing final batch of {len(update_requests)} results to the sheet...", queue=log_q)
            try:
                output_ws.batch_update(update_requests)
                log_message("-> Final batch write successful.", queue=log_q)
            except Exception as e:
                log_message(f"-> WARNING: Failed to write final batch to sheet: {e}", is_error=True, queue=log_q)

        log_message("[+] All results have been processed and written.", queue=log_q)

    except Exception as e:
        log_message(f"\n--- A FATAL ERROR occurred in the background process: {e} ---", is_error=True, queue=log_q)
        # Also print traceback to the main console for debugging
        traceback.print_exc(file=sys.stderr)
    finally:
        log_message(f"\n--- Background Analysis Process Finished ---", queue=log_q)
        temp_dir = Path("./temp_video_downloads")
        if temp_dir.exists():
            try:
                for f in temp_dir.iterdir():
                    try: f.unlink()
                    except: pass
                temp_dir.rmdir()
                log_message("[+] Temporary directory cleaned up.", queue=log_q)
            except Exception as e:
                log_message(f"Warning: Could not fully clean up temp directory '{temp_dir}': {e}", is_error=True, queue=log_q)
        # Signal that the main process is done
        log_q.put("---PROCESS_COMPLETE---")

# ==============================================================================
# --- Flask Web Server Routes ---
# ==============================================================================

def find_client_secrets_file() -> Optional[Path]:
    """Looks for the specific client_secrets.json file in the credentials directory."""
    secrets_path = Path("credentials/client_secrets.json")
    if secrets_path.is_file():
        return secrets_path
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-analysis', methods=['POST'])
def run_analysis_route():
    global analysis_processes
    log_queue = app.config['log_queue']

    data = request.json
    
    # --- Server-side Validation ---
    required_fields = ['sheet_url', 'source_sheet_name', 'video_col_letter', 'output_col_letter', 'start_row', 'model_name']
    errors = [f for f in required_fields if not data.get(f)]
    if errors: return jsonify({"error": f"Missing required fields: {', '.join(errors)}"}), 400
    
    client_secrets_path = find_client_secrets_file()
    if not client_secrets_path:
        return jsonify({"error": "Could not find a client_secrets .json file."} ), 400

    try:
        config = {
            "sheet_url": data['sheet_url'],
            "source_sheet": data['source_sheet_name'],
            "output_sheet": data.get('output_sheet_name') or data['source_sheet_name'],
            "video_col": data['video_col_letter'],
            "output_col": data['output_col_letter'],
            "start_row": int(data['start_row']),
            "end_row": int(data['end_row']) if data.get('end_row') else None,
            "model": data['model_name'],
            "workers": int(data.get('workers', 10)), # Default to 10 workers
            "max_wait": 900,
            "prompt_file": "config/prompt.txt",
            "client_secrets": str(client_secrets_path)
        }
    except (ValueError, TypeError):
        return jsonify({"error": "Start row, end row, and workers must be valid numbers."} ), 400

    if not Path(config["prompt_file"]).is_file():
        return jsonify({"error": "The 'prompt.txt' file was not found."} ), 400

    # Reset the queue
    while not log_queue.empty():
        log_queue.get()

    process_id = str(uuid.uuid4())
    analysis_process = multiprocessing.Process(target=analysis_main_logic, args=(config, log_queue))
    analysis_processes[process_id] = analysis_process
    analysis_process.start()
    
    return jsonify({"status": "success", "message": "Analysis started in the background.", "process_id": process_id} )

@app.route('/stream-logs')
def stream_logs():
    process_id = request.args.get('process_id')
    analysis_process = analysis_processes.get(process_id)
    log_queue = app.config['log_queue']

    def generate():
        while True:
            try:
                # Block until a message is available or timeout
                message = log_queue.get(timeout=1.0)
                if message == "---PROCESS_COMPLETE---":
                    yield "event: complete\ndata: Analysis complete.\n\n"
                    break
                yield f"data: {message}\n\n"
            except multiprocessing.queues.Empty:
                # Check if the process is still alive
                if not analysis_process or not analysis_process.is_alive():
                    # Drain any final messages
                    while not log_queue.empty():
                        message = log_queue.get()
                        if message == "---PROCESS_COMPLETE---":
                            break
                        yield f"data: {message}\n\n"
                    yield "event: complete\ndata: Analysis complete.\n\n"
                    break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis_route():
    global analysis_processes
    data = request.json
    process_id = data.get('process_id')
    analysis_process = analysis_processes.get(process_id)

    if not analysis_process or not analysis_process.is_alive():
        return jsonify({"status": "error", "message": "No analysis is currently running for this ID."} ), 400
    
    log_message("-> STOP signal received by server. Terminating child processes...")
    analysis_process.terminate()
    analysis_process.join()
    if process_id in analysis_processes:
        del analysis_processes[process_id]
    log_message("-> Analysis process terminated.")
    return jsonify({"status": "success", "message": "Stop signal sent. The analysis has been terminated."} )


# ==============================================================================
# --- Server Startup ---
# ==============================================================================

if __name__ == '__main__':
    # --- Pre-flight Check for Credentials ---
    # Check for the secrets file before starting the server to provide a clear error.
    secrets_path = Path("credentials/client_secrets.json")
    if not secrets_path.is_file():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print("!!! FATAL ERROR: Credentials file not found.", file=sys.stderr)
        print(f"!!! Expected to find '{secrets_path.name}' inside the '{secrets_path.parent.name}' directory.", file=sys.stderr)
        print("!!! Please follow the setup instructions in README.md to create it.", file=sys.stderr)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        sys.exit(1)

    # 'spawn' is the safest start method for macOS and Windows.
    # It avoids threading-related crashes inside the web server.
    multiprocessing.set_start_method('spawn', force=True)

    # A Manager is required to share a queue between processes when using 'spawn'.
    with multiprocessing.Manager() as manager:
        # Create a manager-owned queue
        log_queue = manager.Queue()

        if not os.getenv('GEMINI_API_KEY'):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
            print("!!! WARNING: GEMINI_API_KEY is not set in your .env file.", file=sys.stderr)
            print("!!! The analysis script will fail without it.", file=sys.stderr)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)

        print("-> Starting integrated web server and analysis engine...")
        print("-> To use the dashboard, open your browser to http://127.0.0.1:5001")
        
        # This is a bit of a hack to pass the queue to the Flask app
        app.config['log_queue'] = log_queue
        
        app.run(host='0.0.0.0', port=5001, debug=False)
