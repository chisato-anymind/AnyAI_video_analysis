#!/usr/bin/env python3
"""
anyai_video.py - A command-line tool to analyze local videos listed in a 
Google Sheet and write results to a specified column. This script is designed
to be called from a server backend, not run interactively.
"""

import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

from pathlib import Path
from collections import defaultdict
import argparse
import os
import sys
import time
import traceback
import concurrent.futures
import threading

# --- Dependencies ---
try:
    from google import genai
except Exception:
    sys.exit("ERROR: Missing dependency: google-genai. Install with: pip install -U google-genai")

try:
    import gspread
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
except Exception:
    sys.exit("ERROR: Missing dependency. Install with: pip install -U gspread google-auth google-auth-oauthlib")

# --- Config ---
VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.wmv', '.mpeg', '.mpg', '.flv', '.3gp', '.3gpp'}
MAX_CELL_LEN = 45000
SHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# --- Helpers ---

def col_to_num(col_str):
    num = 0
    for char in col_str:
        if not 'A' <= char.upper() <= 'Z': return None
        num = num * 26 + (ord(char.upper()) - ord('A')) + 1
    return num

def extract_sheet_id_from_url(url: str) -> str:
    if "/spreadsheets/d/" in url:
        return url.split("/spreadsheets/d/")[1].split("/")[0]
    return url

def get_genai_client(api_key):
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)

def open_worksheet(spreadsheet_id: str, worksheet_name: str, client_secrets_file: str):
    creds = None
    token_path = Path("token.json")
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SHEET_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"WARNING: Failed to refresh token, re-authenticating: {e}", file=sys.stderr)
                creds = None
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SHEET_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                sys.exit(f"FATAL: Failed to run authentication flow from {client_secrets_file}: {e}")
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    try:
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
        return ws
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(f"FATAL: Failed to open Google Sheet. Check permissions and sheet names.")

def index_video_files(base_dir: Path):
    mapping = defaultdict(list)
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            mapping[p.name.lower()].append(p)
    return mapping

def find_matches(name: str, index: dict):
    key = name.strip().lower()
    if not key: return []
    paths = list(index.get(key, []))
    if paths: return paths
    if "." not in key:
        for fname, plist in index.items():
            if Path(fname).stem.lower() == key:
                paths.extend(plist)
    return paths

def upload_video_and_wait(video_path: Path, poll_secs: int, max_wait_secs: int):
    print(f"   - Uploading video: {video_path.name}...")
    video_file = genai.upload_file(path=video_path)
    
    waited = 0
    while video_file.state.name == "PROCESSING":
        if waited >= max_wait_secs:
            raise TimeoutError(f"Timeout waiting for file processing after {max_wait_secs}s.")
        time.sleep(poll_secs)
        waited += poll_secs
        video_file = genai.get_file(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.name}")
        
    return video_file

def analyze_with_gemini(model: str, file_ref, prompt_text: str) -> str:
    print(f"   - Analyzing with {model}...")
    model = genai.GenerativeModel(model_name=model)
    resp = model.generate_content([file_ref, prompt_text])
    return getattr(resp, "text", "").strip()

def safe_update_cell(ws, row: int, col: int, value: str, lock: threading.Lock):
    with lock:
        ws.update_cell(row, col, value)

# --- Worker ---

def process_video_task(task_info, video_index, output_ws, output_col_num, model, prompt_text, max_wait, lock, stop_event):
    row_idx, name = task_info
    file_ref = None
    try:
        if stop_event.is_set():
            return f"Row {row_idx}: SKIPPED - Stop signal received before starting."

        print(f"-> Processing Row {row_idx}: {name}")
        matches = find_matches(name, video_index)
        if not matches:
            print(f"   - Warning: File not found for '{name}'. Skipping row.")
            return f"Row {row_idx}: SKIPPED - File not found"

        video_path = matches[0]
        if len(matches) > 1:
            print(f"   - Warning: Multiple matches for '{name}'. Using first: {video_path.name}")

        if stop_event.is_set():
            return f"Row {row_idx}: HALTED - Stop signal received before upload."

        print(f"   - [{row_idx}] Uploading video...")
        file_ref = upload_video_and_wait(video_path, max_wait_secs=max_wait, poll_secs=10)

        if stop_event.is_set():
            return f"Row {row_idx}: HALTED - Stop signal received before analysis."

        print(f"   - [{row_idx}] Analyzing with Gemini...")
        result_text = analyze_with_gemini(model, file_ref, prompt_text)

        # --- Clean the response to ensure it's only the JSON object ---
        first_brace = result_text.find('{')
        last_brace = result_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            result_text = result_text[first_brace:last_brace+1].strip()
        # -------------------------------------------------------------

        if len(result_text) > MAX_CELL_LEN:
            result_text = result_text[:MAX_CELL_LEN - 20] + "... [TRUNCATED]"

        print(f"   - [{row_idx}] Writing result to sheet...")
        safe_update_cell(output_ws, row_idx, output_col_num, result_text, lock)

        return f"Row {row_idx}: Success"

    except Exception as e:
        error_message = f"ERROR: {e}"
        print(f"   - ERROR on Row {row_idx}: {error_message}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            safe_update_cell(output_ws, row_idx, output_col_num, error_message, lock)
        except Exception as e2:
            print(f"   - FATAL on Row {row_idx}: Could not write error to sheet: {e2}", file=sys.stderr)
        return f"Row {row_idx}: FAILED - {e}"
    finally:
        # Clean up the uploaded file on Gemini's side
        if file_ref:
            try:
                # Add a small delay before deleting
                time.sleep(2) 
                genai.delete_file(name=file_ref.name)
                print(f"   - [{row_idx}] Cleaned up file {file_ref.name}")
            except Exception as e:
                print(f"   - [{row_idx}] Warning: Failed to delete file {file_ref.name}: {e}", file=sys.stderr)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="A command-line tool to analyze videos from a Google Sheet concurrently.")
    parser.add_argument("--client-secrets", required=True, help="Path to OAuth 2.0 Client ID JSON file.")
    parser.add_argument("--sheet-url", required=True, help="URL of the Google Sheet.")
    parser.add_argument("--source-sheet", required=True, help="Name of the sheet with video names.")
    parser.add_argument("--video-col", required=True, help="Column letter for video names.")
    parser.add_argument("--output-sheet", help="Sheet to write results to (defaults to source sheet).")
    parser.add_argument("--output-col", required=True, help="Column letter for results.")
    parser.add_argument("--start-row", type=int, required=True, help="Start row number.")
    parser.add_argument("--end-row", type=int, help="End row number (processes all if omitted).")
    parser.add_argument("--model", required=True, choices=['gemini-1.5-pro', 'gemini-1.5-flash'], help="Gemini model to use.")
    parser.add_argument("--prompt-file", required=True, help="Path to the analysis prompt text file.")
    parser.add_argument("--base-dir", required=True, help="Base folder to search for videos.")
    parser.add_argument("--max-wait", type=int, default=900, help="Max seconds to wait for file processing.")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers.")
    args = parser.parse_args()

    print("-> Authenticating and opening Google Sheet...")
    sheet_id = extract_sheet_id_from_url(args.sheet_url)
    source_ws = open_worksheet(sheet_id, args.source_sheet, args.client_secrets)
    spreadsheet = source_ws.spreadsheet

    try:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    except Exception as e:
        sys.exit(f"FATAL: Could not read prompt file '{args.prompt_file}': {e}")

    if args.output_sheet and args.output_sheet != args.source_sheet:
        try:
            output_ws = spreadsheet.worksheet(args.output_sheet)
        except gspread.WorksheetNotFound:
            sys.exit(f"FATAL: Output worksheet '{args.output_sheet}' not found.")
    else:
        output_ws = source_ws

    video_col_num = col_to_num(args.video_col)
    output_col_num = col_to_num(args.output_col)
    if not video_col_num or not output_col_num:
        sys.exit("FATAL: Invalid column letter provided.")

    print(f"-> Indexing video files in '{args.base_dir}'...")
    video_index = index_video_files(Path(args.base_dir))
    print(f"-> Found {sum(len(v) for v in video_index.values())} video files.")

    print("-> Reading and filtering tasks from sheet...")
    data = source_ws.get_all_values()
    tasks = []
    end_row = args.end_row or len(data)
    video_col_idx = video_col_num - 1
    output_col_idx = output_col_num - 1

    for i, row in enumerate(data, start=1):
        if args.start_row <= i <= end_row:
            if len(row) > video_col_idx:
                name = (row[video_col_idx] or "").strip()
                if not name:
                    continue  # Skip row if no video name is present

                # Check if the output cell is already populated
                output_cell_value = ""
                if len(row) > output_col_idx:
                    output_cell_value = (row[output_col_idx] or "").strip()

                if not output_cell_value:
                    tasks.append((i, name))
                else:
                    print(f"-> Skipping Row {i}: Output cell already has content.")

    if not tasks:
        print("-> No tasks to process. All videos in the specified range may already be analyzed.")
        sys.exit(0)

    print(f"-> Found {len(tasks)} tasks to process using {args.workers} workers.")
    print("-> Starting concurrent processing...")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    get_genai_client(gemini_api_key)
    gspread_lock = threading.Lock()
    stop_event = threading.Event()
    processed_count = 0
    stop_signal_file = Path("stop_signal.txt")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    process_video_task,
                    task, video_index, output_ws, output_col_num,
                    args.model, prompt_text, args.max_wait, gspread_lock, stop_event
                ): task for task in tasks
            }

            # Monitor for stop signal and completed futures
            for future in concurrent.futures.as_completed(future_to_task):
                if stop_signal_file.exists():
                    print("-> Stop signal detected. Cancelling pending tasks...")
                    stop_event.set()
                    # Cancel all futures that haven't started yet
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
                    stop_signal_file.unlink()

                try:
                    result = future.result()
                    print(f"-> COMPLETED: {result}")
                    if "Success" in result:
                        processed_count += 1
                except concurrent.futures.CancelledError:
                    print(f"-> CANCELLED: A task was cancelled due to stop signal.")
                except Exception as exc:
                    task_info = future_to_task[future]
                    print(f"-> FATAL: Task for row {task_info[0]} generated an unexpected exception: {exc}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

    except KeyboardInterrupt:
        print("\n-> Keyboard interrupt received. Shutting down gracefully...")
        stop_event.set()

    finally:
        if stop_signal_file.exists():
            stop_signal_file.unlink()
        print(f"\n-> Done. Successfully processed {processed_count} of {len(tasks)} tasks.")

if __name__ == "__main__":
    main()
