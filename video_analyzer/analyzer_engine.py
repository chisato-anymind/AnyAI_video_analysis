import gspread
import google.generativeai as genai
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def process_video(video_path, prompt, model):
    """Uploads a single video and gets the analysis from Gemini."""
    try:
        video_file = genai.upload_file(path=str(video_path))
        
        while video_file.state.name == "PROCESSING":
            time.sleep(5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            return "Error: Video processing failed."

        response = model.generate_content([prompt, video_file])
        return response.text
    except Exception as e:
        return f"Error: {e}"

def run_analysis(params, api_key, logger, stop_event):
    """The main analysis function, adapted for web server execution."""
    try:
        # --- Initialization ---
        logger.info("Authenticating with Google and initializing Gemini...")
        creds = get_google_credentials() # This function needs to be defined or imported
        if not creds:
            logger.error("Authentication failed.")
            return
        
        gc = gspread.authorize(creds)
        genai.configure(api_key=api_key)
        
        prompt_file = Path("prompt.txt")
        if not prompt_file.is_file():
            logger.error("prompt.txt not found.")
            return
        prompt = prompt_file.read_text()
        logger.info("Prompt loaded successfully.")

        # --- Get Worksheet Objects ---
        logger.info(f"Opening Google Sheet: {params['sheet_url']}")
        spreadsheet = gc.open_by_url(params['sheet_url'])
        input_worksheet = spreadsheet.worksheet(params['source_sheet_name'])
        
        output_sheet_name = params.get('output_sheet_name') or params['source_sheet_name']
        output_worksheet = spreadsheet.worksheet(output_sheet_name)
        logger.info(f"Input sheet: '{input_worksheet.title}', Output sheet: '{output_worksheet.title}'")

        # --- Read Data from Sheets ---
        video_names = input_worksheet.col_values(ord(params['video_col_letter'].upper()) - 64)
        output_data = output_worksheet.col_values(ord(params['output_col_letter'].upper()) - 64)
        
        # --- Prepare Tasks ---
        tasks = []
        video_folder_path = Path(params['video_folder'])
        start_row = int(params.get('start_row', 2))
        end_row = int(params.get('end_row') or len(video_names))

        logger.info(f"Processing rows from {start_row} to {end_row}.")

        for i in range(start_row - 1, end_row):
            if stop_event.is_set():
                logger.warning("Stop signal received. Halting task preparation.")
                break
            if i >= len(video_names): break
            
            row_index = i + 1
            
            if i < len(output_data) and output_data[i].strip():
                continue

            video_base_name = video_names[i].strip()
            if not video_base_name: continue

            found_files = list(video_folder_path.glob(f"{video_base_name}.*"))
            if not found_files:
                tasks.append({'row': row_index, 'result': f"File starting with '{video_base_name}' not found", 'future': None})
            else:
                tasks.append({'row': row_index, 'path': found_files[0], 'future': None})

        # --- Process Videos Concurrently ---
        if not tasks:
            logger.info("No new videos to process.")
            return

        model = genai.GenerativeModel(params['model_name'])
        results_to_update = []
        
        with ThreadPoolExecutor(max_workers=int(params['workers'])) as executor:
            logger.info(f"Submitting {len(tasks)} tasks with a batch size of {params['workers']}...")
            for task in tasks:
                if stop_event.is_set():
                    logger.warning("Stop signal received. Halting submission.")
                    break
                if 'path' in task:
                    task['future'] = executor.submit(process_video, task['path'], prompt, model)

            for task in tasks:
                if stop_event.is_set():
                    logger.warning("Stop signal received during processing. In-flight tasks will complete.")
                    # Note: We don't kill futures, just stop waiting for new ones.
                    # The results processed so far will be updated.
                    break

                row_index = task['row']
                result_text = task.get('result')
                if task.get('future'):
                    try:
                        result_text = task['future'].result()
                        logger.info(f"Row {row_index}: Analysis complete.")
                    except Exception as e:
                        result_text = f"Future Error: {e}"
                        logger.error(f"Row {row_index}: An error occurred: {e}")
                
                results_to_update.append({
                    'range': f"{params['output_col_letter']}{row_index}",
                    'values': [[result_text]]
                })

        # --- Update Google Sheet in a Batch ---
        if results_to_update:
            logger.info(f"\nUpdating Google Sheet '{output_worksheet.title}' with {len(results_to_update)} results...")
            output_worksheet.batch_update(results_to_update)
            logger.info("Sheet updated successfully.")

        logger.info("Analysis run finished.")

    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)

# We need to import or define get_google_credentials
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import os

TOKEN_FILE = "token.json"
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_google_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds
