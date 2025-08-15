import sys
import os
from pathlib import Path
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Configuration ---
CLIENT_SECRETS_FILE = "AnyAgent_OAuth_1.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
FILE_ID = "1bS3Uz8SSm6c6CPd6pPhmSZnQMzW7hO-0"  # A file ID from your logs
TEMP_DIR = Path("./temp_video_downloads")

# --- Helper Functions (from server.py) ---
def log_message(message: str, is_error: bool = False):
    if is_error:
        print(message, file=sys.stderr)
    else:
        print(message, file=sys.stdout)

def get_google_creds(client_secrets_file: str) -> Credentials:
    creds = None
    token_path = Path("token.json")
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as e:
            log_message(f"WARNING: Could not load token.json: {e}. Re-authenticating.", is_error=True)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                log_message("-> Refreshing expired credentials...")
                creds.refresh(Request())
            except Exception as e:
                log_message(f"WARNING: Failed to refresh token, will re-authenticate: {e}", is_error=True)
                if token_path.exists(): token_path.unlink()
                creds = None
        if not creds:
            log_message("-> Performing new user authentication...")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                log_message(f"FATAL: Failed to run authentication flow from '{client_secrets_file}': {e}", is_error=True)
                return None
        try:
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            log_message(f"-> Credentials saved to {token_path}")
        except Exception as e:
            log_message(f"WARNING: Could not write token to {token_path}: {e}", is_error=True)
    return creds

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
def download_drive_file(drive_service, file_id: str, temp_dir: Path) -> Path:
    try:
        log_message(f"-> Attempting to download file: {file_id}")
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
                if status:
                    log_message(f"   - Download progress: {int(status.progress() * 100)}%")

        log_message(f"[+] SUCCESS: Download complete: {local_path.name}")
        return local_path
    except HttpError as e:
        if e.resp.status >= 500:
            log_message(f"   - Retrying download due to server error (5xx): {e}", is_error=True)
            raise
        raise RuntimeError(f"Non-retryable HTTP error downloading file {file_id}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_id} from Drive: {e}")

def main():
    """
    Main function to test Google Drive download.
    """
    log_message("--- Google Drive Download Test ---")
    
    # --- Create temp directory ---
    TEMP_DIR.mkdir(exist_ok=True)
    
    # --- Get credentials ---
    log_message("-> Getting Google credentials...")
    creds = get_google_creds(CLIENT_SECRETS_FILE)
    if not creds:
        log_message("[-] FAILURE: Could not get Google credentials.", is_error=True)
        return
    log_message("[+] Credentials obtained.")
    
    # --- Build Drive service ---
    log_message("-> Building Google Drive service...")
    drive_service = build('drive', 'v3', credentials=creds)
    log_message("[+] Drive service built.")
    
    # --- Download file ---
    try:
        download_drive_file(drive_service, FILE_ID, TEMP_DIR)
    except Exception as e:
        log_message(f"[-] FAILURE: An error occurred during download: {e}", is_error=True)
        import traceback
        traceback.print_exc()

    log_message("\n--- End of Test ---")

if __name__ == "__main__":
    main()
