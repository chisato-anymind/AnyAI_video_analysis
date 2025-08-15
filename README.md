# AnyAI Video Analysis Dashboard

This project is a Flask web server that uses the Gemini API to analyze video files linked in a Google Sheet. It provides a simple web dashboard to start, stop, and monitor the analysis process, which runs in the background.

## Setup Instructions for Collaborators

To run this project, you will need to set up your own Google Cloud credentials to allow the application to access Google Drive and Google Sheets on your behalf. This is a one-time setup.

### Step 1: Configure a Google Cloud Project

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a new project or select an existing one.
3.  Enable the **Google Drive API** and **Google Sheets API** for your project.
    -   In the top search bar, type "Google Drive API" and select it. Click the **"Enable"** button.
    -   Do the same for the "Google Sheets API".

### Step 2: Create OAuth 2.0 Credentials

1.  In the Google Cloud Console, navigate to **"APIs & Services" > "OAuth consent screen"**.
2.  Choose **"External"** and click **"Create"**.
3.  Fill out the required fields:
    -   **App name:** `AnyAI Video Analysis` (or a name of your choice).
    -   **User support email:** Your email address.
    -   **Developer contact information:** Your email address.
4.  Click **"Save and Continue"** through the "Scopes" and "Test users" sections. You do not need to add anything here. Finally, click **"Back to Dashboard"**.
5.  Now, go to **"APIs & Services" > "Credentials"**.
6.  Click **"+ CREATE CREDENTIALS"** at the top and select **"OAuth client ID"**.
7.  For the **"Application type"**, select **"Desktop app"**.
8.  Give it a name (e.g., "Video Analysis Desktop Client") and click **"Create"**.
9.  A window will appear with your credentials. Click the **"DOWNLOAD JSON"** button.

### Step 3: Place the Credentials File in the Project

1.  Rename the file you just downloaded to `client_secrets.json`.
2.  Move this `client_secrets.json` file into the `credentials/` directory at the root of this project.

The project is already configured to keep this file private and will not commit it to Git.

### Step 4: Install Dependencies and Run

1.  Open the **Terminal** app on your Mac.
2.  **Navigate to the project folder.** A simple way to do this is to type `cd ` (with a space after it) into the terminal, and then drag the `AnyAI_video_analysis` folder from Finder directly onto the terminal window. Press Enter.
3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Create a `.env` file** in the root of the project and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
6.  **Run the application:**
    -   From now on, you can simply **double-click the `start.command`** file in the project folder.
    -   This will open a new terminal window, start the server, and open the web interface in your browser.
    -   To stop the server, just close the terminal window.

### First Run Authentication

The first time you run the application, your web browser will automatically open a Google authentication page. Log in with your Google account and grant the application permission. After you approve, a `token.json` file will be automatically created in your `credentials/` directory. This token will be used for all future runs, so you only need to do this once.
