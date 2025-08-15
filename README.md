# AnyAI Video Analysis Dashboard

This project is a Flask web server that uses the Gemini API to analyze video files linked in a Google Sheet. It provides a simple web dashboard to start, stop, and monitor the analysis process, which runs in the background.

## Setup and Run Instructions

### Prerequisites

Before you begin, ensure you have the following files ready:

1.  **`client_secrets.json`**: This is your Google Cloud OAuth 2.0 credential file. It must be placed inside the `credentials/` directory.
2.  **Gemini API Key**: You need a valid API key from Google AI Studio.

### Installation and Execution

1.  **Create an Environment File**: In the main project folder, create a file named `.env` and add your Gemini API key to it like this:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
2.  **Open the Terminal**: Open the **Terminal** app on your Mac.
3.  **Navigate to the Project Folder**: A simple way to do this is to type `cd ` (with a space after it) into the terminal, and then drag the `AnyAI_video_analysis` folder from Finder directly onto the terminal window. Press Enter.
4.  **Create and Activate a Python Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
5.  **Install the Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run the Application**:
    -   From now on, you can simply **double-click the `start.command`** file in the project folder.
    -   This will open a new terminal window, start the server, and open the web interface in your browser.
    -   To stop the server, just close the terminal window.

### First Run Authentication

The first time you run the application, your web browser will automatically open a Google authentication page. Log in with your Google account and grant the application permission. After you approve, a `token.json` file will be automatically created in your `credentials/` directory. This token will be used for all future runs, so you only need to do this once.