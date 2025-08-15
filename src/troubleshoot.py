import sys
import ssl
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager

class ForceTLSV12Adapter(HTTPAdapter):
    """A custom HTTP adapter to force TLS 1.2."""
    def __init__(self, *args, **kwargs):
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context
        )

def check_ssl():
    """
    Performs a series of checks to diagnose SSL/TLS issues.
    """
    print("--- SSL/TLS Troubleshooting ---")
    print(f"Python version: {sys.version}")
    print(f"SSL version: {ssl.OPENSSL_VERSION}")

    # --- Check 1: Standard connection to Google ---
    try:
        print("\n[1] Attempting standard connection to https://www.google.com...")
        response = requests.get("https://www.google.com", timeout=15)
        if response.status_code == 200:
            print("[+] SUCCESS: Standard connection to Google works.")
        else:
            print(f"[-] WARNING: Received status code {response.status_code} from Google.")
    except requests.exceptions.RequestException as e:
        print(f"[-] FAILURE: Could not connect to Google. Error: {e}")
        print("    This indicates a potential network issue, proxy problem, or a firewall blocking the connection.")

    # --- Check 2: Connection with forced TLS 1.2 ---
    try:
        print("\n[2] Attempting connection with forced TLS 1.2...")
        session = requests.Session()
        session.mount("https://", ForceTLSV12Adapter())
        response = session.get("https://www.google.com", timeout=15)
        if response.status_code == 200:
            print("[+] SUCCESS: Connection with forced TLS 1.2 works.")
        else:
            print(f"[-] WARNING: Received status code {response.status_code} with forced TLS 1.2.")
    except requests.exceptions.RequestException as e:
        print(f"[-] FAILURE: Could not connect with forced TLS 1.2. Error: {e}")
        print("    This could indicate that your system's SSL libraries do not support TLS 1.2, which is unlikely but possible.")

    # --- Check 3: Detailed SSL socket connection ---
    try:
        print("\n[3] Attempting a direct SSL socket connection to google.com:443...")
        context = ssl.create_default_context()
        with socket.create_connection(("www.google.com", 443), timeout=15) as sock:
            with context.wrap_socket(sock, server_hostname="www.google.com") as ssock:
                print("[+] SUCCESS: Direct SSL socket connection established.")
                print(f"    - Protocol: {ssock.version()}")
                print(f"    - Cipher: {ssock.cipher()}")
    except Exception as e:
        print(f"[-] FAILURE: Direct SSL socket connection failed. Error: {e}")
        print("    This is a strong indicator of a low-level SSL/TLS issue on your system or network.")

    print("\n--- End of Troubleshooting ---")

if __name__ == "__main__":
    check_ssl()