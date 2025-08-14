import sys
import subprocess

print("--- Python Environment Diagnostic ---")

# Print Python executable path
print(f"[*] Python Executable: {sys.executable}")

# Print Python version
print(f"[*] Python Version: {sys.version}")

# Try to import the problematic package
print("\n[*] Attempting to import 'google.genai'...")
try:
    import google.genai
    print("[+] SUCCESS: 'google.genai' imported successfully.")
    print(f"    - Location: {google.genai.__file__}")
except ImportError as e:
    print(f"[!] FAILED: Could not import 'google.genai'.")
    print(f"    - Error: {e}")
except Exception as e:
    print(f"[!] FAILED: An unexpected error occurred during import.")
    print(f"    - Error: {e}")


# List all installed packages in this environment
print("\n[*] Listing installed packages for this environment...")
try:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'list'])
    print(reqs.decode())
except Exception as e:
    print(f"[!] FAILED: Could not list packages using 'pip list'.")
    print(f"    - Error: {e}")

print("--- End of Diagnostic ---")