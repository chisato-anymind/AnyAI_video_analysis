import sys
import os
import site

print("--- Advanced Environment Diagnostic ---")

# 1. Check for PYTHONPATH environment variable
print("\n[1] Checking for PYTHONPATH environment variable...")
pythonpath = os.environ.get('PYTHONPATH')
if pythonpath:
    print(f"[!] WARNING: PYTHONPATH is set. This can cause major conflicts.")
    print(f"    PYTHONPATH = {pythonpath}")
else:
    print("[+] SUCCESS: PYTHONPATH is not set, which is good.")

# 2. Locate the top-level 'google' package
print("\n[2] Locating the 'google' namespace package...")
try:
    import google
    google_path = google.__path__[0] # Use __path__ for namespace packages
    print(f"[+] Found 'google' package at: {google_path}")

    # 3. Inspect the contents of the 'google' package directory
    print("\n[3] Inspecting the 'google' package directory...")
    try:
        contents = os.listdir(google_path)
        print(f"    - Contains {len(contents)} items.")

        if 'genai' in contents:
            print("[+] SUCCESS: 'genai' sub-directory exists inside the 'google' package.")
        else:
            print("[!] FAILED: 'genai' sub-directory is MISSING from the 'google' package.")

        if '__init__.py' in contents:
            print(f"[!] WARNING: Found an '__init__.py' file in the google namespace directory.")
            print(f"    This can break how modules are loaded. This is a likely source of the problem.")
        else:
            print("[+] SUCCESS: No '__init__.py' file found in the namespace, which is correct.")

    except Exception as e:
        print(f"[!] Error inspecting directory {google_path}: {e}")

except ImportError:
    print("[!] FAILED: Could not import the top-level 'google' package at all.")
except Exception as e:
    print(f"[!] An unexpected error occurred: {e}")


print("\n--- End of Advanced Diagnostic ---")
