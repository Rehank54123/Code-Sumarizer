import os
import subprocess
import sys

if __name__ == "__main__":
    print("Forwarding to src/train.py...")
    # Add src to python path for imports
    env = os.environ.copy()
    src_path = os.path.join(os.getcwd(), "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    subprocess.run([sys.executable, "src/train.py"], env=env)
