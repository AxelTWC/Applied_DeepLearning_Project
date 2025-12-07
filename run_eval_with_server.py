"""
Run eval.py with mock LLM server in parallel threads
"""
import subprocess
import time
import sys
import threading

def run_server():
    """Run mock server in background"""
    proc = subprocess.Popen(
        [sys.executable, "server_mock.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Print server output
    for line in proc.stdout:
        print(f"[SERVER] {line}", end='')
    return proc

def run_eval():
    """Run eval.py after server is ready"""
    time.sleep(2)  # Wait for server to start
    proc = subprocess.Popen(
        [sys.executable, "eval.py", "--base-url", "http://localhost:8000"]
    )
    return proc

if __name__ == "__main__":
    print("Starting mock LLM server and eval.py...\n")
    
    server_proc = run_server()
    eval_proc = run_eval()
    
    try:
        eval_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        eval_proc.terminate()
        server_proc.terminate()
