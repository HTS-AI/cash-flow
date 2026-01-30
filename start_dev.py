"""
Start both backend and frontend servers in parallel
Usage: python start_dev.py
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Global variables to store process references
backend_process = None
frontend_process = None

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shutdown both servers"""
    print("\n\n[INFO] Shutting down servers...")
    if backend_process:
        backend_process.terminate()
    if frontend_process:
        frontend_process.terminate()
    print("[INFO] Servers stopped. Goodbye!")
    sys.exit(0)

def main():
    """Main function to start both servers"""
    global backend_process, frontend_process
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 70)
    print("CASH FLOW PREDICTION SYSTEM - DEVELOPMENT SERVER")
    print("=" * 70)
    print("Starting both backend and frontend servers...")
    print("Backend:  http://localhost:5000")
    print("Frontend: http://localhost:3000")
    print("API Docs: http://localhost:5000/docs")
    print("=" * 70)
    print("\nPress Ctrl+C to stop both servers\n")
    
    # Check if Python is available
    try:
        subprocess.run([sys.executable, "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Python is not installed or not in PATH")
        sys.exit(1)
    
    # Check if Node.js is available
    try:
        subprocess.run(["npm", "--version"], 
                      capture_output=True, check=True, shell=sys.platform == 'win32')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Node.js/npm is not installed or not in PATH")
        sys.exit(1)
    
    # Check if frontend dependencies are installed
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[INFO] Installing frontend dependencies...")
        os.chdir(FRONTEND_DIR)
        subprocess.run(["npm", "install"], check=True, shell=sys.platform == 'win32')
        os.chdir(PROJECT_ROOT)
    
    # Start backend server
    print("[Backend] Starting FastAPI server on port 5000...")
    os.chdir(BACKEND_DIR)
    backend_process = subprocess.Popen(
        [sys.executable, "start_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    os.chdir(PROJECT_ROOT)
    
    # Give backend a moment to start
    time.sleep(2)
    
    # Start frontend server
    print("[Frontend] Starting React server on port 3000...")
    os.chdir(FRONTEND_DIR)
    frontend_process = subprocess.Popen(
        ["npm", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=sys.platform == 'win32'
    )
    os.chdir(PROJECT_ROOT)
    
    print("\n[SUCCESS] Both servers are running!")
    print("Backend logs will appear below...")
    print("-" * 70)
    
    # Monitor processes
    try:
        # Print backend output
        if backend_process.stdout:
            for line in backend_process.stdout:
                print(f"[Backend] {line.rstrip()}")
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    # Wait for processes to complete
    backend_process.wait()
    frontend_process.wait()

if __name__ == "__main__":
    main()
