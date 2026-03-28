import requests
import json
import time
import subprocess
import os
import signal

def test_trade_execution():
    # Start the backend in the background
    backend_dir = "/Users/avinashpriyam/.gemini/antigravity/scratch/algo-trading-app/backend"
    
    # We'll try to run it using the venv if it exists, otherwise system python
    venv_python = os.path.join(backend_dir, ".venv/bin/python3")
    if not os.path.exists(venv_python):
        # Maybe check if .venv exists and is not absolute
        venv_python = "./.venv/bin/python3"
        if not os.path.exists(os.path.join(backend_dir, venv_python)):
            venv_python = "python3"
        
    print(f"Starting backend from {backend_dir}...")
    # Using python -m uvicorn since uvicorn might not be in the path
    backend_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for backend to start
    time.sleep(5)
    
    try:
        # Mock payload from frontend AutoModeView.jsx
        # NOTE: This assumes the "admin" user bypass in endpoints.py is working
        payload = { 
            "id": 1, 
            "asset": "RELIANCE.NS", 
            "action": "BUY", 
            "quantity": 50, 
            "price": 1400.0,
            "target_price": 1480.0,
            "stop_loss": 1350.0
        }
        
        print("Sending trade execution request...")
        response = requests.post("http://127.0.0.1:8001/api/execute", json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Trade executed successfully without crash!")
        else:
            print(f"FAILURE: Received status code {response.status_code}")
            
    except Exception as e:
        print(f"TEST ERROR: {e}")
    finally:
        print("Stopping backend...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()

if __name__ == "__main__":
    test_trade_execution()
