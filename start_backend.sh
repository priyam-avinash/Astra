#!/bin/bash

# Navigate to the backend directory using absolute path
cd "/Users/avinashpriyam/.gemini/antigravity/scratch/algo-trading-app/backend"

echo "Setting up Python Virtual Environment..."
# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

echo "Installing required Python packages (FastAPI, Pandas, Scikit-Learn, etc.)..."
pip3 install -r requirements.txt

echo "Starting the FastAPI Backend Server..."
uvicorn main:app --reload
