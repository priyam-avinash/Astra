#!/bin/bash
# Using the locally downloaded Node.js for compatibility
export PATH="/Users/avinashpriyam/.gemini/antigravity/scratch/node-v20.11.1-darwin-arm64/bin:$PATH"

# Navigate to the frontend directory using absolute path
cd "/Users/avinashpriyam/.gemini/antigravity/scratch/algo-trading-app/frontend"

echo "Starting the React UI Development Server..."
npm run dev
