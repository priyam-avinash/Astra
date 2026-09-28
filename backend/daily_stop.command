#!/usr/bin/env bash
# Stop the ASTRA backend cleanly
cd "$(dirname "$0")"

if [ -f /tmp/astra_backend.pid ]; then
    PID=$(cat /tmp/astra_backend.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && echo "✅ ASTRA backend stopped (PID $PID)"
    fi
    rm -f /tmp/astra_backend.pid
fi

# Cleanup any stragglers on port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo "✅ Cleared port 8000"
fi

echo ""
read -p "Press ENTER to close this Terminal window..."
