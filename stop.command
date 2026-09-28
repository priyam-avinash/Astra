#!/usr/bin/env bash
# ASTRA Stop — Closes the backend (port 8000) and the frontend (port 5173)
# Place at: react-algo-trading-app/stop.command
# Usage:    double-click from Finder, or run from terminal

set +e
cd "$(dirname "$0")"

echo "🛑 Stopping ASTRA services..."
echo ""

KILLED_ANY=0

# ── 1. Kill backend via pid file (cleanest) ──────────────────────────────
if [ -f /tmp/astra_backend.pid ]; then
    PID=$(cat /tmp/astra_backend.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        sleep 1
        # Force-kill if still alive
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null
        fi
        echo "✅ Backend stopped (PID $PID)"
        KILLED_ANY=1
    fi
    rm -f /tmp/astra_backend.pid
fi

# ── 2. Kill anything still listening on backend port 8000 ─────────────────
PIDS_8000=$(lsof -ti:8000 2>/dev/null)
if [ -n "$PIDS_8000" ]; then
    echo "$PIDS_8000" | xargs kill -9 2>/dev/null
    echo "✅ Cleared port 8000 (PIDs: $PIDS_8000)"
    KILLED_ANY=1
fi

# ── 3. Kill anything listening on frontend port 5173 (Vite default) ───────
PIDS_5173=$(lsof -ti:5173 2>/dev/null)
if [ -n "$PIDS_5173" ]; then
    echo "$PIDS_5173" | xargs kill -9 2>/dev/null
    echo "✅ Cleared port 5173 / frontend (PIDs: $PIDS_5173)"
    KILLED_ANY=1
fi

# ── 4. Kill any straggling 'python main.py' or uvicorn invocations ────────
PYTHON_PIDS=$(pgrep -f "python.*main\.py|uvicorn main:app" 2>/dev/null)
if [ -n "$PYTHON_PIDS" ]; then
    echo "$PYTHON_PIDS" | xargs kill -9 2>/dev/null
    echo "✅ Killed straggler python/uvicorn processes (PIDs: $PYTHON_PIDS)"
    KILLED_ANY=1
fi

# ── 5. Kill any vite/npm-dev processes spawned from frontend/ ─────────────
VITE_PIDS=$(pgrep -f "vite|npm.*dev" 2>/dev/null)
if [ -n "$VITE_PIDS" ]; then
    echo "$VITE_PIDS" | xargs kill -9 2>/dev/null
    echo "✅ Killed Vite/npm-dev processes (PIDs: $VITE_PIDS)"
    KILLED_ANY=1
fi

echo ""
if [ "$KILLED_ANY" = "0" ]; then
    echo "ℹ  Nothing was running — ASTRA is already stopped."
else
    echo "✅ All ASTRA services stopped."
fi

echo ""
read -p "Press ENTER to close this Terminal window..."
