#!/usr/bin/env bash
# ASTRA Daily Startup
# ====================
# What this does, in order:
#   1. Starts the FastAPI backend in the background (port 8000)
#   2. Opens your browser to /upstox/login
#   3. Waits for you to finish the Upstox OAuth (token auto-saves to .env)
#   4. Runs the universe backtest with the fresh data
#   5. Backend stays up so the frontend can use it
#
# Usage:
#   cd backend && ./daily_start.sh
#
# Stop the backend later:
#   ./daily_stop.sh   (or just kill the process: pkill -f "python main.py")

set -e

cd "$(dirname "$0")"           # always run from backend/

# ── 1. Activate venv ────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "❌ .venv not found in backend/. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source .venv/bin/activate

# ── 2. Kill any existing backend on port 8000 ──────────────────────────────
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "⚠  Port 8000 already in use — killing existing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── 3. Start backend in background (via uvicorn — robust to missing __main__) ─
echo "🚀 Starting ASTRA backend on http://127.0.0.1:8000 ..."
nohup python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info > /tmp/astra_backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/astra_backend.pid

# Wait up to 30 seconds for the server to come up (Dhan feed init takes ~10s)
echo -n "   waiting for backend"
READY=0
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Backend ready after ${i}s (PID $BACKEND_PID)"
        READY=1
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

if [ "$READY" != "1" ]; then
    echo "❌ Backend failed to start. Last 30 log lines:"
    echo "─────────────────────────────────────────────────────────────────────"
    tail -30 /tmp/astra_backend.log
    echo "─────────────────────────────────────────────────────────────────────"
    echo ""
    read -p "Press ENTER to close..."
    exit 1
fi

# ── 4. Check Upstox token status ────────────────────────────────────────────
STATUS=$(curl -s http://127.0.0.1:8000/upstox/status)
AVAILABLE=$(echo "$STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin)['available'])")

if [ "$AVAILABLE" = "True" ]; then
    echo "✅ Upstox token is already valid — no login needed"
else
    echo ""
    echo "🔑 Upstox token missing/expired — opening browser for login..."
    echo "   After you finish login in the browser, come back here and press ENTER."
    echo ""
    open "http://127.0.0.1:8000/upstox/login"
    read -p "   Press ENTER when you see '✅ Upstox Connected' in the browser..."

    # Re-check
    AVAILABLE=$(curl -s http://127.0.0.1:8000/upstox/status | python3 -c "import sys, json; print(json.load(sys.stdin)['available'])")
    if [ "$AVAILABLE" != "True" ]; then
        echo "❌ Token still not valid. Check the browser tab and re-run /upstox/login manually."
        exit 1
    fi
    echo "✅ Upstox token saved"
fi

# ── 5. Run the daily backtest ──────────────────────────────────────────────
echo ""
echo "📊 Running daily universe backtest (30 days, top 10 per strategy)..."
echo ""
python run_universe_backtest.py --days 30 --top 10

# ── 6. Done ────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "✅ ASTRA is up and running."
echo ""
echo "   Backend log:     tail -f /tmp/astra_backend.log"
echo "   Stop backend:    double-click daily_stop.command"
echo "   Frontend:        cd ../frontend && npm run dev"
echo "─────────────────────────────────────────────────────────────────────"
echo ""
echo "You can close this window — the backend keeps running in the background."
read -p "Press ENTER to close this Terminal window..."
