#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  ASTRA — One-Click Launcher
#  Double-click this file in Finder, or run:  bash start.command
# ════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV="$BACKEND_DIR/.venv/bin/python"
LOG_DIR="$SCRIPT_DIR/.logs"

mkdir -p "$LOG_DIR"

# ── Colours ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
  echo ""
  echo -e "${CYAN}${BOLD}"
  echo "  ███████╗ ███████╗████████╗ ██████╗  █████╗ "
  echo "  ██╔══██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗"
  echo "  ███████║╚█████╗    ██║   ██████╔╝███████║"
  echo "  ██╔══██║ ╚════██╗  ██║   ██╔══██╗██╔══██║"
  echo "  ██║  ██║███████╔╝  ██║   ██║  ██║██║  ██║"
  echo "  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝"
  echo -e "${RESET}${CYAN}       AI Algo Trading Platform — Launcher${RESET}"
  echo ""
}

step() { echo -e "${BOLD}${CYAN}▶ $1${RESET}"; }
ok()   { echo -e "  ${GREEN}✓ $1${RESET}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${RESET}"; }
fail() { echo -e "  ${RED}✗ $1${RESET}"; }

# ── Kill old processes ────────────────────────────────────────────────────
step "Stopping any existing ASTRA processes…"
pkill -f "uvicorn main:app"  2>/dev/null && ok "Backend stopped"   || true
pkill -f "vite"              2>/dev/null && ok "Frontend stopped"  || true
pkill -f "npm run dev"       2>/dev/null                           || true
sleep 1

# Force-free ports in case pkill wasn't enough
for PORT in 8000 5173; do
  PIDS=$(lsof -ti tcp:$PORT 2>/dev/null)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9 2>/dev/null
    ok "Force-freed port $PORT"
  fi
done
sleep 1

# ── Check deps ────────────────────────────────────────────────────────────
step "Checking dependencies…"

if [ ! -f "$VENV" ]; then
  fail "Python venv not found at $VENV"
  echo ""
  echo "  Run this once to set up the backend:"
  echo "  cd $BACKEND_DIR && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  echo ""
  read -p "Press Enter to exit…"
  exit 1
fi
ok "Python venv found"

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  step "node_modules missing — installing frontend packages (first run)…"
  cd "$FRONTEND_DIR" && npm install --silent
  ok "npm install complete"
fi
ok "Frontend node_modules ready"

# ── Check Redis (optional) ────────────────────────────────────────────────
if command -v redis-cli &>/dev/null && redis-cli ping &>/dev/null; then
  ok "Redis is running"
else
  warn "Redis not running — background workers will be disabled (paper trading still works)"
fi

# ── Start Backend ─────────────────────────────────────────────────────────
step "Starting ASTRA backend on port 8000…"
cd "$BACKEND_DIR"
"$VENV" -m uvicorn main:app \
  --host 0.0.0.0 --port 8000 \
  --reload \
  > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$LOG_DIR/backend.pid"

# ── Wait for backend ──────────────────────────────────────────────────────
step "Waiting for backend to become healthy…"
MAX_WAIT=30
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
  HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
  if [ "$HTTP" = "200" ]; then
    ok "Backend healthy at http://localhost:8000"
    break
  fi
  sleep 1
  ELAPSED=$((ELAPSED + 1))
  printf "  Waiting… (%ds)\r" "$ELAPSED"
done

if [ "$HTTP" != "200" ]; then
  fail "Backend did not start within ${MAX_WAIT}s"
  echo ""
  echo "  Check logs: tail -f $LOG_DIR/backend.log"
  echo ""
  # Still continue — frontend might still be useful
fi

# ── Start Frontend ────────────────────────────────────────────────────────
step "Starting ASTRA frontend on port 5173…"
cd "$FRONTEND_DIR"
npm run dev -- --port 5173 \
  > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$LOG_DIR/frontend.pid"

# Wait briefly for Vite to bind
sleep 3
ok "Frontend started at http://localhost:5173"

# ── Open browser ──────────────────────────────────────────────────────────
step "Opening ASTRA in your browser…"
sleep 1
open "http://localhost:5173"
ok "Browser opened"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  ASTRA is running!${RESET}"
echo ""
echo -e "  ${CYAN}Frontend:${RESET}  http://localhost:5173"
echo -e "  ${CYAN}Backend:${RESET}   http://localhost:8000"
echo -e "  ${CYAN}API Docs:${RESET}  http://localhost:8000/docs"
echo ""
echo -e "  ${YELLOW}Logs:${RESET}"
echo -e "    Backend:  tail -f $LOG_DIR/backend.log"
echo -e "    Frontend: tail -f $LOG_DIR/frontend.log"
echo ""
echo -e "  ${RED}Press Ctrl+C to stop both servers${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════${RESET}"
echo ""

# ── Graceful shutdown on Ctrl+C ───────────────────────────────────────────
cleanup() {
  echo ""
  step "Shutting down ASTRA…"
  kill $BACKEND_PID  2>/dev/null && ok "Backend stopped"  || true
  kill $FRONTEND_PID 2>/dev/null && ok "Frontend stopped" || true
  pkill -f "uvicorn main:app" 2>/dev/null || true
  pkill -f "vite"             2>/dev/null || true
  rm -f "$LOG_DIR/backend.pid" "$LOG_DIR/frontend.pid"
  echo ""
  echo -e "${CYAN}Goodbye!${RESET}"
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── Keep running (tail both logs to terminal) ─────────────────────────────
echo -e "${YELLOW}Live logs (Ctrl+C to stop ASTRA):${RESET}"
echo ""
tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" &
TAIL_PID=$!

# Wait until Ctrl+C — uvicorn --reload forks so we can't rely on $BACKEND_PID
while true; do
  sleep 5
  # Auto-exit if both processes have died
  if ! kill -0 $BACKEND_PID 2>/dev/null && ! kill -0 $FRONTEND_PID 2>/dev/null; then
    fail "Both servers exited unexpectedly."
    kill $TAIL_PID 2>/dev/null
    cleanup
  fi
done
