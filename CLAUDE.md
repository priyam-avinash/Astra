# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ASTRA — Automated Stock Trading & Research Assistant

Full-stack algo trading platform for Indian equities and crypto. Three AI prediction engines, real-time WebSocket feeds, position lifecycle management, and Dhan broker integration.

---

## Dev Commands

### Backend (FastAPI)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                          # Runs on http://localhost:8000
```

Without PostgreSQL, the app auto-falls back to SQLite (`astra_trading.db`) — no setup needed for local dev.

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev                             # Runs on http://localhost:5173
npm run build                           # Production build
```

### Retrain ML Models
```bash
cd backend
source .venv/bin/activate
python train_models.py                  # Trains RF + Equity LSTM + Crypto LSTM
```
Models are saved to `backend/app/models/saved_models/`. Restart backend after retraining.

### Run Tests
```bash
cd backend
source .venv/bin/activate
pytest tests/test_api.py -v            # Only 2 tests exist (root + health)
```

### Docker (Full Stack)
```bash
docker-compose up -d                    # PostgreSQL + Redis + Backend + Worker + Frontend
docker-compose logs -f backend          # Stream backend logs
docker-compose logs -f worker           # Stream Celery worker logs
```

### Celery Worker (standalone)
```bash
cd backend && source .venv/bin/activate
celery -A app.services.tasks.celery_app worker --loglevel=info
celery -A app.services.tasks.celery_app beat   # For periodic tasks (position monitoring)
```

### Debug Data Fetching
```bash
cd backend && source .venv/bin/activate
python -c "import yfinance as yf; print(yf.download('TCS.NS', period='6mo'))"
python -c "from app.services.ai_predictor import ai_engine; print(ai_engine.analyze_market_data('TCS.NS'))"
```

---

## Architecture

### Three Prediction Engines (all in `backend/app/services/ai_predictor.py`)

| Engine | Selector (`engine=`) | File | Description |
|--------|---------------------|------|-------------|
| ASTRA 1.0 | `astra` | `ai_predictor.py` | Rule-based 4/6 multi-confirmation system |
| ASTRA.AI | `astra_ai` | `ai_predictor.py` | Random Forest, 20 features, 5-day return target |
| ASTRA.ML | `astra_ml` | `ai_predictor.py` | Bidirectional LSTM (lookback=30), 3-day return target |
| ASTRA.CRYPTO | `astra_crypto` | `crypto_engine.py` | Regime-aware rules (Donchian/BB) + 3-class BiLSTM direction classifier |

All engines share `_compute_features()` (20 features: SMA distances, RSI, MACD, ADX, Stochastic, BB, Donchian, Volume, OBV, Candlestick geometry). Inference always uses `analyze_market_data(symbol, engine=...)` which returns `{signal, confidence, entry_price, target, stop_loss, chartData}`.

### Data Flow
```
GET /api/analyze/{symbol}?engine=astra_ml
  → ai_predictor._fetch_data()          # 4-source fallback: Twelve Data → AlphaVantage → yfinance → yfinance+.NS
  → data_cache.cache_get/put()          # Disk cache at backend/data/cache/ (8h TTL for daily data)
  → _compute_features()                 # 20-feature DataFrame
  → [engine-specific inference]         # signal + ATR-based SL/TP
  → returns chartData (last 180 bars) + indicators
```

### Signal → Trade Lifecycle
```
AI Analysis → TargetSignal (DB, status="Pending Approval")
  → User approves in UI (AutoModeView)
  → POST /api/execute → ActivePosition (DB, status="OPEN")
  → Celery monitor_active_positions (every 15s)
      → checks SL/TP, moves SL to breakeven at 50% target (TSL)
      → auto-square-off → TradeRecord (DB, status="Closed (Worker)")
```

### Database (SQLAlchemy ORM — `backend/app/models/database.py`)
Four tables: `users`, `active_signals` (TargetSignal), `active_positions` (ActivePosition), `trade_history` (TradeRecord). Local dev uses SQLite auto-created at startup. Docker uses PostgreSQL.

### Celery (`backend/app/services/tasks.py`)
- `analyze_asset` — on-demand async analysis task
- `monitor_active_positions` — periodic every 15s, handles TSL + auto-exit

### Broker (`backend/app/services/broker.py`)
Dhan API integration. Reads `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` from `.env`. Falls back to **mock mode** (returns `MOCK-TRD-*` order IDs) when credentials are absent. **Known issue:** Dhan requires numeric `security_id`, not ticker symbols — a lookup table is needed for live execution.

---

## Key Implementation Details

### Auth Bypass (Dev Mode)
`backend/app/api/endpoints.py:21` — `get_current_user()` always returns a hardcoded `admin_bypass` user. All 66 endpoints ignore JWT. The real JWT validation code exists in `auth.py` but is bypassed. Frontend also hard-wires `isLoggedIn=true` in `App.jsx:20`.

### Macro Trend Filter
`_is_macro_bullish()` fetches NIFTY 50 (`^NSEI`) daily data, checks if Close > SMA-200. Result cached 1 hour. BUY signals are blocked (set to HOLD) in bear regime. Requires internet on startup.

### ATR-Based SL/TP Multipliers
- ASTRA 1.0: SL=1.5×ATR, TP=3.0×ATR
- ASTRA.AI: SL=1.6×ATR, TP=4.0×ATR
- ASTRA.ML: SL=1.5×ATR, TP=3.5×ATR
- Crypto major/altcoin/meme: different multipliers in `crypto_engine.py:RISK_CONFIG`

### Crypto Regime Detection (`crypto_engine.py`)
ADX ≥ 25 → Momentum/Breakout mode (Donchian channel). ADX < 20 → Mean Reversion mode (Bollinger Band extremes). Fear & Greed Index and BTC dominance are additional macro filters for crypto BUY signals.

### SMA-200 on Short Datasets
When fewer than 200 bars are available, `_compute_features()` back-fills `SMA_200` using `expanding().mean()` rather than substituting SMA-30. Always fetch `period="2y"` or longer when training or running ML engines.

### Model Serialization
Equity LSTM uses `GlobalAveragePooling1D` (safe to serialize). Crypto LSTM was fixed from a `Lambda` layer to `GlobalAveragePooling1D` — **retrain crypto LSTM** if using a model file built before this fix. Model files: `backend/app/models/saved_models/*.keras` and `*_scaler.joblib`.

### LSTM Training Data Leakage — Fixed
Both equity and crypto LSTM now fit `RobustScaler` only on the training split (first 80% chronologically). The scaler is then applied to val without refitting. Do not revert to fitting on the full dataset.

---

## Environment Variables

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `DATABASE_URL` | No | SQLite | Set to `postgresql://...` for production |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Required for Celery |
| `JWT_SECRET_KEY` | No | hardcoded in auth.py | Change before production |
| `TWELVE_DATA_KEY` | No | — | 800 req/day free, best for NSE stocks |
| `ALPHA_VANTAGE_KEY` | No | `XV1FMHS5UHPIIPAZ` (free fallback) | 25 req/day |
| `DHAN_CLIENT_ID` | No | — | Omit for mock trading mode |
| `DHAN_ACCESS_TOKEN` | No | — | Omit for mock trading mode |
