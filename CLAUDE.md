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
pytest -q                              # offline suite (providers mocked)
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

### Data Flow (v1.13)
```
GET /api/analyze/{symbol}?engine=astra_ml
  → ai_predictor._fetch_data()  → market_data.get_ohlcv(symbol, period, interval)
      normalize_symbol()          # "RELIANCE"→"RELIANCE.NS", "NIFTY"→"^NSEI"
      memory cache (TTL by interval, keyed on symbol+interval, period-aware)
      provider chain by asset class (see _CHAINS): dhan → yahoo(curl_cffi) → yfinance → nse → twelve_data → alpha_vantage
      ProviderHealth: per-provider cooldown on auth/quota/rate-limit/network errors
      disk cache backend/data/cache → served (flagged stale) when all providers fail
  → _compute_features()          # 20-feature DataFrame
  → [engine-specific inference]  # signal + ATR-based SL/TP
  → returns chartData + data_source + data_stale
```
**Never call yfinance/requests for market data directly — use `market_data`.**
Diagnostics: `python check_data.py`, `GET /api/data/health?probe=true`.
Secrets: `app/core/config.get_secret(name)` (Settings DB → env; placeholders ignored).

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
- `monitor_active_positions` — every 15s, handles TSL + auto-exit. Runs via Celery beat when Redis is reachable, otherwise as an asyncio loop inside the API (`main.py` lifespan). Skips stale prices.
- Use `tasks.dispatch(task, **kw)` instead of `.delay()` (falls back to a thread when Redis is down).

### Broker (`backend/app/services/broker.py`)
**Paper trading only.** `PaperTradingEngine` is the only broker; the live Dhan order path was removed in v1.13 and `PAPER_TRADING=false` is ignored. Fills use a live (non-stale) quote ± 0.05% slippage; if only stale data exists the order is rejected (HTTP 503). Dhan credentials are used for market data only (`dhan_data.py`, `dhan_feed.py`, dhanhq ≥ 2.1 `DhanContext` API).

---

## Key Implementation Details

### Auth
`AUTH_MODE=jwt` (default) requires a JWT on every data endpoint; `AUTH_MODE=bypass` (local dev only, set in your `.env`) injects an `admin_bypass` user. The frontend asks `GET /api/auth/mode` and shows the login screen only in jwt mode; `auth.installAuthFetch()` attaches the token to every `/api` call. Registration: `ASTRA_ALLOWED_USERS` (comma-separated emails) if set; otherwise open locally and, on hosted deploys, only the first account (owner) can register. The JWT key is `JWT_SECRET_KEY`, else derived from the Postgres URL.

### Vercel deploy (lite)
`vercel.json` builds `frontend/` as static files and serves the backend as one Python function (`api/index.py`, root `requirements.txt`). Serverless mode (`VERCEL`/`ASTRA_SERVERLESS`): writable data in `/tmp/astra-data`, no Dhan socket or background monitor (SL/TP exits run when positions are fetched), live prices poll `/api/quote` (`VITE_DISABLE_WS=1`). TensorFlow, scikit-learn and ccxt are left out to fit the 500 MB bundle: ASTRA.ML falls back to rules, ASTRA.AI runs from `saved_models/astra_rf_trees.npz` (numpy export, regenerated by `train_models.py`). Persist data by attaching Postgres (`DATABASE_URL`/`POSTGRES_URL`); without it SQLite in `/tmp` resets on cold starts.

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
New training code uses `GlobalAveragePooling1D` (safe to serialize). The committed `.keras` files predate that and contain a `Lambda(reduce_sum)` layer, which Keras 3 refuses and which can't be unmarshalled across Python versions; `services/model_compat.load_keras_model()` rebuilds them with an equivalent layer and loads the original weights. The committed crypto LSTM is a 1-output regressor, but the engine expects a 3-class classifier, so it is ignored until retrained (`python train_models.py`).

### LSTM Training Data Leakage — Fixed
Both equity and crypto LSTM now fit `RobustScaler` only on the training split (first 80% chronologically). The scaler is then applied to val without refitting. Do not revert to fitting on the full dataset.

---

## Environment Variables

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `DATABASE_URL` | No | SQLite | Set to `postgresql://...` for production |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Required for Celery |
| `JWT_SECRET_KEY` | No | hardcoded in auth.py | Change before production |
| `TWELVE_DATA_KEY` | No | — | Free plan: US stocks + crypto (NSE needs paid plan) |
| `ALPHA_VANTAGE_API_KEY` | No | — | 25 req/day, daily only. The old shared public key is blocked. |
| `DHAN_CLIENT_ID` / `DHAN_ACCESS_TOKEN` | No | — | Market data only (Data API plan). Never used for orders. |
| `ASTRA_ALLOW_SYNTHETIC` | No | false | Synthetic intraday bars for backtests (dev only, flagged) |

All keys can also be set in Settings (stored in `app_settings`), which override `.env`.
