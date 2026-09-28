# ASTRA — Automated Stock Trading & Research Assistant

AI research and **paper-trading** platform for Indian equities, crypto and commodities.
FastAPI backend + React (Vite) frontend. **No real-money orders exist in the code.**

## Quick start (macOS)

```bash
# 1. Backend (Python 3.10–3.12)
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional — every key is optional

# 2. Check your data feeds (prints the exact reason for any failure)
python check_data.py

# 3. Run everything
cd .. && bash start.command    # backend :8000, frontend :5173, opens the browser
```

In the app, **Settings → Market Data Providers** shows every provider's live
status and has a **Test all providers now** button. API: `GET /api/data/health?probe=true`.

## Market data

All data flows through `backend/app/services/market_data.py`.

| Asset | Provider order (first that works wins) | Keys needed |
|---|---|---|
| NSE/BSE equities | Dhan → Yahoo → yfinance → NSE site → Twelve Data → Alpha Vantage | none |
| Indices (^NSEI…) | Yahoo → yfinance | none |
| Crypto (BTC-USD…) | Binance public data → Yahoo → yfinance → Twelve Data | none |
| Commodities / FX | Yahoo → yfinance | none |

* Yahoo is called with a Chrome TLS fingerprint (`curl_cffi`). Plain Python
  requests get HTTP 429 from Yahoo.
* Failing providers go into a cooldown whose length depends on the reason (bad key 6h, quota 3h,
  network 30s→15min), so one dead source never slows the app down.
* When every provider fails, the last good data is served from disk and
  **flagged stale**. Paper orders are never filled at stale prices.
* Keys entered in Settings override `.env` and apply immediately.

## Paper trading

`broker.py` only contains `PaperTradingEngine`: fills at live LTP ± 0.05%
slippage, 0.03% brokerage. `PAPER_TRADING=false` is ignored. Stop-loss, target and trailing
stop-loss are monitored every 15 seconds, either by Celery beat (if Redis is running) or inside the API process.

## Tests

```bash
cd backend && pytest -q        # offline; providers are mocked with real response formats
```
