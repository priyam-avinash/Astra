import app.core.config  # noqa: F401  — must be first: loads backend/.env and ./.env

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.api.websockets import router as ws_router
from app.api.upstox_router import router as upstox_router
from app.api.strategies_router import router as strategies_router
from app.api.brokers_router import router as brokers_router

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
# yfinance logs every failed ticker at ERROR; our data layer reports these itself
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# NIFTY 50 stocks streamed via the Dhan WebSocket (only if Dhan is configured)
_NIFTY50_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "SUNPHARMA", "BAJFINANCE", "WIPRO", "ULTRACEMCO", "TITAN",
    "NESTLEIND", "POWERGRID", "NTPC", "ONGC", "TECHM",
    "BAJAJFINSV", "TATASTEEL", "HINDALCO", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "BPCL", "CIPLA", "HEROMOTOCO", "INDUSINDBK",
    "M&M", "BRITANNIA", "TATACONSUM", "APOLLOHOSP", "BAJAJ-AUTO",
    "LTIM", "SBILIFE", "HDFCLIFE", "TATAMOTORS", "UPL",
]

_MONITOR_INTERVAL_SEC = int(os.getenv("POSITION_MONITOR_INTERVAL", "15"))


async def _position_monitor_loop():
    """SL/TP/trailing-SL monitor that runs inside the API process.

    v1.12 relied on Celery beat for this, so without Redis open positions were
    never auto-closed. This loop is used whenever Redis isn't reachable (or
    ASTRA_INPROCESS_MONITOR=true)."""
    from app.services.tasks import monitor_active_positions
    logger.info(f"Position monitor: running in-process every {_MONITOR_INTERVAL_SEC}s")
    while True:
        try:
            await asyncio.to_thread(monitor_active_positions)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Position monitor cycle failed: {e}")
        await asyncio.sleep(_MONITOR_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.config import ENV_FILES_LOADED, env_flag
    from app.services.market_data import HAVE_CURL_CFFI
    logger.info(f"ASTRA starting — PAPER TRADING ONLY | env files: {ENV_FILES_LOADED or 'none'} "
                f"| curl_cffi: {HAVE_CURL_CFFI}")
    if not HAVE_CURL_CFFI:
        logger.warning("curl_cffi not installed — Yahoo may answer HTTP 429. Run: pip install curl_cffi")

    from app.core.config import SERVERLESS
    if SERVERLESS:
        # No long-running process on Vercel: no Dhan socket, and the SL/TP
        # monitor runs on demand when positions are fetched (see endpoints).
        logger.info("Serverless mode: live feed and background monitor disabled")
        yield
        return

    # Dhan live price feed (market data only)
    try:
        from app.services.dhan_feed import dhan_feed_manager
        dhan_feed_manager.start(_NIFTY50_WATCHLIST)
    except Exception as e:
        logger.warning(f"Dhan feed startup error (non-fatal): {e}")

    # Position monitor: in-process unless a Celery broker is available
    monitor_task = None
    try:
        from app.services.tasks import broker_available
        use_inprocess = env_flag("ASTRA_INPROCESS_MONITOR", not broker_available())
    except Exception:
        use_inprocess = True
    if use_inprocess:
        monitor_task = asyncio.create_task(_position_monitor_loop())
    else:
        logger.info("Position monitor: delegated to Celery beat (Redis reachable)")

    yield

    if monitor_task:
        monitor_task.cancel()
    try:
        from app.services.dhan_feed import dhan_feed_manager
        dhan_feed_manager.stop()
    except Exception:
        pass


app = FastAPI(
    title="ASTRA Trading API (Paper Trading)",
    description="AI research & paper-trading backend for Indian equities, crypto and commodities",
    version="1.13.0",
    lifespan=lifespan,
)

# CORS — explicit allowlist from FRONTEND_ORIGINS (comma-separated).
# With allow_credentials=True the CORS spec forbids allow_origins="*".
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [o.strip() for o in os.getenv("FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS allowed origins: {_origins}")

app.include_router(api_router)
app.include_router(ws_router)
app.include_router(upstox_router)
app.include_router(strategies_router)
app.include_router(brokers_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "ASTRA Trading API is running", "mode": "PAPER"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")), log_level="info")
