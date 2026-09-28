import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.api.websockets import router as ws_router
from app.api.upstox_router import router as upstox_router
from app.api.strategies_router import router as strategies_router
from app.api.brokers_router import router as brokers_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NIFTY 50 top stocks to stream via DhanFeed on startup
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    On startup: start DhanFeed if credentials are configured.
    On shutdown: stop the feed cleanly.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    try:
        from app.services.dhan_data import dhan_data_service
        from app.services.dhan_feed import dhan_feed_manager

        if dhan_data_service.is_available():
            logger.info("Dhan HQ credentials detected — starting WebSocket feed...")
            dhan_feed_manager.start(_NIFTY50_WATCHLIST)
        else:
            logger.info(
                "Dhan HQ credentials not configured (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN). "
                "Live feed disabled; fallback data sources active."
            )
    except Exception as e:
        logger.warning(f"Dhan feed startup error (non-fatal): {e}")

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    try:
        from app.services.dhan_feed import dhan_feed_manager
        dhan_feed_manager.stop()
        logger.info("Dhan HQ feed stopped cleanly")
    except Exception as e:
        logger.debug(f"Dhan feed shutdown error (non-fatal): {e}")


app = FastAPI(
    title="ASTRA Trading API Sandbox",
    description="SaaS Backend for Indian & US Markets AI Trading Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allowed origins driven by FRONTEND_ORIGINS env (comma-separated).
# Defaults to localhost dev origins; production must set this explicitly.
import os as _os
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [o.strip() for o in _os.getenv("FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()]
# Security note: when allow_credentials=True, CORS spec FORBIDS allow_origins="*".
# We intentionally use an explicit allowlist instead.
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
    return {"status": "ok", "message": "ASTRA Trading API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
