import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.api.websockets import router as ws_router

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "ASTRA Trading API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
