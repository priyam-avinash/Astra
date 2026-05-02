"""
ASTRA Dhan HQ WebSocket Feed Manager (Phase 2)
===============================================
Manages a persistent DhanFeed WebSocket connection for real-time market data.
Provides sub-second price updates for subscribed NSE equity symbols.

Singleton manager instance: dhan_feed_manager
    feed_manager.start(["RELIANCE", "TCS", ...])
    price = feed_manager.get_price("RELIANCE")
    feed_manager.stop()

Graceful degradation: if credentials are missing / placeholders, all methods
return None / False without raising exceptions or blocking startup.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level price cache (also written into ai_predictor._PRICE_CACHE) ──
# {symbol: {"price": float, "ts": datetime, "source": "dhan_ws"}}
_LIVE_PRICES: dict = {}

# How long a cached tick is considered fresh (seconds)
_TICK_TTL_SEC = 5

# Reconnect backoff on disconnect (seconds)
_RECONNECT_BACKOFF_SEC = 5

# Dhan exchange segment integer for NSE
_NSE_INT = 1  # NSE=1, BSE=4


class DhanFeedManager:
    """
    Singleton that owns the DhanFeed WebSocket lifecycle.

    Public interface
    ----------------
    subscribe(symbol)          — add a symbol to the watchlist (pre-start)
    start(symbols)             — launch background asyncio feed task
    stop()                     — close the connection and cancel the task
    get_price(symbol) → float  — return cached LTP if < TTL, else None
    is_live() → bool           — True if feed is up and receiving data
    """

    def __init__(self):
        self._subscriptions: dict = {}   # symbol → security_id str
        self._live_prices: dict = _LIVE_PRICES
        self._task: Optional[asyncio.Task] = None
        self._connected: bool = False
        self._last_tick_ts: Optional[datetime] = None
        self._feed = None                # DhanFeed instance (set when running)
        self._stop_event: Optional[asyncio.Event] = None

    # ─────────────────────── Public API ─────────────────────────────────────

    def subscribe(self, symbol: str) -> bool:
        """
        Add a symbol to the subscription list.
        Resolves the security_id via dhan_data_service.
        Returns True if the symbol was resolved successfully.
        """
        try:
            from app.services.dhan_data import dhan_data_service
            sec_id, _ = dhan_data_service.resolve(symbol)
            if sec_id:
                self._subscriptions[symbol] = sec_id
                logger.debug(f"DhanFeed: subscribed {symbol} (id={sec_id})")
                return True
            else:
                logger.debug(f"DhanFeed: cannot resolve symbol '{symbol}'")
                return False
        except Exception as e:
            logger.debug(f"DhanFeed.subscribe({symbol}) error: {e}")
            return False

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Return the cached last traded price for a symbol if the tick is
        less than _TICK_TTL_SEC old, else None.
        """
        entry = self._live_prices.get(symbol)
        if entry is None:
            return None
        age = (datetime.now() - entry["ts"]).total_seconds()
        if age <= _TICK_TTL_SEC:
            return entry["price"]
        return None

    def is_live(self) -> bool:
        """True if the feed task is running and has received a tick recently."""
        if not self._connected:
            return False
        if self._last_tick_ts is None:
            return False
        age = (datetime.now() - self._last_tick_ts).total_seconds()
        return age < 30  # No tick for >30s → treat as stale

    def start(self, symbols: list) -> None:
        """
        Start the feed background task for the given list of symbols.
        Safe to call multiple times — no-ops if already running or credentials
        are missing.
        """
        try:
            from app.services.dhan_data import dhan_data_service
            if not dhan_data_service.is_available():
                logger.info("DhanFeed: credentials not configured — feed disabled")
                return
        except Exception as e:
            logger.debug(f"DhanFeed.start: dhan_data_service unavailable: {e}")
            return

        if self._task is not None and not self._task.done():
            logger.debug("DhanFeed: feed already running — ignoring duplicate start()")
            return

        # Resolve all symbols
        for sym in symbols:
            self.subscribe(sym)

        if not self._subscriptions:
            logger.warning("DhanFeed: no symbols resolved; feed not started")
            return

        # Get or create the running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. called from sync context in tests)
            logger.debug("DhanFeed.start: no running event loop; deferred start required")
            return

        self._stop_event = asyncio.Event()
        self._task = loop.create_task(self._feed_loop())
        logger.info(f"DhanFeed: background task started for {list(self._subscriptions.keys())}")

    def stop(self) -> None:
        """Gracefully stop the feed and close the WebSocket connection."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._connected = False
        logger.info("DhanFeed: feed stopped")

    # ─────────────────────── Internal feed loop ──────────────────────────────

    async def _feed_loop(self) -> None:
        """
        Main async loop: connect DhanFeed, read ticks, reconnect on failure.
        Runs until stop() is called or the task is cancelled.
        """
        import os

        client_id = os.getenv("DHAN_CLIENT_ID", "")
        access_token = os.getenv("DHAN_ACCESS_TOKEN", "")

        while self._stop_event is None or not self._stop_event.is_set():
            try:
                instruments = self._build_instruments()
                if not instruments:
                    logger.warning("DhanFeed: no instruments to subscribe; waiting...")
                    await asyncio.sleep(_RECONNECT_BACKOFF_SEC)
                    continue

                logger.info(f"DhanFeed: connecting with {len(instruments)} instrument(s)...")
                await self._run_feed_session(client_id, access_token, instruments)

            except asyncio.CancelledError:
                logger.info("DhanFeed: loop task cancelled")
                break
            except Exception as e:
                logger.warning(f"DhanFeed: connection error ({e}); reconnecting in {_RECONNECT_BACKOFF_SEC}s...")
                self._connected = False

            if self._stop_event is not None and self._stop_event.is_set():
                break

            await asyncio.sleep(_RECONNECT_BACKOFF_SEC)

        self._connected = False
        logger.info("DhanFeed: feed loop exited")

    async def _run_feed_session(self, client_id: str, access_token: str, instruments: list) -> None:
        """
        Open a single DhanFeed session, read ticks in a loop, update caches.
        Raises on connection failure so the outer loop can reconnect.
        """
        try:
            from dhanhq.marketfeed import DhanFeed, Ticker  # type: ignore
        except ImportError:
            logger.warning("dhanhq.marketfeed not available; DhanFeed disabled")
            # Sleep so the outer loop doesn't busy-spin
            await asyncio.sleep(60)
            return

        feed = None
        try:
            feed = DhanFeed(client_id, access_token, instruments, version="v1")
            self._feed = feed
            await feed.connect()
            self._connected = True
            logger.info("DhanFeed: WebSocket connected")

            while self._stop_event is None or not self._stop_event.is_set():
                try:
                    data = feed.get_data()
                    if data:
                        self._process_tick(data)
                    await asyncio.sleep(0.05)  # ~20 polls/sec
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"DhanFeed tick error: {e}")
                    break

        finally:
            self._connected = False
            if feed is not None:
                try:
                    await feed.close_connection()
                except Exception:
                    pass
            self._feed = None

    def _build_instruments(self) -> list:
        """
        Build the instruments list for DhanFeed.
        Format: [(exchange_segment_int, security_id_str, subscription_type_int), ...]
        Uses NSE=1, subscription type Ticker=15.
        """
        try:
            from dhanhq.marketfeed import Ticker  # type: ignore
            sub_type = Ticker  # 15
        except ImportError:
            sub_type = 15  # fallback constant

        instruments = []
        for symbol, sec_id in self._subscriptions.items():
            try:
                instruments.append((_NSE_INT, str(sec_id), sub_type))
            except Exception as e:
                logger.debug(f"DhanFeed: skipping {symbol}: {e}")
        return instruments

    def _process_tick(self, data: dict) -> None:
        """
        Parse a DhanFeed tick and update _live_prices + ai_predictor._PRICE_CACHE.
        """
        try:
            # DhanFeed returns a dict with keys like "security_id", "LTP", "last_price", etc.
            sec_id_str = str(data.get("security_id", data.get("securityId", "")))
            ltp_raw = (
                data.get("LTP")
                or data.get("last_price")
                or data.get("ltp")
                or data.get("LastTradedPrice")
            )
            if not sec_id_str or ltp_raw is None:
                return

            ltp = round(float(ltp_raw), 2)
            if ltp <= 0:
                return

            # Reverse-lookup symbol from security_id
            symbol = self._sec_id_to_symbol(sec_id_str)
            if symbol is None:
                return

            now = datetime.now()
            entry = {"price": ltp, "ts": now, "source": "dhan_ws"}
            self._live_prices[symbol] = entry
            self._last_tick_ts = now

            # Mirror into ai_predictor's price cache for get_realtime_price()
            try:
                from app.services.ai_predictor import _PRICE_CACHE
                _PRICE_CACHE[symbol] = (ltp, now)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"DhanFeed _process_tick error: {e}")

    def _sec_id_to_symbol(self, sec_id_str: str) -> Optional[str]:
        """Reverse-lookup: security_id string → symbol."""
        for sym, sid in self._subscriptions.items():
            if sid == sec_id_str:
                return sym
        return None


# ── Module-level singleton ───────────────────────────────────────────────────
dhan_feed_manager = DhanFeedManager()
