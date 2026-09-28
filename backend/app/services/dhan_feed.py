"""
ASTRA Dhan HQ WebSocket Feed Manager
====================================
Streams live LTP ticks for a watchlist of NSE equities (market data only).

v1.13: dhanhq >= 2.1 removed `DhanFeed` in favour of `MarketFeed(DhanContext,
instruments, version="v2", on_message=…)`, which runs its own asyncio loop in
a background thread (`.start()`). The old code imported DhanFeed, hit
ImportError, and silently slept forever — so the live feed never worked.

    dhan_feed_manager.start(["RELIANCE", "TCS"])
    dhan_feed_manager.get_price("RELIANCE")   # → float | None (fresh ticks only)
    dhan_feed_manager.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_TICK_TTL_SEC = 10          # a tick older than this is not "live"
_RECONNECT_BACKOFF_SEC = 15
_NSE_EQ = 1                 # MarketFeed exchange code for NSE equities
_TICKER = 15                # MarketFeed.Ticker


class DhanFeedManager:
    def __init__(self):
        self._subscriptions: dict = {}          # symbol → security_id
        self._by_id: dict = {}                  # security_id → symbol
        self._live_prices: dict = {}            # symbol → {"price", "ts", "source"}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = False
        self._last_tick_ts: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._feed = None

    # ── public ───────────────────────────────────────────────────────────
    def subscribe(self, symbol: str) -> bool:
        from app.services.dhan_data import dhan_data_service
        sym = symbol.strip().upper().replace(".NS", "")
        sec_id, _ = dhan_data_service.resolve(sym)
        if sec_id:
            self._subscriptions[sym] = str(sec_id)
            self._by_id[str(sec_id)] = sym
            return True
        return False

    def get_price(self, symbol: str) -> Optional[float]:
        entry = self._live_prices.get(symbol.strip().upper().replace(".NS", ""))
        if entry and (datetime.now() - entry["ts"]).total_seconds() <= _TICK_TTL_SEC:
            return entry["price"]
        return None

    def is_live(self) -> bool:
        return bool(self._connected and self._last_tick_ts
                    and (datetime.now() - self._last_tick_ts).total_seconds() < 30)

    def status(self) -> dict:
        return {"running": bool(self._thread and self._thread.is_alive()),
                "connected": self._connected, "live": self.is_live(),
                "symbols": len(self._subscriptions),
                "last_tick": self._last_tick_ts.isoformat(timespec="seconds") if self._last_tick_ts else None,
                "last_error": self._last_error}

    def start(self, symbols: list) -> None:
        from app.services.dhan_data import dhan_data_service
        if not dhan_data_service.is_available():
            logger.info("DhanFeed: credentials not configured — live feed disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        for s in symbols:
            self.subscribe(s)
        if not self._subscriptions:
            logger.warning("DhanFeed: no symbols resolved; feed not started")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="dhan-feed", daemon=True)
        self._thread.start()
        logger.info(f"DhanFeed: started for {len(self._subscriptions)} symbols")

    def stop(self) -> None:
        self._stop.set()
        feed = self._feed
        if feed is not None:
            try:
                feed._running = False
            except Exception:
                pass
        self._connected = False

    # ── internals ────────────────────────────────────────────────────────
    def _loop(self):
        from app.services.dhan_data import dhan_data_service
        while not self._stop.is_set():
            try:
                from dhanhq.marketfeed import MarketFeed  # type: ignore
            except ImportError as e:
                self._last_error = f"dhanhq MarketFeed unavailable: {e}"
                logger.warning(f"DhanFeed: {self._last_error}")
                return
            ctx = dhan_data_service.context()
            if ctx is None:
                self._last_error = "DhanContext unavailable (credentials?)"
                return
            instruments = [(_NSE_EQ, sid, _TICKER) for sid in self._subscriptions.values()]
            try:
                feed = MarketFeed(ctx, instruments, version="v2",
                                  on_connect=self._on_connect, on_message=self._on_message,
                                  on_error=self._on_error)
                self._feed = feed
                feed.run()          # blocks until feed._running = False or fatal error
            except Exception as e:
                self._last_error = str(e)[:200]
                logger.warning(f"DhanFeed: disconnected ({e}); retrying in {_RECONNECT_BACKOFF_SEC}s")
            finally:
                self._connected = False
                self._feed = None
            self._stop.wait(_RECONNECT_BACKOFF_SEC)

    def _on_connect(self, _feed):
        self._connected = True
        self._last_error = None
        logger.info("DhanFeed: WebSocket connected")

    def _on_error(self, _feed, err):
        self._last_error = str(err)[:200]
        logger.debug(f"DhanFeed error: {err}")
        if self._stop.is_set():
            try:
                _feed._running = False
            except Exception:
                pass

    def _on_message(self, _feed, data):
        if self._stop.is_set():
            _feed._running = False
            return
        try:
            if not isinstance(data, dict):
                return
            sym = self._by_id.get(str(data.get("security_id", "")))
            ltp = data.get("LTP") or data.get("last_price")
            if not sym or ltp is None:
                return
            price = round(float(ltp), 2)
            if price <= 0:
                return
            now = datetime.now()
            self._live_prices[sym] = {"price": price, "ts": now, "source": "dhan_ws"}
            self._last_tick_ts = now
            try:
                from app.services.market_data import market_data
                market_data.put_live_price(sym, price, "dhan_ws")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"DhanFeed tick parse error: {e}")


dhan_feed_manager = DhanFeedManager()
