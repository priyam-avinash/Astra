from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
import os
from datetime import datetime

import redis.asyncio as aioredis # type: ignore

router = APIRouter()
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Fallback poll interval (seconds) when DhanFeed is not live
_FALLBACK_POLL_INTERVAL = 15


@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    await websocket.accept()
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("market_ticks")

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                except Exception:
                    pass
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        await pubsub.unsubscribe("market_ticks")
        await redis_client.aclose()
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try:
            await pubsub.unsubscribe("market_ticks")
            await redis_client.aclose()
        except:
            pass


@router.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    Real-time price feed endpoint.

    Protocol
    --------
    On connect:
        → {"type": "snapshot", "prices": {"RELIANCE": {"price": 2885.5, "source": "dhan_ws", "ts": "..."}, ...}}

    Every second (while DhanFeed is live) or every 15s (fallback):
        → {"type": "tick", "symbol": "RELIANCE", "price": 2885.5, "source": "dhan_ws"|"cache", "ts": "..."}

    Client → server:
        {"action": "subscribe", "symbol": "RELIANCE"}
    """
    await websocket.accept()
    logger.info("WS /ws/prices: client connected")

    # Import feed manager and fallback engine (graceful — never crash on import)
    try:
        from app.services.dhan_feed import dhan_feed_manager
    except Exception:
        dhan_feed_manager = None  # type: ignore

    try:
        from app.services.ai_predictor import ai_engine
    except Exception:
        ai_engine = None  # type: ignore

    # Active symbol set for this connection
    subscribed_symbols: set = set()

    # Add any symbols already tracked by the feed manager
    if dhan_feed_manager is not None:
        for sym in dhan_feed_manager._subscriptions:
            subscribed_symbols.add(sym)

    def _get_price_entry(symbol: str) -> dict:
        """Get the best available price for a symbol with metadata."""
        # 1. Try DhanFeed live price (freshest)
        if dhan_feed_manager is not None:
            live = dhan_feed_manager.get_price(symbol)
            if live and live > 0:
                return {"price": live, "source": "dhan_ws", "ts": datetime.now().isoformat()}
        # 2. Fallback: ai_engine real-time price (may use cache or external APIs)
        if ai_engine is not None:
            try:
                p = ai_engine.get_realtime_price(symbol)
                if p and p > 0:
                    return {"price": p, "source": "cache", "ts": datetime.now().isoformat()}
            except Exception:
                pass
        return {}

    # Send initial snapshot
    try:
        snapshot_prices = {}
        for sym in subscribed_symbols:
            entry = _get_price_entry(sym)
            if entry:
                snapshot_prices[sym] = entry
        await websocket.send_json({"type": "snapshot", "prices": snapshot_prices})
    except Exception as e:
        logger.debug(f"WS /ws/prices: snapshot send error: {e}")

    # Track last-sent prices to detect changes
    last_sent: dict = {}
    last_fallback_poll = datetime.now()

    async def _receive_loop():
        """Coroutine: handle incoming client messages (subscribe actions)."""
        nonlocal subscribed_symbols
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                    if msg.get("action") == "subscribe":
                        sym = str(msg.get("symbol", "")).strip().upper()
                        if sym and sym not in subscribed_symbols:
                            subscribed_symbols.add(sym)
                            # Also subscribe in feed manager for future connections
                            if dhan_feed_manager is not None:
                                dhan_feed_manager.subscribe(sym)
                            logger.debug(f"WS /ws/prices: client subscribed to {sym}")
                            # Send immediate price if available
                            entry = _get_price_entry(sym)
                            if entry:
                                await websocket.send_json({
                                    "type": "tick",
                                    "symbol": sym,
                                    **entry,
                                })
                except Exception:
                    pass
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug(f"WS /ws/prices: receive loop error: {e}")

    receive_task = asyncio.create_task(_receive_loop())

    try:
        while True:
            now = datetime.now()
            feed_live = dhan_feed_manager is not None and dhan_feed_manager.is_live()

            if feed_live:
                # Push any changed prices every 1 second
                for sym in list(subscribed_symbols):
                    entry = _get_price_entry(sym)
                    if not entry:
                        continue
                    prev_price = last_sent.get(sym)
                    curr_price = entry.get("price")
                    if curr_price != prev_price:
                        try:
                            await websocket.send_json({
                                "type": "tick",
                                "symbol": sym,
                                **entry,
                            })
                            last_sent[sym] = curr_price
                        except Exception as e:
                            logger.debug(f"WS /ws/prices: send error: {e}")
                            break
                await asyncio.sleep(1.0)
            else:
                # Fallback: poll every 15 seconds
                elapsed = (now - last_fallback_poll).total_seconds()
                if elapsed >= _FALLBACK_POLL_INTERVAL:
                    last_fallback_poll = now
                    for sym in list(subscribed_symbols):
                        entry = _get_price_entry(sym)
                        if not entry:
                            continue
                        try:
                            await websocket.send_json({
                                "type": "tick",
                                "symbol": sym,
                                **entry,
                            })
                            last_sent[sym] = entry.get("price")
                        except Exception as e:
                            logger.debug(f"WS /ws/prices: fallback send error: {e}")
                            break
                await asyncio.sleep(1.0)

    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("WS /ws/prices: client disconnected")
    except Exception as e:
        logger.warning(f"WS /ws/prices: unexpected error: {e}")
    finally:
        receive_task.cancel()
        try:
            await receive_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("WS /ws/prices: connection cleaned up")
