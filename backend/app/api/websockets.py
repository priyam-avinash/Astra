from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis # type: ignore

router = APIRouter()
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Fallback poll interval (seconds) when DhanFeed is not live
_FALLBACK_POLL_INTERVAL = 15

# Per-connection cap on explicit {"action": "subscribe"} symbols — each one is
# polled through market_data, so an unbounded set is a cheap DoS.
MAX_SUBSCRIPTIONS_PER_CONNECTION = 50

_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


# ── Handshake auth ───────────────────────────────────────────────────────────
# Browsers can't set an Authorization header on a WebSocket, so the access JWT
# comes as ?token=. Cross-site pages could otherwise open these sockets with
# the victim's network position, so Origin is checked too (when sent).

def _allowed_origins() -> set:
    raw = os.getenv("FRONTEND_ORIGINS", _DEFAULT_ORIGINS)
    return {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}


def _origin_ok(websocket: WebSocket) -> bool:
    origin = (websocket.headers.get("origin") or "").strip().rstrip("/")
    if not origin:
        return True          # non-browser client (CLI, tests) — token still required in jwt mode
    if origin in _allowed_origins():
        return True
    host = websocket.headers.get("host", "")
    return bool(host) and origin in (f"http://{host}", f"https://{host}")   # same origin


def _user_exists(username: str) -> bool:
    from app.models.database import SessionLocal, User
    db = SessionLocal()
    try:
        return db.query(User.id).filter(User.username == username).first() is not None
    finally:
        db.close()


async def _authenticate_ws(websocket: WebSocket) -> Optional[str]:
    """Return a principal name if the handshake may proceed, else None.
    Must be called BEFORE accept(); the caller closes with 1008 on None."""
    if not _origin_ok(websocket):
        logger.warning("WS %s: rejected origin %r", websocket.url.path, websocket.headers.get("origin"))
        return None
    if (os.getenv("AUTH_MODE", "jwt") or "jwt").strip().lower() == "bypass":
        return "admin_bypass"
    token = websocket.query_params.get("token") or ""
    if not token:
        return None
    try:
        from app.services.auth import decode_access_token
        username = decode_access_token(token)["sub"]
    except Exception:
        return None
    try:
        if not await asyncio.to_thread(_user_exists, username):
            return None
    except Exception as e:
        logger.warning("WS auth: user lookup failed: %s", e)
        return None
    return username


async def _reject(websocket: WebSocket):
    # close() before accept() → the handshake is refused (HTTP 403)
    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)


@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    if await _authenticate_ws(websocket) is None:
        await _reject(websocket)
        return
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

    Auth: ?token=<access JWT> (not needed with AUTH_MODE=bypass); the Origin,
    when present, must be in FRONTEND_ORIGINS or same-origin.
    """
    if await _authenticate_ws(websocket) is None:
        await _reject(websocket)
        return
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

    # Symbols streamed by the Dhan feed are pushed only from the in-memory tick
    # cache (no network). Symbols the client subscribes to explicitly are also
    # polled via market_data — always in a worker thread, never on the event
    # loop (v1.12 called blocking HTTP here, which froze the whole API server).
    feed_symbols: set = set(dhan_feed_manager._subscriptions) if dhan_feed_manager is not None else set()

    def _live_entry(symbol: str) -> dict:
        if dhan_feed_manager is not None:
            live = dhan_feed_manager.get_price(symbol)
            if live and live > 0:
                return {"price": live, "source": "dhan_ws", "ts": datetime.now().isoformat()}
        return {}

    def _polled_entry(symbol: str) -> dict:
        entry = _live_entry(symbol)
        if entry:
            return entry
        try:
            from app.services.market_data import market_data
            info = market_data.get_quote_info(symbol)
            if info["price"] > 0:
                return {"price": info["price"], "source": info["source"] or "cache",
                        "stale": info["stale"], "ts": datetime.now().isoformat()}
        except Exception:
            pass
        return {}

    async def _get_price_entry(symbol: str) -> dict:
        if symbol in subscribed_symbols:
            return await asyncio.to_thread(_polled_entry, symbol)
        return _live_entry(symbol)

    # Initial snapshot: whatever the live feed already has (instant, no network)
    try:
        snapshot_prices = {s: e for s in feed_symbols if (e := _live_entry(s))}
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
                        sym = str(msg.get("symbol", "")).strip().upper()[:32]
                        if sym and sym not in subscribed_symbols:
                            if len(subscribed_symbols) >= MAX_SUBSCRIPTIONS_PER_CONNECTION:
                                await websocket.send_json({
                                    "type": "error", "symbol": sym,
                                    "error": f"subscription limit ({MAX_SUBSCRIPTIONS_PER_CONNECTION}) reached",
                                })
                                continue
                            subscribed_symbols.add(sym)
                            # Also subscribe in feed manager for future connections
                            if dhan_feed_manager is not None:
                                dhan_feed_manager.subscribe(sym)
                            logger.debug(f"WS /ws/prices: client subscribed to {sym}")
                            # Send immediate price if available
                            entry = await _get_price_entry(sym)
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
                for sym in list(subscribed_symbols | feed_symbols):
                    entry = await _get_price_entry(sym)
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
                        entry = await _get_price_entry(sym)
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
