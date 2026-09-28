"""
Broker Registry — single source of truth for broker plugins.

Brokers are tried in priority order (low number = high priority) until one
returns data. This replaces ad-hoc fallback chains scattered across the codebase
with a single, testable pipeline.

Priority order chosen by reliability + rate-limit characteristics:
  10. Upstox    (1000 req/day, no rate-limit during quota)
  20. Dhan      (unlimited but requires daily session)
  30. Zerodha   (200/min, requires daily login)
  90. Yahoo     (universal fallback — graceful 429 handling)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from app.brokers.base import BrokerClient
from app.brokers.dhan_client import DhanBroker
from app.brokers.upstox_client import UpstoxBroker
from app.brokers.yahoo_client import YahooBroker
from app.brokers.zerodha_client import ZerodhaBroker

logger = logging.getLogger(__name__)


# (priority, broker_class) — lower priority = tried first
_BROKER_DEFS: list[tuple[int, type[BrokerClient]]] = [
    (10, UpstoxBroker),
    (20, DhanBroker),
    (30, ZerodhaBroker),
    (90, YahooBroker),
]


_BROKER_CACHE: dict[str, BrokerClient] = {}


def get_broker(name: str) -> Optional[BrokerClient]:
    """Get a broker by name (case-insensitive). Returns cached singleton."""
    key = name.strip().lower()
    if key in _BROKER_CACHE:
        return _BROKER_CACHE[key]
    for _, cls in _BROKER_DEFS:
        if cls().name.lower() == key:
            _BROKER_CACHE[key] = cls()
            return _BROKER_CACHE[key]
    return None


def list_brokers() -> list[dict]:
    """Return metadata for every registered broker (used by API + frontend)."""
    out = []
    for prio, cls in sorted(_BROKER_DEFS):
        b = cls()
        out.append({
            "name":      b.name,
            "priority":  prio,
            "available": b.is_available(),
        })
    return out


def fetch_ohlcv(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Try each available broker in priority order. Return the first non-empty result.
    Empty DataFrame if all fail.
    """
    for prio, cls in sorted(_BROKER_DEFS):
        b = cls()
        if not b.is_available():
            continue
        try:
            df = b.get_ohlcv(symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                logger.debug(f"[broker-registry] {symbol} {period}/{interval}: {b.name} returned {len(df)} bars")
                return df
        except Exception as e:
            logger.debug(f"[broker-registry] {b.name} failed for {symbol}: {e}")
    return pd.DataFrame()


def fetch_quote(symbol: str) -> Optional[float]:
    """Try each broker for a real-time quote in priority order."""
    for prio, cls in sorted(_BROKER_DEFS):
        b = cls()
        if not b.is_available():
            continue
        try:
            q = b.get_quote(symbol)
            if q and q > 0:
                return q
        except Exception:
            continue
    return None
