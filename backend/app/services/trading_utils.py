"""
Shared helpers for the paper trading book: currency, FX, price rounding,
IST trading day and NSE market hours.

The paper book's base currency is INR. Positions in USD-priced assets
(crypto -USD pairs, global futures like GC=F, US stocks) are converted with a
live USD/INR rate when they are added to INR totals.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
FALLBACK_USD_INR = 84.0
_FX = {"rate": None, "at": 0.0, "source": None}


def currency_of(symbol: str) -> str:
    """INR for NSE/BSE stocks, Indian indices and INR crypto pairs; USD otherwise."""
    from app.services.market_data import normalize_symbol, asset_class
    s = normalize_symbol(symbol or "") or (symbol or "").upper()
    if s.endswith((".NS", ".BO", "-INR")) or s.startswith(("^NSE", "^CNX", "^BSE", "^INDIA")):
        return "INR"
    if asset_class(s) in ("india_equity", "india_index"):
        return "INR"
    return "USD"


def usd_inr() -> tuple[float, str]:
    """(rate, source). Cached 1h; falls back to a fixed rate if no quote is available."""
    if _FX["rate"] and time.time() - _FX["at"] < 3600:
        return _FX["rate"], _FX["source"]
    rate, src = None, None
    try:
        from app.services.market_data import market_data
        info = market_data.get_quote_info("USDINR=X")
        if info.get("price") and 50 < float(info["price"]) < 200:
            rate, src = float(info["price"]), info.get("source") or "quote"
    except Exception as e:
        logger.debug(f"USDINR quote failed: {e}")
    if rate is None:
        rate, src = FALLBACK_USD_INR, "fallback"
    _FX.update(rate=rate, at=time.time(), source=src)
    return rate, src


def to_inr(amount: float, currency: str) -> float:
    if not amount or currency == "INR":
        return float(amount or 0.0)
    return float(amount) * usd_inr()[0]


def px(p: float) -> float:
    """Round a price by magnitude (2 dp for ≥100, 4 dp for ≥1, else 8 dp) — keeps
    sub-rupee / sub-dollar assets (DOGE, SHIB) from collapsing to 0."""
    if p is None:
        return p
    p = float(p)
    a = abs(p)
    return round(p, 2 if a >= 100 else 4 if a >= 1 else 8)


def ist_day_start_utc(now: datetime | None = None) -> datetime:
    """Naive-UTC datetime of today's 00:00 IST (DB timestamps are naive UTC)."""
    now = now or datetime.now(timezone.utc)
    ist_now = now.astimezone(IST)
    start = ist_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).replace(tzinfo=None)


def nse_open(now: datetime | None = None) -> bool:
    """NSE cash session Mon–Fri 09:15–15:30 IST (exchange holidays not modelled)."""
    t = (now or datetime.now(timezone.utc)).astimezone(IST)
    if t.weekday() >= 5:
        return False
    mins = t.hour * 60 + t.minute
    return 9 * 60 + 15 <= mins <= 15 * 60 + 30


def is_india_equity(symbol: str) -> bool:
    from app.services.market_data import normalize_symbol, asset_class
    s = normalize_symbol(symbol or "")
    return asset_class(s) == "india_equity"
