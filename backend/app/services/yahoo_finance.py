"""
ASTRA Yahoo Finance Service
============================
Direct wrapper around Yahoo Finance v8 API and NSE India public API.
No API key. No account. No daily activation. Completely free.

Why not yfinance (the library)?
  The library's cookie/crumb auth handshake fails in environments where
  Yahoo's consent page DNS resolves differently. This service calls the
  raw v8 chart endpoint directly with a standard browser User-Agent,
  which works reliably without any auth dance.

Capabilities
------------
  get_ohlcv(symbol, period, interval)   → pd.DataFrame (OHLCV, IST-indexed)
  get_quote(symbol)                     → float (last traded price)
  get_nse_quote(symbol)                 → dict  (full NSE market snapshot)

Supported intervals / periods
------------------------------
  interval "1d"  → periods: 1mo 3mo 6mo 1y 2y 5y
  interval "1h"  → periods: 1mo (max ~730 h bars)
  interval "15m" → periods: 5d  (Yahoo limit: 7d of 15m)
  interval "5m"  → periods: 5d
  interval "1m"  → periods: 1d  (Yahoo limit: 1d of 1m)

Symbol format: use NSE suffix (.NS) for Indian stocks → "RELIANCE.NS"
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_YF_HOSTS  = [
    "https://query1.finance.yahoo.com/v8/finance/chart",
    "https://query2.finance.yahoo.com/v8/finance/chart",
]
_NSE_BASE  = "https://www.nseindia.com"
_IST_TZ    = "Asia/Kolkata"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Rotate host index to spread load across query1/query2
_host_idx = 0

# Period → yfinance range string
_PERIOD_MAP = {
    "1d":  "1d",
    "5d":  "5d",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y":  "1y",
    "2y":  "2y",
    "5y":  "5y",
}

# Interval → Yahoo interval string
_INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "60m",
    "1d":  "1d",
    "1wk": "1wk",
}

# ── OHLCV cache (TTL per interval) ────────────────────────────────────────────
_CACHE: dict = {}   # key → (DataFrame, cached_at)
_TTL = {
    "1m":  60,        # 1 min
    "5m":  60,        # 1 min
    "15m": 300,       # 5 min
    "1h":  900,       # 15 min
    "1d":  14_400,    # 4 h
}

_NSE_SESSION: Optional[requests.Session] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_nse_session() -> requests.Session:
    """Return a warm NSE session (loads cookies from nseindia.com home page)."""
    global _NSE_SESSION
    if _NSE_SESSION is None:
        _NSE_SESSION = requests.Session()
        try:
            _NSE_SESSION.get(_NSE_BASE, headers=_HEADERS, timeout=10)
        except Exception as e:
            logger.debug(f"NSE session init warning: {e}")
    return _NSE_SESSION


def _parse_chart_response(data: dict, interval: str) -> pd.DataFrame:
    """Convert Yahoo v8 chart JSON → clean OHLCV DataFrame with IST index."""
    try:
        result = data["chart"]["result"]
        if not result:
            return pd.DataFrame()
        r      = result[0]
        ts     = r.get("timestamp", [])
        q      = r["indicators"]["quote"][0]
        opens  = q.get("open",   [None] * len(ts))
        highs  = q.get("high",   [None] * len(ts))
        lows   = q.get("low",    [None] * len(ts))
        closes = q.get("close",  [None] * len(ts))
        vols   = q.get("volume", [None] * len(ts))

        rows = []
        for t, o, h, l, c, v in zip(ts, opens, highs, lows, closes, vols):
            if c is None:
                continue
            rows.append({
                "Date":   pd.Timestamp(t, unit="s", tz="UTC").tz_convert(_IST_TZ).tz_localize(None),
                "Open":   float(o) if o is not None else float(c),
                "High":   float(h) if h is not None else float(c),
                "Low":    float(l) if l is not None else float(c),
                "Close":  float(c),
                "Volume": int(v)   if v is not None else 0,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df.index.name = "Date"
        return df

    except Exception as e:
        logger.debug(f"Yahoo chart parse error: {e}")
        return pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV bars from Yahoo Finance.

    Parameters
    ----------
    symbol   : NSE/BSE symbol with suffix, e.g. "RELIANCE.NS", "TCS.NS"
    period   : "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"
    interval : "1m", "5m", "15m", "1h", "1d", "1wk"

    Returns pd.DataFrame with DatetimeIndex (IST, tz-naive) and OHLCV columns,
    or empty DataFrame on failure.

    Notes
    -----
    Yahoo limits:
      15m data → max 60 days lookback
      1h  data → max 730 days
      1d  data → unlimited (up to 5y)
    """
    cache_key = f"{symbol}|{period}|{interval}"
    ttl       = _TTL.get(interval, 14_400)

    if cache_key in _CACHE:
        cached_df, cached_at = _CACHE[cache_key]
        if (datetime.now() - cached_at).total_seconds() < ttl and not cached_df.empty:
            return cached_df

    yf_range    = _PERIOD_MAP.get(period, "6mo")
    yf_interval = _INTERVAL_MAP.get(interval, "1d")

    params = f"?range={yf_range}&interval={yf_interval}&includePrePost=false"

    # Try both hosts with simple retry on 429
    global _host_idx
    for attempt in range(4):
        host = _YF_HOSTS[(_host_idx + attempt) % len(_YF_HOSTS)]
        url  = f"{host}/{symbol}{params}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 429:
                logger.debug(f"Yahoo 429 on {host} for {symbol}, backing off {2**attempt}s")
                time.sleep(2 ** attempt)
                _host_idx = (_host_idx + 1) % len(_YF_HOSTS)
                continue
            if resp.status_code != 200:
                logger.debug(f"Yahoo {symbol} HTTP {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            err  = data.get("chart", {}).get("error")
            if err:
                logger.debug(f"Yahoo {symbol} API error: {err}")
                return pd.DataFrame()

            df = _parse_chart_response(data, interval)
            if not df.empty:
                logger.info(f"Yahoo: {symbol} {interval}/{period} → {len(df)} bars")
                _CACHE[cache_key] = (df, datetime.now())
                _host_idx = (_host_idx + 1) % len(_YF_HOSTS)  # round-robin for next call
            return df

        except Exception as e:
            logger.debug(f"Yahoo get_ohlcv attempt {attempt} failed for {symbol}: {e}")
            time.sleep(1)

    return pd.DataFrame()


def get_quote(symbol: str) -> float:
    """
    Get real-time last traded price from Yahoo Finance.
    Returns 0.0 on failure.
    """
    global _host_idx
    for attempt in range(3):
        host = _YF_HOSTS[(_host_idx + attempt) % len(_YF_HOSTS)]
        try:
            resp = requests.get(
                f"{host}/{symbol}?range=1d&interval=1m&includePrePost=false",
                headers=_HEADERS, timeout=8
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            ltp  = meta.get("regularMarketPrice") or meta.get("previousClose", 0.0)
            return round(float(ltp), 2)
        except Exception as e:
            logger.debug(f"Yahoo get_quote attempt {attempt} failed for {symbol}: {e}")
    return 0.0


def get_nse_quote(symbol: str) -> dict:
    """
    Get live NSE market snapshot for a symbol via NSE India public API.
    Returns a dict with keys: lastPrice, change, pChange, open, high, low,
    previousClose, totalTradedVolume, weekHighLow, etc.
    Returns {} on failure.
    """
    clean = symbol.upper().replace(".NS", "").replace(".BSE", "")
    try:
        sess = _get_nse_session()
        url  = f"{_NSE_BASE}/api/quote-equity?symbol={clean}"
        resp = sess.get(url, headers={**_HEADERS, "Referer": _NSE_BASE}, timeout=10)
        data = resp.json()
        pi   = data.get("priceInfo", {})
        ti   = data.get("tradeInfo", {})
        return {
            "symbol":             clean,
            "lastPrice":          pi.get("lastPrice", 0.0),
            "change":             pi.get("change", 0.0),
            "pChange":            pi.get("pChange", 0.0),
            "open":               pi.get("open", 0.0),
            "high":               pi.get("intraDayHighLow", {}).get("max", 0.0),
            "low":                pi.get("intraDayHighLow", {}).get("min", 0.0),
            "previousClose":      pi.get("previousClose", 0.0),
            "vwap":               pi.get("vwap", 0.0),
            "totalTradedVolume":  ti.get("totalTradedVolume", 0),
            "totalTradedValue":   ti.get("totalTradedValue", 0.0),
            "week52High":         pi.get("weekHighLow", {}).get("max", 0.0),
            "week52Low":          pi.get("weekHighLow", {}).get("min", 0.0),
            "source":             "NSE_direct",
        }
    except Exception as e:
        logger.debug(f"NSE get_nse_quote({clean}) failed: {e}")
        # Reset session on failure so next call re-initialises
        global _NSE_SESSION
        _NSE_SESSION = None
        return {}


def get_nse_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    """
    Fetch daily OHLCV from NSE India public equity history API.
    No API key required. Requires session warmup (handled automatically).

    Parameters
    ----------
    symbol : NSE symbol, e.g. "RELIANCE" or "RELIANCE.NS"
    days   : lookback in calendar days (default 90 ≈ 3 months of trading days)

    Returns pd.DataFrame with DatetimeIndex (IST, tz-naive) and OHLCV columns,
    or empty DataFrame on failure.
    """
    global _NSE_SESSION
    from datetime import timedelta

    clean = symbol.upper().replace(".NS", "").replace(".BSE", "")
    try:
        sess  = _get_nse_session()
        end   = datetime.now()
        start = end - timedelta(days=days + 10)   # +10 for weekends/holidays buffer
        url   = (
            f"{_NSE_BASE}/api/historical/cm/equity"
            f"?symbol={clean}&series=[%22EQ%22]"
            f"&from={start.strftime('%d-%m-%Y')}&to={end.strftime('%d-%m-%Y')}&csv=false"
        )
        resp = sess.get(url, headers={**_HEADERS, "Referer": _NSE_BASE}, timeout=12)
        if resp.status_code != 200:
            logger.debug(f"NSE historical HTTP {resp.status_code} for {clean} — CloudFlare/bot block")
            _NSE_SESSION = None   # force re-init next call
            return pd.DataFrame()
        data = resp.json()
        rows_raw = data.get("data", [])
        if not rows_raw:
            return pd.DataFrame()

        rows = []
        for r in rows_raw:
            try:
                # NSE date format: "14-Apr-2025"
                dt = pd.to_datetime(r.get("CH_TIMESTAMP") or r.get("mTIMESTAMP"), dayfirst=True)
                rows.append({
                    "Date":   dt,
                    "Open":   float(r.get("CH_OPENING_PRICE", 0) or 0),
                    "High":   float(r.get("CH_TRADE_HIGH_PRICE", 0) or 0),
                    "Low":    float(r.get("CH_TRADE_LOW_PRICE", 0) or 0),
                    "Close":  float(r.get("CH_CLOSING_PRICE", 0) or 0),
                    "Volume": int(r.get("CH_TOT_TRADED_QTY", 0) or 0),
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df.index.name = "Date"
        # Drop rows with zero Close (bad data)
        df = df[df["Close"] > 0]
        logger.info(f"NSE OHLCV: {clean} → {len(df)} bars")
        return df

    except Exception as e:
        logger.debug(f"NSE get_nse_ohlcv({clean}) failed: {e}")
        _NSE_SESSION = None   # Reset session so next call re-initialises cookies
        return pd.DataFrame()


def clear_cache(symbol: Optional[str] = None) -> None:
    """Clear OHLCV cache — all symbols or a specific one."""
    global _CACHE
    if symbol is None:
        _CACHE = {}
    else:
        _CACHE = {k: v for k, v in _CACHE.items() if not k.startswith(symbol)}


# ── Module-level singleton (matches pattern used by dhan_data_service) ────────
class YahooFinanceService:
    """Thin wrapper so callers can import `yahoo_service` as a singleton."""

    @staticmethod
    def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        sym = symbol.strip()
        if not sym.endswith((".NS", ".BSE", "-USD", "-INR")):
            sym = sym + ".NS"
        return get_ohlcv(sym, period, interval)

    @staticmethod
    def get_quote(symbol: str) -> float:
        sym = symbol.strip()
        if not sym.endswith((".NS", ".BSE", "-USD", "-INR")):
            sym = sym + ".NS"
        return get_quote(sym)

    @staticmethod
    def get_nse_quote(symbol: str) -> dict:
        return get_nse_quote(symbol)

    @staticmethod
    def get_nse_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
        return get_nse_ohlcv(symbol, days)

    @staticmethod
    def is_available() -> bool:
        return True   # No credentials needed


yahoo_service = YahooFinanceService()
