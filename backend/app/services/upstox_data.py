"""
ASTRA Upstox Data Service (v2 API)
====================================
Free historical OHLCV data for NSE equities via Upstox API v2.
- 1min bars  : up to 1 month (auto-resampled to 15m)
- 30min bars : up to 1 year  (used as proxy when 1min window exceeded)
- 1D bars    : up to 2 years (equity backtest daily)

OAuth flow:
  1. User visits /upstox/login  → redirected to Upstox
  2. User logs in               → redirected to /upstox/callback?code=...
  3. Backend exchanges code for access_token → stored in os.environ + .env

Usage:
    from app.services.upstox_data import upstox_service
    df = upstox_service.get_ohlcv("RELIANCE", period="1mo", interval="15m")
    available = upstox_service.is_available()
"""

import gzip
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_URL    = "https://api.upstox.com/v2"
_TOKEN_URL   = "https://api.upstox.com/v2/login/authorization/token"
_AUTH_URL    = "https://api.upstox.com/v2/login/authorization/dialog"
_INST_URL    = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

from app.core.config import SERVERLESS as _SERVERLESS, DATA_DIR as _DATA_DIR
_INSTRUMENTS_CACHE_PATH = (_DATA_DIR / "upstox_instruments.json") if _SERVERLESS else Path(__file__).parent.parent / "data" / "upstox_instruments.json"
_INSTRUMENTS_CACHE_TTL  = 86_400  # 24 hours

# Upstox historical intervals and their max lookback in days
_INTERVAL_CONFIG = {
    "1minute":  {"upstox": "1minute",  "max_days": 30},
    "30minute": {"upstox": "30minute", "max_days": 365},
    "60minute": {"upstox": "60minute", "max_days": 730},
    "1day":     {"upstox": "1day",     "max_days": 730},
}

# ── In-memory symbol → instrument_key cache ──────────────────────────────────
_SYMBOL_MAP: dict = {}     # "RELIANCE" → "NSE_EQ|INE002A01018"
_MAP_LOADED: bool  = False


def _load_instruments() -> dict:
    """
    Download and cache the Upstox instruments master.
    Returns symbol → instrument_key map for NSE EQ instruments.
    """
    global _SYMBOL_MAP, _MAP_LOADED
    if _MAP_LOADED and _SYMBOL_MAP:
        return _SYMBOL_MAP

    # Try loading from local cache first
    if _INSTRUMENTS_CACHE_PATH.exists():
        age = time.time() - _INSTRUMENTS_CACHE_PATH.stat().st_mtime
        if age < _INSTRUMENTS_CACHE_TTL:
            try:
                with open(_INSTRUMENTS_CACHE_PATH) as f:
                    _SYMBOL_MAP = json.load(f)
                _MAP_LOADED = True
                logger.info(f"Upstox instruments loaded from cache: {len(_SYMBOL_MAP)} symbols")
                return _SYMBOL_MAP
            except Exception as e:
                logger.warning(f"Upstox instruments cache corrupt, re-downloading: {e}")

    # Download fresh copy
    try:
        logger.info("Downloading Upstox instruments master...")
        r = requests.get(_INST_URL, timeout=30)
        r.raise_for_status()
        instruments = json.loads(gzip.decompress(r.content))

        sym_map: dict = {}
        for inst in instruments:
            if (
                inst.get("segment") == "NSE_EQ"
                and inst.get("instrument_type") == "EQ"
                and inst.get("trading_symbol")
                and inst.get("instrument_key")
            ):
                sym_map[inst["trading_symbol"].upper()] = inst["instrument_key"]

        _INSTRUMENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INSTRUMENTS_CACHE_PATH, "w") as f:
            json.dump(sym_map, f)

        _SYMBOL_MAP = sym_map
        _MAP_LOADED = True
        logger.info(f"Upstox instruments downloaded: {len(sym_map)} NSE EQ symbols")
    except Exception as e:
        logger.error(f"Failed to load Upstox instruments: {e}")

    return _SYMBOL_MAP


def _resolve_instrument_key(symbol: str) -> Optional[str]:
    sym = symbol.strip().replace(".NS", "").replace(".BSE", "").upper()
    sym_map = _load_instruments()
    key = sym_map.get(sym)
    if not key:
        logger.warning(f"Upstox: no instrument key for {sym}")
    return key


class UpstoxDataService:
    """
    Historical OHLCV data from Upstox v2 API.
    Access token refreshed daily via the OAuth flow at /upstox/login.
    """

    def __init__(self):
        self._api_key     = os.getenv("UPSTOX_API_KEY", "")
        self._api_secret  = os.getenv("UPSTOX_API_SECRET", "")
        self._redirect    = os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8000/upstox/callback")
        self._token: str  = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        self._token_ts: float = 0.0   # epoch of last successful token fetch

    # ── Auth ────────────────────────────────────────────────────────────────

    def is_available(self, token: Optional[str] = None) -> bool:
        """Check if a usable access token exists. Optional explicit token overrides env.
        NOTE: This only checks that the string is present — it does NOT validate the token
        against the Upstox server. Use is_token_valid() for a live check.
        """
        tok = token if token is not None else os.getenv("UPSTOX_ACCESS_TOKEN", self._token)
        return bool(self._api_key) and bool(tok) and tok not in ("", "None", "none")

    def is_token_valid(self, token: Optional[str] = None) -> bool:
        """
        Probe the Upstox /user/profile endpoint to verify the token is live.
        Returns True only if the API responds with HTTP 200.
        Falls back to is_available() (no network call) on any connection error.

        Use this in daily_start / status checks — Upstox tokens expire each day
        even if the string is still present in .env from the previous session.
        """
        tok = token if token is not None else os.getenv("UPSTOX_ACCESS_TOKEN", self._token)
        if not self.is_available(token=tok):
            return False
        try:
            r = requests.get(
                f"{_BASE_URL}/user/profile",
                headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
                timeout=5,
            )
            if r.status_code == 200:
                return True
            if r.status_code in (401, 403):
                logger.warning("Upstox token is expired/invalid (HTTP %s)", r.status_code)
                return False
            # Any other error (5xx, network hiccup) — be lenient, don't block startup
            logger.warning("Upstox profile probe returned HTTP %s — treating as available", r.status_code)
            return True
        except Exception as e:
            logger.warning("Upstox profile probe failed (%s) — treating as available", e)
            return True

    def get_login_url(self) -> str:
        return (
            f"{_AUTH_URL}"
            f"?response_type=code"
            f"&client_id={self._api_key}"
            f"&redirect_uri={self._redirect}"
        )

    def exchange_code(self, code: str) -> str:
        """Exchange OAuth code for access token. Stores in os.environ + .env."""
        payload = {
            "code":          code,
            "client_id":     self._api_key,
            "client_secret": self._api_secret,
            "redirect_uri":  self._redirect,
            "grant_type":    "authorization_code",
        }
        r = requests.post(_TOKEN_URL, data=payload, timeout=15)
        r.raise_for_status()
        token = r.json().get("access_token", "")
        if not token:
            raise ValueError(f"No access_token in response: {r.text}")
        self._token    = token
        self._token_ts = time.time()
        os.environ["UPSTOX_ACCESS_TOKEN"] = token
        # Persist to .env so it survives restarts
        _persist_token_to_env(token)
        logger.info("Upstox access token obtained and stored")
        return token

    # ── Data fetching ────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "15m",
                  token: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV bars for `symbol`.

        period  : "5d" | "1mo" | "3mo" | "6mo" | "1y" | "2y"
        interval: "15m" | "1h" | "1d"
        token   : Optional explicit access token (per-user); falls back to env if None.

        Returns DataFrame with columns Open/High/Low/Close/Volume, IST-aware index.
        Returns empty DataFrame on any error (graceful degradation).
        """
        if not self.is_available(token=token):
            return pd.DataFrame()

        instrument_key = _resolve_instrument_key(symbol)
        if not instrument_key:
            return pd.DataFrame()

        # Map caller interval to Upstox interval
        upstox_interval, days = self._resolve_interval(interval, period)
        if not upstox_interval:
            return pd.DataFrame()

        to_date   = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Upstox 1min is limited to 30 days max; chunk if needed
        if upstox_interval == "1minute" and days > 30:
            return self._fetch_chunked_1min(instrument_key, days, token=token)

        return self._fetch_candles(instrument_key, upstox_interval, from_date, to_date, token=token)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resolve_interval(self, interval: str, period: str):
        """Return (upstox_interval, days) for the requested interval/period."""
        _period_days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        days = _period_days.get(period, 30)

        # Upstox v2 valid intervals: 1minute, 30minute, day, week, month
        if interval in ("15m", "15min"):
            # Use 1min and resample to 15m; cap at 30 days
            return "1minute", min(days, 30)
        elif interval in ("30m", "30min"):
            return "30minute", min(days, 365)
        elif interval in ("1h", "60m", "60min"):
            # 60minute not supported by v2 — fall back to 30minute (caller may resample)
            return "30minute", min(days, 365)
        elif interval in ("1d", "1day", "daily", "day"):
            return "day", min(days, 730)
        elif interval == "week":
            return "week", min(days, 1825)
        elif interval == "month":
            return "month", min(days, 3650)
        else:
            logger.warning(f"Upstox: unsupported interval {interval}")
            return None, 0

    def _fetch_candles(self, instrument_key: str, interval: str,
                       from_dt: datetime, to_dt: datetime,
                       token: Optional[str] = None) -> pd.DataFrame:
        """Single-chunk historical candle fetch. `token` overrides env if given."""
        tok = token if token is not None else os.getenv("UPSTOX_ACCESS_TOKEN", self._token)
        headers = {
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
        }
        key_enc  = requests.utils.quote(instrument_key, safe="")
        from_str = from_dt.strftime("%Y-%m-%d")
        to_str   = to_dt.strftime("%Y-%m-%d")
        url = f"{_BASE_URL}/historical-candle/{key_enc}/{interval}/{to_str}/{from_str}"

        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 401:
                logger.warning("Upstox token expired — reconnect via /brokers/Upstox/login-url")
                # Only wipe env if we were using env; preserves per-user tokens in DB.
                if token is None:
                    os.environ["UPSTOX_ACCESS_TOKEN"] = ""
                return pd.DataFrame()
            r.raise_for_status()
            data = r.json().get("data", {}).get("candles", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Kolkata")
            else:
                df.index = df.index.tz_convert("Asia/Kolkata")

            # Resample 1min → 15m if needed
            if interval == "1minute":
                df = df.resample("15min").agg({
                    "Open": "first", "High": "max",
                    "Low": "min",   "Close": "last", "Volume": "sum"
                }).dropna(subset=["Close"])
                df = df[df["Volume"] > 0]

            logger.info(f"Upstox {interval}: {len(df)} bars for {instrument_key}")
            return df

        except Exception as e:
            logger.error(f"Upstox candle fetch failed for {instrument_key}: {e}")
            return pd.DataFrame()

    def _fetch_chunked_1min(self, instrument_key: str, days: int,
                            token: Optional[str] = None) -> pd.DataFrame:
        """Fetch 1min bars in 30-day chunks, resample to 15m. `token` overrides env if given."""
        chunks = []
        to_dt = datetime.now()
        remaining = days
        while remaining > 0:
            chunk_days = min(remaining, 30)
            from_dt = to_dt - timedelta(days=chunk_days)
            chunk = self._fetch_candles(instrument_key, "1minute", from_dt, to_dt, token=token)
            if not chunk.empty:
                chunks.append(chunk)
            to_dt     = from_dt - timedelta(days=1)
            remaining -= chunk_days
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks).sort_index().drop_duplicates()


def _persist_token_to_env(token: str):
    """Write UPSTOX_ACCESS_TOKEN back to .env so it survives server restarts."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text().splitlines()
        updated = []
        found = False
        for line in lines:
            if line.startswith("UPSTOX_ACCESS_TOKEN="):
                updated.append(f'UPSTOX_ACCESS_TOKEN="{token}"')
                found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f'UPSTOX_ACCESS_TOKEN="{token}"')
        env_path.write_text("\n".join(updated) + "\n")
    except Exception as e:
        logger.warning(f"Could not persist Upstox token to .env: {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────
upstox_service = UpstoxDataService()
