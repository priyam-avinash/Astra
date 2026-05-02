"""
ASTRA Dhan HQ Data Service (Phase 1)
=====================================
Wraps the dhanhq library to provide OHLCV historical data and real-time quotes
from the Dhan API. Used as the primary (priority #0.5) data source in the
ASTRA prediction engine — unlimited, native NSE/BSE, no external rate limits.

Graceful degradation: if DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN are missing or
contain placeholder text, all public methods return empty DataFrame / 0.0
without raising exceptions.
"""
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# ── Scrip master path ────────────────────────────────────────────────────────
_SCRIP_MASTER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "dhan_scrip_master.csv"
)

# ── Symbol → security_id cache (populated at import time) ───────────────────
# Key: trading symbol (e.g. "RELIANCE"), Value: security_id string (e.g. "2885")
_SYMBOL_CACHE: dict = {}

# ── Interval mapping (yfinance → Dhan) ──────────────────────────────────────
_INTRADAY_INTERVAL_MAP = {
    "1h": 60,
    "15m": 15,
    "5m": 5,
    "1m": 1,
}

# ── Period → days mapping ────────────────────────────────────────────────────
_PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
}

_PLACEHOLDER_TOKENS = {
    "ENTER_YOUR_DHAN_CLIENT_ID_HERE",
    "ENTER_YOUR_DHAN_ACCESS_TOKEN_HERE",
    "",
    "None",
    "none",
}


def _load_symbol_cache() -> dict:
    """
    Parse the Dhan scrip master CSV and build a symbol → security_id map
    for NSE EQ instruments (series == 'EQ', instrument == 'EQUITY', exchange == 'NSE').
    Falls back gracefully if the file is missing.
    """
    cache: dict = {}
    try:
        df = pd.read_csv(_SCRIP_MASTER_PATH, dtype=str, low_memory=False)
        # Filter: NSE exchange, EQUITY instrument type, EQ series
        mask = (
            (df["SEM_EXM_EXCH_ID"].str.strip() == "NSE")
            & (df["SEM_INSTRUMENT_NAME"].str.strip() == "EQUITY")
            & (df["SEM_SERIES"].str.strip() == "EQ")
        )
        nse_eq = df[mask][["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]].dropna()
        for _, row in nse_eq.iterrows():
            sym = str(row["SEM_TRADING_SYMBOL"]).strip()
            sec_id = str(row["SEM_SMST_SECURITY_ID"]).strip()
            if sym and sec_id:
                cache[sym] = sec_id
        logger.info(f"Dhan scrip master loaded: {len(cache)} NSE EQ symbols cached")
    except FileNotFoundError:
        logger.warning(f"Dhan scrip master not found at {_SCRIP_MASTER_PATH}; symbol resolution disabled")
    except Exception as e:
        logger.error(f"Failed to load Dhan scrip master: {e}")
    return cache


# Populate cache at module import
_SYMBOL_CACHE = _load_symbol_cache()


class DhanDataService:
    """
    Service for fetching OHLCV historical data and real-time quotes from Dhan HQ.

    Usage:
        from app.services.dhan_data import dhan_data_service
        df = dhan_data_service.get_ohlcv("RELIANCE", period="6mo", interval="1d")
        price = dhan_data_service.get_quote("RELIANCE")
    """

    def __init__(self):
        self._client_id: str = os.getenv("DHAN_CLIENT_ID", "")
        self._access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "")
        self._client = None  # lazy-initialized on first use

    def _get_client(self):
        """Lazy-initialize and cache the dhanhq client."""
        if self._client is not None:
            return self._client
        # Re-read env vars in case they were set after import
        self._client_id = os.getenv("DHAN_CLIENT_ID", "")
        self._access_token = os.getenv("DHAN_ACCESS_TOKEN", "")
        if not self.is_available():
            return None
        try:
            from dhanhq import dhanhq  # type: ignore
            self._client = dhanhq(self._client_id, self._access_token)
            logger.info("Dhan HQ client initialized successfully")
        except ImportError:
            logger.warning("dhanhq library not installed; Dhan data source disabled")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Dhan HQ client: {e}")
            self._client = None
        return self._client

    # ─────────────────────── Public API ─────────────────────────────────────

    def is_available(self) -> bool:
        """Return True only if credentials are set and not placeholder values."""
        cid = os.getenv("DHAN_CLIENT_ID", "")
        tok = os.getenv("DHAN_ACCESS_TOKEN", "")
        return (
            bool(cid)
            and bool(tok)
            and cid.strip() not in _PLACEHOLDER_TOKENS
            and tok.strip() not in _PLACEHOLDER_TOKENS
        )

    def resolve(self, symbol: str) -> tuple:
        """
        Resolve a trading symbol to (security_id, exchange_segment).
        Returns (security_id_str, "NSE_EQ") if found, else (None, "").
        Strips common suffixes like '.NS' before lookup.
        """
        clean = symbol.strip().replace(".NS", "").replace(".BSE", "").upper()
        sec_id = _SYMBOL_CACHE.get(clean)
        if sec_id:
            return sec_id, "NSE_EQ"
        return None, ""

    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV data for the given symbol.

        Parameters
        ----------
        symbol : str
            NSE trading symbol, e.g. "RELIANCE"
        period : str
            "1mo", "3mo", "6mo", "1y", "2y"
        interval : str
            "1d" → daily bars
            "1h" → 60-min intraday
            "15m" → 15-min intraday
            "5m"  → 5-min intraday
            "1m"  → 1-min intraday

        Returns
        -------
        pd.DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume],
        or empty DataFrame on failure.
        """
        if not self.is_available():
            return pd.DataFrame()

        sec_id, exchange_segment = self.resolve(symbol)
        if sec_id is None:
            logger.debug(f"Dhan: symbol '{symbol}' not found in scrip master")
            return pd.DataFrame()

        client = self._get_client()
        if client is None:
            return pd.DataFrame()

        # Compute date range
        days = _PERIOD_DAYS.get(period, 180)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        try:
            if interval == "1d":
                response = client.historical_daily_data(
                    security_id=sec_id,
                    exchange_segment=exchange_segment,
                    instrument_type="EQUITY",
                    from_date=from_str,
                    to_date=to_str,
                    expiry_code=0,
                )
            elif interval in _INTRADAY_INTERVAL_MAP:
                dhan_interval = _INTRADAY_INTERVAL_MAP[interval]
                response = client.intraday_minute_data(
                    security_id=sec_id,
                    exchange_segment=exchange_segment,
                    instrument_type="EQUITY",
                    from_date=from_str,
                    to_date=to_str,
                    interval=dhan_interval,
                )
            else:
                logger.debug(f"Dhan: unsupported interval '{interval}'; falling back")
                return pd.DataFrame()

            return self._parse_response(response, interval)

        except Exception as e:
            logger.debug(f"Dhan get_ohlcv failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> float:
        """
        Fetch real-time last traded price for a symbol using quote_data.
        Returns 0.0 if unavailable.
        """
        if not self.is_available():
            return 0.0

        sec_id, exchange_segment = self.resolve(symbol)
        if sec_id is None:
            return 0.0

        client = self._get_client()
        if client is None:
            return 0.0

        try:
            # quote_data expects {exchange_segment: [security_id_int]}
            sec_id_int = int(sec_id)
            response = client.quote_data({exchange_segment: [sec_id_int]})
            # Response structure: {"data": {"NSE_EQ": {str(sec_id): {..., "last_price": ...}}}}
            data = response.get("data", {})
            seg_data = data.get(exchange_segment, {})
            entry = seg_data.get(str(sec_id_int), seg_data.get(sec_id, {}))
            ltp = entry.get("last_price") or entry.get("ltp") or entry.get("LTP")
            if ltp is not None and float(ltp) > 0:
                return round(float(ltp), 2)
        except Exception as e:
            logger.debug(f"Dhan get_quote failed for {symbol}: {e}")

        return 0.0

    # ─────────────────────── Internal helpers ───────────────────────────────

    def _parse_response(self, response: dict, interval: str) -> pd.DataFrame:
        """
        Convert the dhanhq API response dict into a DataFrame with the standard
        OHLCV column names and a DatetimeIndex.
        """
        try:
            if not response or not isinstance(response, dict):
                return pd.DataFrame()

            opens = response.get("open", [])
            highs = response.get("high", [])
            lows = response.get("low", [])
            closes = response.get("close", [])
            volumes = response.get("volume", [])
            timestamps = response.get("timestamp", [])

            if not closes or not timestamps:
                return pd.DataFrame()

            # Align lengths (defensive)
            min_len = min(len(opens), len(highs), len(lows), len(closes), len(volumes), len(timestamps))
            if min_len == 0:
                return pd.DataFrame()

            df = pd.DataFrame({
                "Open": opens[:min_len],
                "High": highs[:min_len],
                "Low": lows[:min_len],
                "Close": closes[:min_len],
                "Volume": volumes[:min_len],
            })

            # Parse timestamps — Dhan returns Unix epoch integers or ISO strings
            ts_raw = timestamps[:min_len]
            try:
                ts_series = pd.to_datetime(ts_raw, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
            except Exception:
                try:
                    ts_series = pd.to_datetime(ts_raw)
                except Exception:
                    ts_series = pd.RangeIndex(min_len)

            df.index = ts_series
            df.index.name = "Date"
            df = df.astype(float)
            df["Volume"] = df["Volume"].astype(int)
            df = df.sort_index()

            return df

        except Exception as e:
            logger.debug(f"Dhan response parse error: {e}")
            return pd.DataFrame()


# ── Module-level singleton ───────────────────────────────────────────────────
dhan_data_service = DhanDataService()
