"""
ASTRA Dhan HQ Data Service
==========================
Market DATA only (historical candles + quotes). This module never places
orders — ASTRA is paper-trading only (see broker.py).

Fixes in v1.13
--------------
* dhanhq >= 2.1 requires `dhanhq(DhanContext(client_id, token))`; the old
  `dhanhq(client_id, token)` call raised inside the library and the client
  silently stayed None. Both signatures are supported now.
* Candle arrays live under response["data"] (the old parser read the top
  level, so every Dhan response parsed as empty).
* Failures raise DhanError(kind, msg) so the data layer can show *why*
  (expired token, no Data-API subscription, …) instead of "no data".

Note: Dhan's historical/quote endpoints require an active Data API plan on
your Dhan account, and access tokens expire (24h for session tokens).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from app.core.config import get_secret

logger = logging.getLogger(__name__)

_SCRIP_MASTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dhan_scrip_master.csv")

_INTRADAY_INTERVAL_MAP = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


class DhanError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _load_symbol_cache() -> dict:
    """NSE EQ trading symbol → security_id, from the Dhan scrip master CSV."""
    cache: dict = {}
    try:
        df = pd.read_csv(_SCRIP_MASTER_PATH, dtype=str, low_memory=False,
                         usecols=["SEM_EXM_EXCH_ID", "SEM_INSTRUMENT_NAME", "SEM_SERIES",
                                  "SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"])
        mask = ((df["SEM_EXM_EXCH_ID"].str.strip() == "NSE")
                & (df["SEM_INSTRUMENT_NAME"].str.strip() == "EQUITY")
                & (df["SEM_SERIES"].str.strip() == "EQ"))
        sub = df.loc[mask, ["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]].dropna()
        cache = dict(zip(sub["SEM_TRADING_SYMBOL"].str.strip().str.upper(),
                         sub["SEM_SMST_SECURITY_ID"].str.strip()))
        logger.info(f"Dhan scrip master loaded: {len(cache)} NSE EQ symbols")
    except FileNotFoundError:
        logger.warning(f"Dhan scrip master not found at {_SCRIP_MASTER_PATH}")
    except Exception as e:
        logger.error(f"Failed to load Dhan scrip master: {e}")
    return cache


_SYMBOL_CACHE: dict = {}


def _symbols() -> dict:
    global _SYMBOL_CACHE
    if not _SYMBOL_CACHE:
        _SYMBOL_CACHE = _load_symbol_cache()
    return _SYMBOL_CACHE


def _classify(remarks) -> str:
    text = str(remarks).lower()
    if any(k in text for k in ("token", "auth", "invalid client", "dh-901", "dh-902", "unauthor", "expired")):
        return "auth"
    if any(k in text for k in ("subscri", "not enabled", "data api", "dh-903", "permission")):
        return "auth"
    if any(k in text for k in ("rate", "too many", "dh-904", "limit")):
        return "rate_limit"
    return "no_data"


class DhanDataService:
    def __init__(self):
        self._client = None
        self._client_key = None   # (client_id, token) the client was built with

    # ── credentials / client ───────────────────────────────────────────────
    @staticmethod
    def _creds(token: Optional[str] = None):
        """(client_id, access_token). A per-user `token` overrides the global one."""
        return get_secret("dhan_client_id"), (token or get_secret("dhan_access_token"))

    def is_available(self, token: Optional[str] = None) -> bool:
        cid, tok = self._creds(token)
        return bool(cid and tok)

    def _get_client(self, token: Optional[str] = None):
        cid, tok = self._creds(token)
        if not (cid and tok):
            raise DhanError("unsupported", "Dhan credentials not configured")
        if self._client is not None and self._client_key == (cid, tok):
            return self._client
        if token:   # per-user token: one-shot client, don't replace the shared one
            return self._build(cid, tok)
        self._client, self._client_key = self._build(cid, tok), (cid, tok)
        logger.info("Dhan HQ data client initialised")
        return self._client

    @staticmethod
    def _build(cid, tok):
        try:
            from dhanhq import dhanhq  # type: ignore
        except ImportError:
            raise DhanError("unsupported", "dhanhq library not installed")
        try:
            try:
                from dhanhq import DhanContext  # dhanhq >= 2.1
                client = dhanhq(DhanContext(str(cid), str(tok)))
            except ImportError:
                client = dhanhq(str(cid), str(tok))   # dhanhq 2.0.x
        except Exception as e:
            raise DhanError("auth", f"client init failed: {e}")
        return client

    def resolve(self, symbol: str) -> tuple:
        clean = symbol.strip().upper().replace(".NS", "").replace(".BSE", "").replace(".BO", "")
        sec_id = _symbols().get(clean)
        return (sec_id, "NSE_EQ") if sec_id else (None, "")

    @staticmethod
    def _unwrap(response) -> dict:
        """dhanhq returns {"status", "remarks", "data"}; raise on failure."""
        if not isinstance(response, dict):
            raise DhanError("no_data", f"unexpected response type {type(response).__name__}")
        if response.get("status") == "failure":
            remarks = response.get("remarks") or response.get("data") or "request failed"
            raise DhanError(_classify(remarks), str(remarks)[:200])
        data = response.get("data", response)
        return data if isinstance(data, dict) else {}

    # ── historical candles ─────────────────────────────────────────────────
    def get_ohlcv(self, symbol: str, days: int = None, interval: str = "1d", period: str = None,
                  token: Optional[str] = None) -> pd.DataFrame:
        if not days:
            days = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731}.get(period or "6mo", 183)
        sec_id, seg = self.resolve(symbol)
        if not sec_id:
            raise DhanError("no_data", f"'{symbol}' not in Dhan scrip master")
        client = self._get_client(token)
        to_date = datetime.now()
        if interval == "1d":
            from_date = to_date - timedelta(days=days)
            resp = client.historical_daily_data(sec_id, seg, "EQUITY",
                                                from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
        elif interval in _INTRADAY_INTERVAL_MAP:
            from_date = to_date - timedelta(days=min(days, 89))   # Dhan: max 90 days per call
            resp = client.intraday_minute_data(sec_id, seg, "EQUITY",
                                               from_date.strftime("%Y-%m-%d %H:%M:%S"),
                                               to_date.strftime("%Y-%m-%d %H:%M:%S"),
                                               interval=_INTRADAY_INTERVAL_MAP[interval])
        else:
            raise DhanError("no_data", f"interval {interval} unsupported")
        return self._parse_candles(self._unwrap(resp))

    @staticmethod
    def _parse_candles(data: dict) -> pd.DataFrame:
        closes, ts = data.get("close") or [], data.get("timestamp") or []
        if not closes or not ts:
            raise DhanError("no_data", "empty candle response")
        n = min(len(closes), len(ts), *(len(data.get(k) or []) for k in ("open", "high", "low")))
        df = pd.DataFrame({
            "Open": data["open"][:n], "High": data["high"][:n], "Low": data["low"][:n],
            "Close": closes[:n], "Volume": (data.get("volume") or [0] * n)[:n],
        })
        idx = pd.to_datetime(pd.Series(ts[:n]), unit="s", utc=True, errors="coerce")
        df.index = idx.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).values
        df.index.name = "Date"
        return df.dropna().sort_index()

    # ── quotes ─────────────────────────────────────────────────────────────
    def get_quote(self, symbol: str, token: Optional[str] = None) -> float:
        sec_id, seg = self.resolve(symbol)
        if not sec_id:
            return 0.0
        try:
            client = self._get_client(token)
            data = self._unwrap(client.ticker_data({seg: [int(sec_id)]}))
            # v2 shape: {"data": {"NSE_EQ": {"2885": {"last_price": …}}}, "status": "success"}
            inner = data.get("data", data)
            entry = (inner.get(seg) or {}).get(str(sec_id)) or {}
            ltp = entry.get("last_price") or entry.get("LTP")
            return round(float(ltp), 2) if ltp else 0.0
        except DhanError as e:
            logger.debug(f"Dhan quote failed for {symbol}: {e}")
            return 0.0
        except Exception as e:
            logger.debug(f"Dhan quote error for {symbol}: {e}")
            return 0.0

    def context(self):
        """DhanContext for the WebSocket feed (None if unavailable)."""
        cid, tok = self._creds()
        if not (cid and tok):
            return None
        try:
            from dhanhq import DhanContext  # type: ignore
            return DhanContext(str(cid), str(tok))
        except Exception:
            return None


dhan_data_service = DhanDataService()
