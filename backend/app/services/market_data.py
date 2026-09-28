"""
ASTRA Market Data Layer (v1.13)
===============================
One entry point for every price / OHLCV / news lookup in the app.

    from app.services.market_data import market_data
    df    = market_data.get_ohlcv("RELIANCE", period="2y", interval="1d")
    price = market_data.get_quote("RELIANCE")
    info  = market_data.health()          # per-provider status for the UI

Design
------
* Symbols are normalised once ("RELIANCE" → "RELIANCE.NS", "NIFTY" → "^NSEI").
* Each asset class has an ordered provider chain. Free, keyless, unlimited
  providers come first; keyed/quota providers (Twelve Data, Alpha Vantage)
  are only tried when a real key is configured.
* Every provider call is time-boxed. Auth/quota failures put the provider in
  cooldown (with the reason) so one dead key never slows the whole app down.
* Successful fetches are cached in memory (TTL by interval) and on disk. If
  every provider fails, the last good disk copy is served and flagged stale —
  the UI keeps working on weekends / flaky networks.
* Yahoo is called through curl_cffi with a Chrome TLS fingerprint. Plain
  python-requests is now routinely answered with HTTP 429 by Yahoo.

Every DataFrame returned has columns Open/High/Low/Close/Volume, a tz-naive
DatetimeIndex (exchange local time), and df.attrs = {"source", "stale", "symbol"}.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import pandas as pd

from app.core.config import BACKEND_DIR, env_flag, get_secret

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
HTTP_TIMEOUT = float(os.getenv("ASTRA_HTTP_TIMEOUT", "8"))
DISK_CACHE_DIR = BACKEND_DIR / "data" / "cache"
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  Errors & provider health
# ════════════════════════════════════════════════════════════════════════════

class ProviderError(Exception):
    """kind: 'auth' | 'quota' | 'rate_limit' | 'unsupported' | 'no_data' | 'network'"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


# How long a provider sits out after each kind of failure (seconds)
_COOLDOWN = {
    "auth": 6 * 3600,        # bad / expired key — no point retrying soon
    "quota": 3 * 3600,       # daily credits gone
    "rate_limit": 60,        # per-minute limit
    "network": 30,           # DNS / connection refused / timeout
    "unsupported": 0,        # symbol-specific, provider itself is fine
    "no_data": 0,
}


@dataclass
class _ProviderState:
    name: str
    ok: int = 0
    failed: int = 0
    consecutive_failures: int = 0
    last_ok: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_kind: Optional[str] = None
    last_error_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    unsupported: set = field(default_factory=set)   # asset classes it can't serve


class ProviderHealth:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, _ProviderState] = {}

    def _get(self, name: str) -> _ProviderState:
        if name not in self._state:
            self._state[name] = _ProviderState(name)
        return self._state[name]

    def available(self, name: str, asset_class: str) -> bool:
        with self._lock:
            st = self._get(name)
            if asset_class in st.unsupported:
                return False
            return not (st.cooldown_until and datetime.now() < st.cooldown_until)

    def record_ok(self, name: str):
        with self._lock:
            st = self._get(name)
            st.ok += 1
            st.consecutive_failures = 0
            st.last_ok = datetime.now()
            st.cooldown_until = None

    def record_fail(self, name: str, err: ProviderError, asset_class: str = ""):
        with self._lock:
            st = self._get(name)
            st.failed += 1
            st.last_error = err.message[:300]
            st.last_error_kind = err.kind
            st.last_error_at = datetime.now()
            if err.kind in ("no_data",):
                return  # a symbol simply has no data here; provider is healthy
            if err.kind == "unsupported" and asset_class:
                st.unsupported.add(asset_class)
                return
            st.consecutive_failures += 1
            base = _COOLDOWN.get(err.kind, 30)
            if err.kind in ("network", "rate_limit"):
                # escalate: 30s, 60s, 120s … capped at 15 min
                base = min(base * (2 ** max(0, st.consecutive_failures - 1)), 900)
            if base:
                st.cooldown_until = datetime.now() + timedelta(seconds=base)

    def reset(self, name: Optional[str] = None):
        with self._lock:
            if name:
                self._state.pop(name, None)
            else:
                self._state.clear()

    def snapshot(self) -> dict:
        now = datetime.now()
        out = {}
        with self._lock:
            for name, st in self._state.items():
                cooling = bool(st.cooldown_until and now < st.cooldown_until)
                out[name] = {
                    "ok": st.ok,
                    "failed": st.failed,
                    "status": "cooldown" if cooling else ("ok" if st.last_ok and st.consecutive_failures == 0 else ("error" if st.failed else "idle")),
                    "last_ok": st.last_ok.isoformat(timespec="seconds") if st.last_ok else None,
                    "last_error": st.last_error,
                    "last_error_kind": st.last_error_kind,
                    "last_error_at": st.last_error_at.isoformat(timespec="seconds") if st.last_error_at else None,
                    "cooldown_until": st.cooldown_until.isoformat(timespec="seconds") if cooling else None,
                    "unsupported": sorted(st.unsupported),
                }
        return out


HEALTH = ProviderHealth()


# ════════════════════════════════════════════════════════════════════════════
#  HTTP (curl_cffi Chrome impersonation when available)
# ════════════════════════════════════════════════════════════════════════════

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
_BASE_HEADERS = {"User-Agent": _UA, "Accept": "application/json, text/plain, */*",
                 "Accept-Language": "en-US,en;q=0.9"}
_tls = threading.local()

try:
    from curl_cffi import requests as _cffi_requests  # type: ignore
    HAVE_CURL_CFFI = True
except Exception:  # pragma: no cover
    _cffi_requests = None
    HAVE_CURL_CFFI = False


def _session(kind: str = "browser"):
    """Thread-local HTTP session. 'browser' impersonates Chrome (for Yahoo/NSE)."""
    key = f"s_{kind}"
    s = getattr(_tls, key, None)
    if s is None:
        if kind == "browser" and HAVE_CURL_CFFI:
            s = _cffi_requests.Session(impersonate="chrome")
        else:
            import requests
            s = requests.Session()
        s.headers.update(_BASE_HEADERS)
        setattr(_tls, key, s)
    return s


def _reset_session(kind: str = "browser"):
    setattr(_tls, f"s_{kind}", None)


def _http_get(url: str, *, params: Optional[dict] = None, kind: str = "browser",
              headers: Optional[dict] = None, timeout: float = HTTP_TIMEOUT):
    try:
        return _session(kind).get(url, params=params, headers=headers, timeout=timeout)
    except Exception as e:  # DNS, TLS, timeout, connection refused …
        _reset_session(kind)
        raise ProviderError("network", f"{type(e).__name__}: {str(e)[:160]}")


def _json(resp, provider: str) -> dict:
    try:
        return resp.json()
    except Exception:
        snippet = (getattr(resp, "text", "") or "")[:120].replace("\n", " ")
        raise ProviderError("no_data", f"{provider}: non-JSON response (HTTP {resp.status_code}): {snippet}")


# ════════════════════════════════════════════════════════════════════════════
#  Symbols, periods, intervals
# ════════════════════════════════════════════════════════════════════════════

_INDEX_ALIASES = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY 50": "^NSEI", "NSEI": "^NSEI",
    "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK", "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN", "INDIAVIX": "^INDIAVIX", "VIX": "^VIX",
}
_CRYPTO_QUOTES = ("-USD", "-USDT", "-INR", "-EUR", "-BTC", "-ETH")


def normalize_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if not s:
        return s
    if s in _INDEX_ALIASES:
        return _INDEX_ALIASES[s]
    if s.endswith(".BSE"):
        s = s[:-4] + ".BO"
    if s.startswith("^") or "=" in s or "." in s or s.endswith(_CRYPTO_QUOTES):
        return s
    return s + ".NS"          # bare tickers are NSE equities (incl. BAJAJ-AUTO, M&M)


def asset_class(symbol: str) -> str:
    s = normalize_symbol(symbol)
    if s.endswith((".NS", ".BO")):
        return "india_equity"
    if s.startswith("^"):
        return "index"
    if s.endswith(_CRYPTO_QUOTES):
        return "crypto"
    if "=" in s:
        return "futures_fx"
    return "global_equity"


def bare_symbol(symbol: str) -> str:
    return re.sub(r"\.(NS|BO)$", "", normalize_symbol(symbol))


_PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "1wk": 7, "1mo": 31, "2mo": 62, "3mo": 92,
                "6mo": 183, "ytd": 366, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653, "max": 10000}


def period_days(period: str) -> int:
    p = (period or "6mo").lower()
    if p in _PERIOD_DAYS:
        return _PERIOD_DAYS[p]
    m = re.fullmatch(r"(\d+)(d|mo|y)", p)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return n * {"d": 1, "mo": 31, "y": 366}[unit]
    return 183


_INTERVAL_ALIASES = {"60m": "1h", "1hr": "1h", "1day": "1d", "daily": "1d", "1w": "1wk", "weekly": "1wk"}
_INTRADAY = {"1m", "5m", "15m", "30m", "1h", "4h"}
# Yahoo intraday lookback limits (days)
_YAHOO_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "4h": 729}


def norm_interval(interval: str) -> str:
    i = (interval or "1d").lower()
    return _INTERVAL_ALIASES.get(i, i)


def _ttl(interval: str) -> int:
    return {"1m": 60, "5m": 120, "15m": 300, "30m": 600, "1h": 900, "4h": 1800,
            "1d": 4 * 3600, "1wk": 12 * 3600}.get(interval, 3600)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, index and dtypes; drop empty rows."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.capitalize() for c in df.columns if isinstance(c, str)})
    if "Adj close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj close"]
    missing = [c for c in OHLCV if c not in df.columns]
    if "Close" in missing:
        return pd.DataFrame(columns=OHLCV)
    for c in missing:
        df[c] = df["Close"] if c != "Volume" else 0
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0)
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("4h", origin="start_day", offset="15min").agg(agg).dropna(subset=["Close"])


# ════════════════════════════════════════════════════════════════════════════
#  Providers — each returns a DataFrame or raises ProviderError
# ════════════════════════════════════════════════════════════════════════════

_YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
_yahoo_host_idx = 0


def _yahoo_chart(symbol: str, params: dict) -> dict:
    global _yahoo_host_idx
    last_err = None
    for attempt in range(2):
        host = _YAHOO_HOSTS[(_yahoo_host_idx + attempt) % 2]
        resp = _http_get(f"{host}/v8/finance/chart/{quote(symbol, safe='')}", params=params)
        if resp.status_code == 429:
            last_err = ProviderError("rate_limit", "Yahoo HTTP 429 (Too Many Requests)")
            continue
        if resp.status_code == 404:
            raise ProviderError("no_data", f"Yahoo: symbol {symbol} not found")
        if resp.status_code != 200:
            last_err = ProviderError("network", f"Yahoo HTTP {resp.status_code}")
            continue
        data = _json(resp, "Yahoo")
        err = (data.get("chart") or {}).get("error")
        if err:
            raise ProviderError("no_data", f"Yahoo: {err.get('description') or err}")
        _yahoo_host_idx = (_yahoo_host_idx + 1) % 2
        return data
    raise last_err or ProviderError("network", "Yahoo unreachable")


def _yahoo_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    yi = {"1h": "60m", "4h": "60m"}.get(interval, interval)
    max_days = _YAHOO_MAX_DAYS.get(interval)
    rng_days = min(days, max_days) if max_days else days
    if interval in _INTRADAY:
        rng = f"{max(1, rng_days)}d"
    else:
        rng = next((k for k in ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y")
                    if _PERIOD_DAYS[k] >= rng_days), "max")
    data = _yahoo_chart(symbol, {"range": rng, "interval": yi, "includePrePost": "false",
                                 "events": "div,splits"})
    res = (data.get("chart") or {}).get("result") or []
    if not res:
        raise ProviderError("no_data", f"Yahoo: empty result for {symbol}")
    r = res[0]
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts or not q:
        raise ProviderError("no_data", f"Yahoo: no bars for {symbol} ({interval})")
    tz = (r.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None)
    df = pd.DataFrame({"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
                       "Close": q.get("close"), "Volume": q.get("volume")}, index=idx)
    # Use split/dividend-adjusted closes for daily+ bars when available
    adj = ((r.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if adj and interval in ("1d", "1wk") and len(adj) == len(df):
        ratio = pd.Series(adj, index=idx) / df["Close"]
        ratio = ratio.where(ratio.notna(), 1.0)
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] * ratio
    df = _clean(df)
    if interval == "4h":
        df = _resample_4h(df)
    return df


def _yfinance_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ProviderError("unsupported", "yfinance not installed")
    yi = {"4h": "1h"}.get(interval, interval)
    max_days = _YAHOO_MAX_DAYS.get(interval)
    start = datetime.now() - timedelta(days=min(days, max_days) if max_days else days)
    kwargs = dict(start=start.strftime("%Y-%m-%d"), interval=yi, auto_adjust=True,
                  progress=False, threads=False, timeout=HTTP_TIMEOUT)
    try:
        df = yf.download(symbol, multi_level_index=False, **kwargs)
    except TypeError:          # very old yfinance without multi_level_index
        df = yf.download(symbol, **kwargs)
    except Exception as e:
        msg = str(e)
        kind = "rate_limit" if "Rate" in msg or "429" in msg else "network"
        raise ProviderError(kind, f"yfinance: {msg[:160]}")
    df = _clean(df)
    if df.empty:
        raise ProviderError("no_data", f"yfinance: no data for {symbol}")
    return _resample_4h(df) if interval == "4h" else df


def _twelve_data_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    key = get_secret("twelve_data_key")
    if not key:
        raise ProviderError("unsupported", "no Twelve Data key configured")
    ac = asset_class(symbol)
    params = {"apikey": key, "format": "JSON", "order": "ASC"}
    if ac == "india_equity":
        params["symbol"] = bare_symbol(symbol)
        params["exchange"] = "BSE" if symbol.endswith(".BO") else "NSE"
    elif ac == "crypto":
        params["symbol"] = symbol.replace("-", "/").replace("/USDT", "/USD")
    elif ac == "global_equity":
        params["symbol"] = symbol
    else:
        raise ProviderError("unsupported", f"Twelve Data: {ac} not mapped")
    ti = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
          "4h": "4h", "1d": "1day", "1wk": "1week"}.get(interval)
    if not ti:
        raise ProviderError("no_data", f"Twelve Data: interval {interval} unsupported")
    per_day = {"1m": 375, "5m": 75, "15m": 25, "30m": 13, "1h": 7, "4h": 2, "1d": 1, "1wk": 0.2}[interval]
    params["interval"] = ti
    params["outputsize"] = int(min(5000, max(30, days * per_day)))
    resp = _http_get("https://api.twelvedata.com/time_series", params=params, kind="plain")
    data = _json(resp, "Twelve Data")
    if data.get("status") == "error" or "values" not in data:
        code = data.get("code")
        msg = str(data.get("message", data))[:200]
        low = msg.lower()
        if code == 401 or "apikey" in low or "api key" in low:
            raise ProviderError("auth", f"Twelve Data: invalid API key — {msg}")
        if code == 429 or "credits" in low:
            kind = "quota" if "for the day" in low or "daily" in low else "rate_limit"
            raise ProviderError(kind, f"Twelve Data: {msg}")
        if "plan" in low or "grow" in low or "pro " in low or "upgrade" in low:
            raise ProviderError("unsupported", f"Twelve Data: {msg}")
        raise ProviderError("no_data", f"Twelve Data: {msg}")
    df = pd.DataFrame(data["values"]).set_index("datetime")
    return _clean(df)


def _alpha_vantage_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    key = get_secret("alpha_vantage_key")
    if not key:
        raise ProviderError("unsupported", "no Alpha Vantage key configured")
    if interval != "1d":
        raise ProviderError("no_data", "Alpha Vantage free tier: daily bars only")
    ac = asset_class(symbol)
    if ac == "india_equity":
        av_sym = bare_symbol(symbol) + ".BSE"
    elif ac == "global_equity":
        av_sym = symbol
    else:
        raise ProviderError("unsupported", f"Alpha Vantage: {ac} not mapped")
    resp = _http_get("https://www.alphavantage.co/query", kind="plain", params={
        "function": "TIME_SERIES_DAILY", "symbol": av_sym, "apikey": key,
        "outputsize": "compact"})
    data = _json(resp, "Alpha Vantage")
    ts = data.get("Time Series (Daily)")
    if not ts:
        info = str(data.get("Information") or data.get("Note") or data.get("Error Message") or data)[:220]
        low = info.lower()
        if "invalid api" in low or "apikey is invalid" in low:
            raise ProviderError("auth", f"Alpha Vantage: {info}")
        if "rate limit" in low or "requests per day" in low or "premium" in low or "frequency" in low:
            raise ProviderError("quota", f"Alpha Vantage: {info}")
        raise ProviderError("no_data", f"Alpha Vantage: {info}")
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
    return _clean(df)


_BINANCE_HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")


def _binance_pair(symbol: str) -> str:
    base, _, q = symbol.partition("-")
    q = {"USD": "USDT", "": "USDT"}.get(q, q)
    return f"{base}{q}"


def _binance_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    if asset_class(symbol) != "crypto" or symbol.endswith("-INR"):
        raise ProviderError("unsupported", "Binance: USD crypto pairs only")
    bi = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h",
          "1d": "1d", "1wk": "1w"}.get(interval)
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1, "1wk": 0.15}[interval]
    limit = int(min(1000, max(30, days * per_day)))
    last = None
    for host in _BINANCE_HOSTS:
        try:
            resp = _http_get(f"{host}/api/v3/klines", kind="plain",
                             params={"symbol": _binance_pair(symbol), "interval": bi, "limit": limit})
        except ProviderError as e:
            last = e
            continue
        if resp.status_code in (403, 451):
            last = ProviderError("network", f"Binance HTTP {resp.status_code} (region blocked)")
            continue
        if resp.status_code == 400:
            raise ProviderError("no_data", f"Binance: unknown pair {_binance_pair(symbol)}")
        if resp.status_code != 200:
            last = ProviderError("network", f"Binance HTTP {resp.status_code}")
            continue
        rows = _json(resp, "Binance")
        if not rows:
            raise ProviderError("no_data", "Binance: empty klines")
        df = pd.DataFrame([r[:6] for r in rows], columns=["t"] + OHLCV)
        df.index = pd.to_datetime(df.pop("t"), unit="ms")
        return _clean(df)
    raise last or ProviderError("network", "Binance unreachable")


def _dhan_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    from app.services.dhan_data import dhan_data_service, DhanError
    if not dhan_data_service.is_available():
        raise ProviderError("unsupported", "Dhan credentials not configured")
    if interval not in ("1d", "1m", "5m", "15m", "1h"):
        raise ProviderError("no_data", f"Dhan: interval {interval} unsupported")
    try:
        df = dhan_data_service.get_ohlcv(bare_symbol(symbol), days=days, interval=interval)
    except DhanError as e:
        raise ProviderError(e.kind, f"Dhan: {e}")
    df = _clean(df)
    if df.empty:
        raise ProviderError("no_data", f"Dhan: no bars for {symbol}")
    return df


def _upstox_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """Upstox v2 historical candles (needs the daily OAuth token via /upstox/login)."""
    try:
        from app.services.upstox_data import upstox_service
    except Exception:
        raise ProviderError("unsupported", "Upstox module unavailable")
    if not upstox_service.is_available():
        raise ProviderError("unsupported", "Upstox not connected (log in via /upstox/login)")
    if asset_class(symbol) != "india_equity" or symbol.endswith(".BO"):
        raise ProviderError("unsupported", "Upstox: NSE equities only")
    ui = {"15m": "15m", "30m": "30m", "1h": "30m", "1d": "1d"}.get(interval)
    if not ui:
        raise ProviderError("no_data", f"Upstox: interval {interval} unsupported")
    period = next((p for p, d in (("5d", 5), ("1mo", 31), ("3mo", 92), ("6mo", 183), ("1y", 366))
                   if days <= d), "2y")
    df = upstox_service.get_ohlcv(bare_symbol(symbol), period=period, interval=ui)
    if (df is None or df.empty) and not upstox_service.is_available():
        raise ProviderError("auth", "Upstox token expired — reconnect via /upstox/login")
    df = _clean(df)
    if interval == "1h" and not df.empty:
        df = df.resample("1h", origin="start_day", offset="15min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    return df


# NSE India public site — daily history (needs cookie warm-up; often bot-blocked)
_NSE = "https://www.nseindia.com"


def _nse_session():
    s = getattr(_tls, "nse", None)
    if s is None:
        s = _session("nse_browser") if not HAVE_CURL_CFFI else _cffi_requests.Session(impersonate="chrome")
        s.headers.update(_BASE_HEADERS)
        try:
            s.get(_NSE, timeout=HTTP_TIMEOUT)
        except Exception as e:
            raise ProviderError("network", f"NSE warm-up failed: {str(e)[:120]}")
        _tls.nse = s
    return s


def _nse_get(path: str, params: dict) -> dict:
    try:
        resp = _nse_session().get(f"{_NSE}{path}", params=params,
                                  headers={"Referer": _NSE + "/"}, timeout=HTTP_TIMEOUT)
    except ProviderError:
        raise
    except Exception as e:
        _tls.nse = None
        raise ProviderError("network", f"NSE: {str(e)[:120]}")
    if resp.status_code in (401, 403):
        _tls.nse = None
        raise ProviderError("rate_limit", f"NSE HTTP {resp.status_code} (bot protection)")
    if resp.status_code != 200:
        _tls.nse = None
        raise ProviderError("network", f"NSE HTTP {resp.status_code}")
    return _json(resp, "NSE")


def _nse_ohlcv(symbol: str, days: int, interval: str) -> pd.DataFrame:
    if asset_class(symbol) != "india_equity" or symbol.endswith(".BO"):
        raise ProviderError("unsupported", "NSE: NSE equities only")
    if interval != "1d":
        raise ProviderError("no_data", "NSE: daily bars only")
    end = datetime.now()
    frames = []
    # NSE caps each request at ~1 year
    remaining, cursor = min(days, 800), end
    while remaining > 0:
        chunk = min(remaining, 360)
        start = cursor - timedelta(days=chunk)
        data = _nse_get("/api/historical/cm/equity", {
            "symbol": bare_symbol(symbol), "series": '["EQ"]',
            "from": start.strftime("%d-%m-%Y"), "to": cursor.strftime("%d-%m-%Y")})
        rows = data.get("data") or []
        if rows:
            frames.append(pd.DataFrame({
                "Open": [r.get("CH_OPENING_PRICE") for r in rows],
                "High": [r.get("CH_TRADE_HIGH_PRICE") for r in rows],
                "Low": [r.get("CH_TRADE_LOW_PRICE") for r in rows],
                "Close": [r.get("CH_CLOSING_PRICE") for r in rows],
                "Volume": [r.get("CH_TOT_TRADED_QTY") for r in rows],
            }, index=pd.to_datetime([r.get("CH_TIMESTAMP") or r.get("mTIMESTAMP") for r in rows])))
        remaining -= chunk
        cursor = start - timedelta(days=1)
    df = _clean(pd.concat(frames) if frames else pd.DataFrame())
    if df.empty:
        raise ProviderError("no_data", f"NSE: no history for {symbol}")
    return df


_PROVIDERS: dict[str, Callable[[str, int, str], pd.DataFrame]] = {
    "dhan": _dhan_ohlcv,
    "yahoo": _yahoo_ohlcv,
    "yfinance": _yfinance_ohlcv,
    "nse": _nse_ohlcv,
    "twelve_data": _twelve_data_ohlcv,
    "alpha_vantage": _alpha_vantage_ohlcv,
    "binance": _binance_ohlcv,
    "upstox": _upstox_ohlcv,
}

# yfinance talks to the same Yahoo servers; if Yahoo is in network/rate-limit
# cooldown, skip it instead of paying another timeout per symbol.
_DEPENDS_ON = {"yfinance": "yahoo"}

_CHAINS = {
    "india_equity":  ["dhan", "upstox", "yahoo", "yfinance", "nse", "twelve_data", "alpha_vantage"],
    "index":         ["yahoo", "yfinance"],
    "crypto":        ["binance", "yahoo", "yfinance", "twelve_data"],
    "futures_fx":    ["yahoo", "yfinance"],
    "global_equity": ["yahoo", "yfinance", "twelve_data", "alpha_vantage"],
}


def _is_configured(provider: str) -> bool:
    if provider == "twelve_data":
        return bool(get_secret("twelve_data_key"))
    if provider == "alpha_vantage":
        return bool(get_secret("alpha_vantage_key"))
    if provider == "dhan":
        try:
            from app.services.dhan_data import dhan_data_service
            return dhan_data_service.is_available()
        except Exception:
            return False
    if provider == "nse":
        return env_flag("ASTRA_ENABLE_NSE", True)
    if provider == "upstox":
        try:
            from app.services.upstox_data import upstox_service
            return bool(upstox_service.is_available())
        except Exception:
            return False
    return True


# ════════════════════════════════════════════════════════════════════════════
#  Caches
# ════════════════════════════════════════════════════════════════════════════

def _disk_path(symbol: str, interval: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.replace("^", "IDX_").replace("=", "_F"))
    return DISK_CACHE_DIR / f"{safe}__{interval}.csv"


def _disk_put(symbol: str, interval: str, df: pd.DataFrame, source: str):
    try:
        out = df.copy()
        out["_source"] = source
        out.to_csv(_disk_path(symbol, interval))
    except Exception as e:
        logger.debug(f"disk cache write failed for {symbol}: {e}")


def _disk_get(symbol: str, interval: str) -> tuple[pd.DataFrame, Optional[str], Optional[datetime]]:
    path = _disk_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame(), None, None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        src = str(df["_source"].iloc[-1]) if "_source" in df.columns and len(df) else "disk"
        return _clean(df.drop(columns=["_source"], errors="ignore")), src, datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return pd.DataFrame(), None, None


def _expected_bars(days: int, interval: str, ac: str) -> int:
    trading = 1.0 if ac == "crypto" else 0.68
    per_day = {"1d": 1, "1wk": 1 / 7}.get(interval)
    if per_day is None:
        return 0      # intraday: don't second-guess provider depth
    return int(days * trading * per_day)


# ════════════════════════════════════════════════════════════════════════════
#  Public service
# ════════════════════════════════════════════════════════════════════════════

class MarketDataService:
    def __init__(self):
        self._mem: dict[tuple, tuple] = {}         # (sym, interval) → (df, fetched_at, days, source)
        self._price: dict[str, tuple] = {}         # sym → (price, ts, source)
        self._locks: dict[tuple, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.last_source: dict[str, dict] = {}     # sym → {"source", "stale", "at"}

    # ── helpers ────────────────────────────────────────────────────────────
    def _lock(self, key) -> threading.Lock:
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    @staticmethod
    def _slice(df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df.empty or days >= 10000:
            return df
        cutoff = df.index.max() - pd.Timedelta(days=days)
        return df[df.index >= cutoff]

    @staticmethod
    def _tag(df: pd.DataFrame, symbol: str, source: str, stale: bool) -> pd.DataFrame:
        df = df.copy()
        df.attrs = {"source": source, "stale": stale, "symbol": symbol}
        return df

    # ── OHLCV ──────────────────────────────────────────────────────────────
    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d",
                  min_bars: int = 1) -> pd.DataFrame:
        """Return OHLCV bars (possibly stale) or an empty DataFrame — never raises."""
        sym = normalize_symbol(symbol)
        itv = norm_interval(interval)
        days = period_days(period)
        if not sym:
            return pd.DataFrame(columns=OHLCV)
        key = (sym, itv)

        cached = self._mem.get(key)
        if cached:
            df, at, cdays, src = cached
            if (datetime.now() - at).total_seconds() < _ttl(itv) and cdays >= days:
                return self._tag(self._slice(df, days), sym, src, False)

        with self._lock(key):                          # collapse concurrent identical fetches
            cached = self._mem.get(key)
            if cached:
                df, at, cdays, src = cached
                if (datetime.now() - at).total_seconds() < _ttl(itv) and cdays >= days:
                    return self._tag(self._slice(df, days), sym, src, False)
            df, src = self._fetch_chain(sym, days, itv)

        if df is not None and not df.empty:
            self._mem[key] = (df, datetime.now(), days, src)
            _disk_put(sym, itv, df, src)
            self.last_source[sym] = {"source": src, "stale": False, "at": datetime.now().isoformat(timespec="seconds")}
            out = self._slice(df, days)
            if len(out) >= min_bars:
                return self._tag(out, sym, src, False)

        # Every provider failed → last good copy from disk, flagged stale
        disk, dsrc, dat = _disk_get(sym, itv)
        if not disk.empty:
            logger.warning(f"[data] {sym} {itv}: all providers failed — serving stale cache "
                           f"from {dat:%Y-%m-%d %H:%M} ({dsrc})")
            self.last_source[sym] = {"source": f"{dsrc} (cached {dat:%d %b %H:%M})", "stale": True,
                                     "at": datetime.now().isoformat(timespec="seconds")}
            return self._tag(self._slice(disk, days), sym, f"{dsrc} (stale)", True)

        self.last_source[sym] = {"source": None, "stale": True, "at": datetime.now().isoformat(timespec="seconds")}
        return pd.DataFrame(columns=OHLCV)

    def _fetch_chain(self, sym: str, days: int, itv: str) -> tuple[pd.DataFrame, Optional[str]]:
        ac = asset_class(sym)
        want = _expected_bars(days, itv, ac)
        best, best_src = pd.DataFrame(), None
        errors = []
        for name in _CHAINS.get(ac, ["yahoo", "yfinance"]):
            if not _is_configured(name) or not HEALTH.available(name, ac):
                continue
            dep = _DEPENDS_ON.get(name)
            if dep and not HEALTH.available(dep, ac):
                continue        # same upstream is down — don't wait on it twice
            t0 = time.monotonic()
            try:
                df = _PROVIDERS[name](sym, days, itv)
                if df is None or df.empty:
                    raise ProviderError("no_data", f"{name}: empty result for {sym} ({itv})")
            except ProviderError as e:
                HEALTH.record_fail(name, e, ac)
                errors.append(f"{name}: {e.message}")
                continue
            except Exception as e:  # provider bug — never let it break the chain
                err = ProviderError("network", f"{type(e).__name__}: {str(e)[:150]}")
                HEALTH.record_fail(name, err, ac)
                errors.append(f"{name}: {err.message}")
                continue
            HEALTH.record_ok(name)
            logger.info(f"[data] {sym} {itv}/{days}d ← {name}: {len(df)} bars ({time.monotonic() - t0:.1f}s)")
            if len(df) > len(best):
                best, best_src = df, name
            if not want or len(df) >= 0.6 * want:
                break            # good enough — stop here
        if best.empty and errors:
            logger.warning(f"[data] {sym} {itv}: no provider succeeded → " + " | ".join(errors[:4]))
        return best, best_src

    # ── Quotes ─────────────────────────────────────────────────────────────
    def get_quote_info(self, symbol: str, max_age: int = 30) -> dict:
        """{"price", "source", "stale", "ts"} — price 0.0 when unknown."""
        sym = normalize_symbol(symbol)
        hit = self._price.get(sym)
        if hit and (datetime.now() - hit[1]).total_seconds() < max_age:
            return {"price": hit[0], "source": hit[2], "stale": False, "ts": hit[1].isoformat(timespec="seconds")}

        ac = asset_class(sym)
        price, source = 0.0, None

        # 1. Dhan live WebSocket tick
        if ac == "india_equity":
            try:
                from app.services.dhan_feed import dhan_feed_manager
                p = dhan_feed_manager.get_price(bare_symbol(sym))
                if p and p > 0:
                    price, source = float(p), "dhan_ws"
            except Exception:
                pass
        # 2. Dhan REST quote
        if not price and ac == "india_equity" and _is_configured("dhan") and HEALTH.available("dhan", ac):
            try:
                from app.services.dhan_data import dhan_data_service
                p = dhan_data_service.get_quote(bare_symbol(sym))
                if p > 0:
                    price, source = p, "dhan"
            except Exception:
                pass
        # 3. Binance ticker (USD crypto)
        if not price and ac == "crypto" and not sym.endswith("-INR") and HEALTH.available("binance", ac):
            for host in _BINANCE_HOSTS:
                try:
                    r = _http_get(f"{host}/api/v3/ticker/price", kind="plain",
                                  params={"symbol": _binance_pair(sym)}, timeout=5)
                    if r.status_code == 200:
                        price, source = float(_json(r, "Binance")["price"]), "binance"
                        break
                except Exception:
                    continue
        # 4. Yahoo chart meta (works for every asset class)
        if not price and HEALTH.available("yahoo", ac):
            try:
                data = _yahoo_chart(sym, {"range": "1d", "interval": "1m", "includePrePost": "false"})
                meta = data["chart"]["result"][0]["meta"]
                p = meta.get("regularMarketPrice") or meta.get("previousClose")
                if p:
                    price, source = float(p), "yahoo"
                    HEALTH.record_ok("yahoo")
            except ProviderError as e:
                HEALTH.record_fail("yahoo", e, ac)
            except Exception:
                pass
        # 5. Last close from OHLCV (any provider, possibly stale)
        stale = False
        if not price:
            df = self.get_ohlcv(sym, period="5d", interval="1d")
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                stale = bool(df.attrs.get("stale"))
                source = f"last_close:{df.attrs.get('source')}"

        if price and math.isfinite(price) and price > 0:
            price = round(price, 2)
            if not stale:
                self._price[sym] = (price, datetime.now(), source)
            return {"price": price, "source": source, "stale": stale,
                    "ts": datetime.now().isoformat(timespec="seconds")}
        return {"price": 0.0, "source": None, "stale": True, "ts": None}

    def get_quote(self, symbol: str) -> float:
        return self.get_quote_info(symbol)["price"]

    def put_live_price(self, symbol: str, price: float, source: str = "dhan_ws"):
        self._price[normalize_symbol(symbol)] = (round(float(price), 2), datetime.now(), source)

    # ── News (Yahoo search — free, no key) ─────────────────────────────────
    def get_news(self, symbol: str, limit: int = 12) -> list[dict]:
        sym = normalize_symbol(symbol)
        if not HEALTH.available("yahoo", asset_class(sym)):
            return []
        try:
            q = bare_symbol(sym) if asset_class(sym) == "india_equity" else sym
            resp = _http_get(f"{_YAHOO_HOSTS[1]}/v1/finance/search",
                             params={"q": q, "newsCount": limit, "quotesCount": 0, "enableFuzzyQuery": "false"})
            if resp.status_code != 200:
                return []
            items = _json(resp, "Yahoo").get("news") or []
            return [{"title": n.get("title", ""), "publisher": n.get("publisher", ""),
                     "link": n.get("link", ""),
                     "published": datetime.fromtimestamp(n["providerPublishTime"]).isoformat()
                     if n.get("providerPublishTime") else None}
                    for n in items if n.get("title")]
        except Exception as e:
            logger.debug(f"news fetch failed for {sym}: {e}")
            return []

    # ── Fundamentals (yfinance handles Yahoo's crumb/cookie dance) ──────────
    def get_fundamentals(self, symbol: str, timeout: float = 12.0) -> dict:
        sym = normalize_symbol(symbol)
        try:
            import yfinance as yf
        except ImportError:
            return {}
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(lambda: yf.Ticker(sym).info or {})
        try:
            info = fut.result(timeout=timeout)
            HEALTH.record_ok("yahoo_fundamentals")
            return info if isinstance(info, dict) else {}
        except FTE:
            HEALTH.record_fail("yahoo_fundamentals", ProviderError("network", "timed out"))
            return {}
        except Exception as e:
            HEALTH.record_fail("yahoo_fundamentals", ProviderError("network", str(e)[:160]))
            return {}
        finally:
            ex.shutdown(wait=False)

    # ── Diagnostics ────────────────────────────────────────────────────────
    def health(self) -> dict:
        snap = HEALTH.snapshot()
        providers = {}
        for name in ["dhan", "upstox", "yahoo", "yfinance", "nse", "binance", "twelve_data", "alpha_vantage", "yahoo_fundamentals"]:
            entry = snap.get(name, {"ok": 0, "failed": 0, "status": "idle", "last_ok": None,
                                    "last_error": None, "last_error_kind": None,
                                    "cooldown_until": None, "unsupported": []})
            entry["configured"] = _is_configured(name) if name in _PROVIDERS else True
            if not entry["configured"]:
                entry["status"] = "not_configured"
            providers[name] = entry
        return {
            "providers": providers,
            "curl_cffi": HAVE_CURL_CFFI,
            "chains": _CHAINS,
            "cached_series": len(self._mem),
            "disk_cache_files": len(list(DISK_CACHE_DIR.glob("*.csv"))),
        }

    def probe(self, symbols: Optional[list] = None) -> dict:
        """Actively test each provider once (used by /api/data/health?probe=true)."""
        symbols = symbols or ["RELIANCE.NS", "^NSEI", "BTC-USD"]
        HEALTH.reset()
        self._mem.clear()
        self._price.clear()
        results = {}
        for sym in symbols:
            ac = asset_class(sym)
            for name in _CHAINS.get(ac, []):
                if not _is_configured(name):
                    results[f"{name}:{sym}"] = {"ok": False, "detail": "not configured"}
                    continue
                t0 = time.monotonic()
                try:
                    df = _PROVIDERS[name](sym, 31, "1d")
                    HEALTH.record_ok(name)
                    results[f"{name}:{sym}"] = {"ok": True, "bars": len(df),
                                                "last_close": round(float(df["Close"].iloc[-1]), 2) if len(df) else None,
                                                "last_date": str(df.index[-1].date()) if len(df) else None,
                                                "secs": round(time.monotonic() - t0, 2)}
                except ProviderError as e:
                    HEALTH.record_fail(name, e, ac)
                    results[f"{name}:{sym}"] = {"ok": False, "kind": e.kind, "detail": e.message,
                                                "secs": round(time.monotonic() - t0, 2)}
                except Exception as e:
                    results[f"{name}:{sym}"] = {"ok": False, "kind": "bug", "detail": f"{type(e).__name__}: {e}"}
        return results

    def clear_cache(self, symbol: Optional[str] = None):
        if symbol:
            sym = normalize_symbol(symbol)
            self._mem = {k: v for k, v in self._mem.items() if k[0] != sym}
            self._price.pop(sym, None)
        else:
            self._mem.clear()
            self._price.clear()


market_data = MarketDataService()
