"""
Offline tests for the v1.13 market-data layer. Provider HTTP calls are replaced
with canned responses in each provider's real wire format, so these run
without network access.
"""
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services import market_data as md
from app.services.market_data import ProviderError, market_data


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def yahoo_chart(n=300, start=1000.0, interval_sec=86400, tz="Asia/Kolkata"):
    t0 = int(time.time()) - n * interval_sec
    ts = [t0 + i * interval_sec for i in range(n)]
    closes = list(start + np.cumsum(np.random.default_rng(1).normal(0, 5, n)))
    return {"chart": {"result": [{
        "meta": {"regularMarketPrice": closes[-1], "exchangeTimezoneName": tz},
        "timestamp": ts,
        "indicators": {"quote": [{"open": closes, "high": [c + 3 for c in closes],
                                  "low": [c - 3 for c in closes], "close": closes,
                                  "volume": [100000] * n}],
                       "adjclose": [{"adjclose": closes}]},
    }], "error": None}}


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    md.HEALTH.reset()
    market_data.clear_cache()
    market_data.last_source.clear()
    monkeypatch.setattr(md, "DISK_CACHE_DIR", tmp_path)
    yield


# ── symbols ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected,cls", [
    ("reliance", "RELIANCE.NS", "india_equity"),
    ("RELIANCE.NS", "RELIANCE.NS", "india_equity"),
    ("TCS.BSE", "TCS.BO", "india_equity"),
    ("BAJAJ-AUTO", "BAJAJ-AUTO.NS", "india_equity"),
    ("M&M", "M&M.NS", "india_equity"),
    ("NIFTY", "^NSEI", "index"),
    ("^NSEI", "^NSEI", "index"),
    ("BTC-USD", "BTC-USD", "crypto"),
    ("BTC-INR", "BTC-INR", "crypto"),
    ("GC=F", "GC=F", "futures_fx"),
])
def test_normalize(raw, expected, cls):
    assert md.normalize_symbol(raw) == expected
    assert md.asset_class(raw) == cls


# ── Yahoo ───────────────────────────────────────────────────────────────────

def test_yahoo_parses_and_caches(monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return FakeResp(200, yahoo_chart(500))
    monkeypatch.setattr(md, "_http_get", fake_get)
    df = market_data.get_ohlcv("RELIANCE", period="2y", interval="1d")
    assert len(df) > 400
    assert list(df.columns) == md.OHLCV
    assert df.attrs["source"] == "yahoo" and df.attrs["stale"] is False
    assert df.index.tz is None
    n = len(calls)
    # shorter period is served from the 2y cache — no new call
    df6 = market_data.get_ohlcv("RELIANCE.NS", period="6mo", interval="1d")
    assert len(calls) == n and 100 < len(df6) < len(df)


def test_cache_does_not_serve_short_history_for_long_request(monkeypatch):
    """v1.12 bug: cache keyed only on symbol+interval, so a 1mo fetch was
    returned when the ML engine asked for 2y."""
    sizes = iter([25, 500])
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, yahoo_chart(next(sizes))))
    assert len(market_data.get_ohlcv("TCS", period="1mo")) <= 25
    assert len(market_data.get_ohlcv("TCS", period="2y")) > 400


def test_yahoo_429_triggers_cooldown_and_fallback(monkeypatch):
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(429, {}))
    monkeypatch.setattr(md, "_yfinance_ohlcv", lambda *a: (_ for _ in ()).throw(AssertionError("yfinance should be skipped while Yahoo cools down")))
    nse_df = md._clean(pd.DataFrame({"Close": np.linspace(100, 120, 300)},
                                    index=pd.bdate_range(end=datetime.now(), periods=300)))
    monkeypatch.setitem(md._PROVIDERS, "nse", lambda s, d, i: nse_df)
    df = market_data.get_ohlcv("INFY", period="1y")
    assert df.attrs["source"] == "nse"
    snap = md.HEALTH.snapshot()
    assert snap["yahoo"]["status"] == "cooldown" and snap["yahoo"]["last_error_kind"] == "rate_limit"


def test_network_failure_serves_stale_disk_copy(monkeypatch):
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, yahoo_chart(300)))
    assert not market_data.get_ohlcv("HDFCBANK", period="1y").empty
    # everything down now; memory cache cleared → disk copy is used
    market_data.clear_cache()
    md.HEALTH.reset()

    def down(url, **kw):
        raise ProviderError("network", "DNS failure")
    monkeypatch.setattr(md, "_http_get", down)
    monkeypatch.setitem(md._PROVIDERS, "yfinance", lambda *a: (_ for _ in ()).throw(ProviderError("network", "down")))
    monkeypatch.setitem(md._PROVIDERS, "nse", lambda *a: (_ for _ in ()).throw(ProviderError("network", "down")))
    df = market_data.get_ohlcv("HDFCBANK", period="1y")
    assert not df.empty and df.attrs["stale"] is True and "stale" in df.attrs["source"]


def test_total_failure_returns_empty_quickly(monkeypatch):
    def down(url, **kw):
        raise ProviderError("network", "offline")
    monkeypatch.setattr(md, "_http_get", down)
    monkeypatch.setitem(md._PROVIDERS, "yfinance", lambda *a: (_ for _ in ()).throw(ProviderError("network", "offline")))
    t = time.monotonic()
    for sym in ["A1", "A2", "A3", "A4", "A5"]:
        assert market_data.get_ohlcv(sym).empty
    assert time.monotonic() - t < 2.0


# ── keyed providers: error classification ───────────────────────────────────

def test_twelve_data_bad_key_is_auth(monkeypatch):
    monkeypatch.setattr(md, "get_secret", lambda n: "bad" if n == "twelve_data_key" else None)
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, {
        "code": 401, "message": "**apikey** parameter is incorrect or not specified.", "status": "error"}))
    with pytest.raises(ProviderError) as e:
        md._twelve_data_ohlcv("AAPL", 31, "1d")
    assert e.value.kind == "auth"


def test_twelve_data_daily_credits_is_quota(monkeypatch):
    monkeypatch.setattr(md, "get_secret", lambda n: "k" if n == "twelve_data_key" else None)
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, {
        "code": 429, "message": "You have run out of API credits for the day.", "status": "error"}))
    with pytest.raises(ProviderError) as e:
        md._twelve_data_ohlcv("BTC-USD", 31, "1d")
    assert e.value.kind == "quota"


def test_twelve_data_plan_restriction_marks_asset_class(monkeypatch):
    monkeypatch.setattr(md, "get_secret", lambda n: "k" if n == "twelve_data_key" else None)
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, {
        "code": 404, "message": "This symbol is available starting with the Grow plan.", "status": "error"}))
    with pytest.raises(ProviderError) as e:
        md._twelve_data_ohlcv("RELIANCE.NS", 31, "1d")
    assert e.value.kind == "unsupported"
    md.HEALTH.record_fail("twelve_data", e.value, "india_equity")
    assert not md.HEALTH.available("twelve_data", "india_equity")
    assert md.HEALTH.available("twelve_data", "crypto")


def test_alpha_vantage_rate_limit_is_quota(monkeypatch):
    monkeypatch.setattr(md, "get_secret", lambda n: "k" if n == "alpha_vantage_key" else None)
    monkeypatch.setattr(md, "_http_get", lambda url, **kw: FakeResp(200, {
        "Information": "We have detected your API key as ... standard API rate limit is 25 requests per day."}))
    with pytest.raises(ProviderError) as e:
        md._alpha_vantage_ohlcv("RELIANCE.NS", 100, "1d")
    assert e.value.kind == "quota"


def test_public_alpha_vantage_key_is_never_used(monkeypatch):
    from app.core import config
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "XV1FMHS5UHPIIPAZ")
    monkeypatch.setattr(config, "_db_setting", lambda k: None)
    assert config.get_secret("alpha_vantage_key") is None


def test_placeholder_keys_are_treated_as_missing(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "_db_setting", lambda k: None)
    monkeypatch.setenv("DHAN_CLIENT_ID", "ENTER_YOUR_DHAN_CLIENT_ID_HERE")
    assert config.get_secret("dhan_client_id") is None
    monkeypatch.setenv("DHAN_CLIENT_ID", "1100123456")
    assert config.get_secret("dhan_client_id") == "1100123456"


# ── Binance ─────────────────────────────────────────────────────────────────

def test_binance_klines_and_region_block_fallback(monkeypatch):
    now_ms = int(time.time() * 1000)
    rows = [[now_ms - (60 - i) * 86_400_000, "100", "110", "95", str(100 + i), "12.5", 0, "0", 0, "0", "0", "0"]
            for i in range(60)]
    seen = []

    def fake(url, **kw):
        seen.append(url)
        if "binance.vision" in url:
            return FakeResp(451, {"msg": "restricted location"})
        return FakeResp(200, rows)
    monkeypatch.setattr(md, "_http_get", fake)
    df = md._binance_ohlcv("BTC-USD", 60, "1d")
    assert len(df) == 60 and df["Close"].iloc[-1] == 159
    assert any("binance.vision" in u for u in seen) and any("api.binance.com" in u for u in seen)


# ── Dhan ────────────────────────────────────────────────────────────────────

def test_dhan_parses_nested_data_and_classifies_errors():
    from app.services.dhan_data import DhanDataService, DhanError
    svc = DhanDataService()
    now = int(time.time())
    ok = {"status": "success", "remarks": "", "data": {
        "open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1, 2], "close": [1.5, 2.5, 3.5],
        "volume": [10, 20, 30], "timestamp": [now - 172800, now - 86400, now]}}
    df = svc._parse_candles(svc._unwrap(ok))
    assert list(df["Close"]) == [1.5, 2.5, 3.5]
    with pytest.raises(DhanError) as e:
        svc._unwrap({"status": "failure", "remarks": {"error_code": "DH-901", "error_message": "Invalid token"}, "data": ""})
    assert e.value.kind == "auth"


# ── quotes ──────────────────────────────────────────────────────────────────

def test_quote_from_yahoo_meta_then_cached(monkeypatch):
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return FakeResp(200, yahoo_chart(5, start=2500.0))
    monkeypatch.setattr(md, "_http_get", fake)
    info = market_data.get_quote_info("TCS")
    assert info["price"] > 0 and info["source"] == "yahoo"
    n = len(calls)
    assert market_data.get_quote("TCS.NS") == info["price"]
    assert len(calls) == n


def test_provider_missing_latest_session_loses_to_fresh_one(monkeypatch):
    """Upstox historical candles exclude today; a provider that is up to date must win."""
    days = pd.bdate_range(end=pd.Timestamp.now() - pd.Timedelta(days=7), periods=400)
    stale = md._clean(pd.DataFrame({"Close": np.linspace(100, 120, 400)}, index=days))
    fresh = md._clean(pd.DataFrame({"Close": np.linspace(100, 119, 380)},
                                   index=pd.bdate_range(end=pd.Timestamp.now(), periods=380)))
    monkeypatch.setitem(md._CHAINS, "india_equity", ["upstox", "yahoo"])
    monkeypatch.setattr(md, "_is_configured", lambda n: True)
    monkeypatch.setitem(md._PROVIDERS, "upstox", lambda s, d, i: stale)
    monkeypatch.setitem(md._PROVIDERS, "yahoo", lambda s, d, i: fresh)
    df = market_data.get_ohlcv("RELIANCE", period="2y")
    assert df.attrs["source"] == "yahoo"
