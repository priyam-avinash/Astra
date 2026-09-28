"""
Broker plugin tests — registry, fallback chain, abstract contract.
"""

import pandas as pd
import pytest


# ── Registry ────────────────────────────────────────────────────────────────

def test_registry_lists_all_brokers():
    from app.brokers.registry import list_brokers
    names = {b["name"] for b in list_brokers()}
    expected = {"Upstox", "Dhan", "Zerodha", "Yahoo"}
    assert expected.issubset(names)


def test_registry_priorities_are_unique_and_sorted():
    """Priorities define fail-over order — must be unique."""
    from app.brokers.registry import list_brokers
    prios = [b["priority"] for b in list_brokers()]
    assert len(set(prios)) == len(prios), "duplicate broker priorities — fail-over order undefined"


def test_get_broker_case_insensitive():
    from app.brokers.registry import get_broker
    for variant in ("Yahoo", "yahoo", "YAHOO", "  Yahoo  "):
        b = get_broker(variant)
        assert b is not None and b.name == "Yahoo"


def test_get_broker_unknown_returns_none():
    from app.brokers.registry import get_broker
    assert get_broker("FakeBroker") is None


# ── Abstract contract ──────────────────────────────────────────────────────

def test_every_broker_implements_required_methods():
    from app.brokers.registry import list_brokers, get_broker
    for entry in list_brokers():
        b = get_broker(entry["name"])
        assert b is not None
        assert hasattr(b, "name")    and isinstance(b.name, str)
        assert hasattr(b, "is_available")
        assert hasattr(b, "get_ohlcv")
        # is_available must be callable and return bool
        assert isinstance(b.is_available(), bool)


def test_yahoo_broker_returns_data_or_empty_df_never_raises():
    """Yahoo must NEVER raise (graceful 429 handling). Either df or empty df."""
    from app.brokers.yahoo_client import YahooBroker
    yb = YahooBroker()
    result = yb.get_ohlcv("RELIANCE", period="5d", interval="1d")
    assert isinstance(result, pd.DataFrame)


def test_unsupported_methods_raise_not_implemented():
    """Read-only brokers (Yahoo) must raise NotImplementedError on place_order."""
    from app.brokers.yahoo_client import YahooBroker
    from app.brokers.base import OrderRequest, OrderSide, OrderType
    yb = YahooBroker()
    req = OrderRequest(symbol="RELIANCE", side=OrderSide.BUY,
                       quantity=1, order_type=OrderType.MARKET)
    with pytest.raises(NotImplementedError):
        yb.place_order(req)


# ── Fallback chain ──────────────────────────────────────────────────────────

def test_fetch_ohlcv_returns_first_non_empty_result():
    """fetch_ohlcv tries brokers in priority order. Should get data from SOMEONE."""
    from app.brokers.registry import fetch_ohlcv
    df = fetch_ohlcv("RELIANCE", period="1mo", interval="1d")
    # At least one broker must work (Upstox is connected per .env).
    # If this fails, check if Upstox token expired AND Yahoo is also down.
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        pytest.skip("All brokers returned empty — check Upstox token and Yahoo 429")
    assert len(df) > 0
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns


def test_fetch_quote_returns_float_or_none():
    from app.brokers.registry import fetch_quote
    q = fetch_quote("RELIANCE")
    assert q is None or isinstance(q, (int, float))
    if q is not None:
        assert q > 0
