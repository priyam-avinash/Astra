"""
Broker safety lock — the most safety-critical tests in the codebase.

These tests verify that NO code path can execute a real broker order.
If any of these fail, the system can lose real money. Treat with paranoia.

Three independent locks must each hold:
  1. PAPER_MODE constant is hardcoded True (cannot be flipped via env)
  2. broker_service singleton is always PaperTradingEngine
  3. DhanLiveBroker.execute_trade raises unconditionally even if instantiated directly
"""

import os

import pytest


def test_paper_mode_constant_is_true_regardless_of_env(monkeypatch):
    """The PAPER_MODE constant must remain True even if env says false."""
    monkeypatch.setenv("PAPER_TRADING", "false")
    # Reload the module after env is set
    import importlib
    import app.services.broker as br
    importlib.reload(br)
    assert br.PAPER_MODE is True, (
        "CRITICAL: PAPER_MODE flipped to False — env override should be impossible"
    )


def test_broker_service_is_always_paper_trading_engine():
    """broker_service singleton must always be PaperTradingEngine."""
    from app.services.broker import broker_service, PaperTradingEngine
    assert isinstance(broker_service, PaperTradingEngine), (
        f"CRITICAL: broker_service is {type(broker_service).__name__}, not PaperTradingEngine"
    )


def test_dhan_live_broker_execute_trade_raises_unconditionally():
    """Even if someone instantiates DhanLiveBroker directly, execute_trade must raise."""
    from app.services.broker import DhanLiveBroker
    live = DhanLiveBroker()
    with pytest.raises(RuntimeError, match="LIVE ORDER BLOCKED"):
        live.execute_trade("RELIANCE", "BUY", 10, 1000.0)


def test_dhan_live_broker_raises_even_with_credentials(monkeypatch):
    """Hard block must fire regardless of credentials being set."""
    monkeypatch.setenv("DHAN_CLIENT_ID", "real_client_id")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "real_token_string")
    from app.services.broker import DhanLiveBroker
    live = DhanLiveBroker()
    with pytest.raises(RuntimeError, match="LIVE ORDER BLOCKED"):
        live.execute_trade("RELIANCE", "SELL", 1, 100.0)


def test_paper_trade_returns_mode_paper(monkeypatch):
    """Sanity: paper trades must label themselves PAPER and never LIVE."""
    from app.services.market_data import market_data
    # v1.13: paper fills need a live (non-stale) quote — stub it so this runs offline
    monkeypatch.setattr(market_data, "get_quote_info",
                        lambda s, max_age=30: {"price": 1000.0, "source": "test", "stale": False, "ts": None})
    from app.services.broker import broker_service
    result = broker_service.execute_trade("RELIANCE", "BUY", 1, 1000.0)
    assert result["mode"] == "PAPER", (
        f"Paper trade returned mode={result['mode']} — should be PAPER"
    )
    assert result["order_id"].startswith("PAPER-"), (
        "Order ID should be prefixed PAPER-"
    )


def test_auth_bypass_in_live_mode_refuses_startup(monkeypatch):
    """If AUTH_MODE=bypass and PAPER_TRADING=false, importing endpoints must raise."""
    monkeypatch.setenv("AUTH_MODE", "bypass")
    monkeypatch.setenv("PAPER_TRADING", "false")
    import importlib
    import app.api.endpoints as ep_mod
    with pytest.raises(RuntimeError, match="REFUSING TO START"):
        importlib.reload(ep_mod)
