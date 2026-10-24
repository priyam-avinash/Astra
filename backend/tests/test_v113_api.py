"""API-level tests: app boots, paper-only broker, execute → monitor → auto-exit."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from app.services import market_data as md
from app.services.market_data import market_data


@pytest.fixture(scope="module")
def client():
    # These tests exercise trading endpoints, so run with the dev auth bypass
    import importlib, os
    os.environ["AUTH_MODE"] = "bypass"
    os.environ["PAPER_TRADING"] = "true"
    import app.api.endpoints as ep_mod
    importlib.reload(ep_mod)
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    os.environ.pop("AUTH_MODE", None)


@pytest.fixture
def price(monkeypatch):
    """Controllable fake live price for every symbol."""
    box = {"p": 1000.0}
    monkeypatch.setattr(market_data, "get_quote", lambda s: box["p"])
    monkeypatch.setattr(market_data, "get_quote_info",
                        lambda s, max_age=30: {"price": box["p"], "source": "test", "stale": False, "ts": None})
    return box


def test_root_and_health(client):
    assert client.get("/").json()["mode"] == "PAPER"
    assert client.get("/health").json()["status"] == "healthy"


def test_broker_is_paper_only(monkeypatch):
    import importlib
    monkeypatch.setenv("PAPER_TRADING", "false")
    from app.services import broker
    importlib.reload(broker)
    assert broker.PAPER_MODE is True
    assert type(broker.broker_service).__name__ == "PaperTradingEngine"
    with pytest.raises(RuntimeError, match="LIVE ORDER BLOCKED"):
        broker.DhanLiveBroker().execute_trade("RELIANCE", "BUY", 1, 100.0)


def test_data_health_endpoint(client):
    d = client.get("/api/data/health").json()
    assert d["mode"] == "PAPER"
    assert {"yahoo", "binance", "twelve_data", "alpha_vantage", "dhan"} <= set(d["providers"])
    assert d["providers"]["twelve_data"]["status"] == "not_configured"


def test_signals_queue_has_no_fake_seed(client):
    assert client.get("/api/signals").json()["queue"] == []


def _make_signal(client, asset, side="BUY", entry=1480.0, sl=1450.0, tp=1550.0):
    r = client.post("/api/signals", json={"asset": asset, "type": "Equity", "signal": side, "confidence": 70,
                                          "entry_price": entry, "target_price": tp, "stop_loss": sl,
                                          "engine": "astra_ai"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_execute_records_fill_price_and_monitor_auto_exits(client, price, monkeypatch):
    monkeypatch.setattr("app.services.ai_predictor.AIPredictionEngine._fetch_data",
                        lambda self, *a, **k: pd.DataFrame())
    price["p"] = 1000.0
    sid = _make_signal(client, "TESTCO.NS")                 # stale signal priced at 1480
    r = client.post("/api/execute", json={"id": sid, "asset": "TESTCO.NS", "action": "BUY",
                                          "quantity": 10, "price": 1480.0,
                                          "target_price": 1550.0, "stop_loss": 1450.0})
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "PAPER"
    pos = [p for p in client.get("/api/positions").json()["positions"] if p["asset"] == "TESTCO.NS"][0]
    # entry = real LTP + 0.05% slippage, NOT the 1480 the client sent
    assert pos["entry_price"] == pytest.approx(1000.5, abs=0.01)
    # SL/TP keep the signal's distances (30 / 70) but are measured from the fill
    assert pos["stop_loss"] == pytest.approx(970.5, abs=0.01)
    assert pos["target_price"] == pytest.approx(1070.5, abs=0.01)
    assert pos["currency"] == "INR"

    # the same signal can't be executed twice
    r2 = client.post("/api/execute", json={"id": sid, "asset": "TESTCO.NS", "action": "BUY",
                                           "quantity": 10, "price": 1480.0})
    assert r2.status_code == 409

    # price moves through target → background monitor closes it
    from app.services.tasks import monitor_active_positions
    monkeypatch.setattr("app.services.tasks.dispatch", lambda *a, **k: None)
    price["p"] = 1080.0
    monitor_active_positions()
    open_assets = [p["asset"] for p in client.get("/api/positions").json()["positions"]]
    assert "TESTCO.NS" not in open_assets
    hist = [h for h in client.get("/api/history").json()["history"] if h["asset"] == "TESTCO.NS"]
    closing = [h for h in hist if "Target" in h["action"]]
    assert len(closing) == 1 and closing[0]["pnl"] > 0


def test_squareoff_twice_books_once(client, price, monkeypatch):
    monkeypatch.setattr("app.services.ai_predictor.AIPredictionEngine._fetch_data",
                        lambda self, *a, **k: pd.DataFrame())
    price["p"] = 500.0
    r = client.post("/api/execute/manual", json={"asset": "DBLCO.NS", "action": "BUY", "quantity": 4, "price": 500.0})
    assert r.status_code == 200, r.text
    pid = [p for p in client.get("/api/positions").json()["positions"] if p["asset"] == "DBLCO.NS"][0]["id"]
    price["p"] = 510.0
    assert client.post(f"/api/positions/squareoff/{pid}").status_code == 200
    assert client.post(f"/api/positions/squareoff/{pid}").status_code == 409
    closes = [h for h in client.get("/api/history").json()["history"]
              if h["asset"] == "DBLCO.NS" and "Square Off" in h["action"]]
    assert len(closes) == 1


@pytest.mark.market_closed
def test_nse_orders_rejected_when_market_closed(client, price):
    price["p"] = 100.0
    r = client.post("/api/execute/manual", json={"asset": "CLOSEDCO.NS", "action": "BUY", "quantity": 1, "price": 100})
    assert r.status_code == 503 and "closed" in r.json()["detail"].lower()


def test_manual_order_rejects_bad_input(client, price):
    assert client.post("/api/execute/manual", json={"asset": "X.NS", "action": "BUY",
                                                    "quantity": 0, "price": 10}).status_code == 400
    assert client.post("/api/execute/manual", json={"asset": "X.NS", "action": "HOLD",
                                                    "quantity": 1, "price": 10}).status_code == 400


def test_analyze_with_data(client, monkeypatch):
    n = 520
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = 1000 + np.cumsum(np.random.default_rng(7).normal(0.5, 8, n))
    df = pd.DataFrame({"Open": close - 2, "High": close + 6, "Low": close - 6, "Close": close,
                       "Volume": np.random.default_rng(3).integers(1e5, 5e5, n)}, index=idx)
    df.attrs = {"source": "yahoo", "stale": False}
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **k: df.copy())
    for engine in ("astra", "astra_ai"):
        d = client.get(f"/api/analyze/TESTCO?engine={engine}").json()
        assert "error" not in d, d.get("error")
        assert d["signal"] in ("BUY", "SELL", "HOLD")
        assert d["data_source"] == "yahoo" and d["chartData"]
        assert d["stop_loss"] != d["target"]


def test_settings_accept_data_keys(client):
    r = client.put("/api/settings", json={"settings": {"twelve_data_key": "abcd1234efgh"}})
    assert "twelve_data_key" in r.json()["updated"]
    s = client.get("/api/settings").json()
    assert s["twelve_data_key"].endswith("efgh") and s["twelve_data_key"].startswith("•")
    assert s["key_sources"]["twelve_data_key"] == "settings"
    client.put("/api/settings", json={"settings": {"twelve_data_key": ""}})


def test_no_fill_on_stale_price(client, monkeypatch):
    monkeypatch.setattr(market_data, "get_quote_info",
                        lambda s, max_age=30: {"price": 1234.0, "source": "last_close:yahoo (stale)", "stale": True, "ts": None})
    r = client.post("/api/execute/manual", json={"asset": "STALECO.NS", "action": "BUY", "quantity": 1, "price": 1200})
    assert r.status_code == 503 and "No live price" in r.json()["detail"]
    assert "STALECO.NS" not in [p["asset"] for p in client.get("/api/positions").json()["positions"]]
