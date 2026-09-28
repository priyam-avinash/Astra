"""
Basic API smoke tests.
Detailed tests for auth / brokers / strategies live in their own test files.
"""


def test_root_returns_status_ok(bypass_client):
    r = bypass_client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "ASTRA" in body.get("message", "")


def test_health_returns_healthy(bypass_client):
    r = bypass_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_brokers_endpoint_lists_registered_brokers(bypass_client):
    r = bypass_client.get("/brokers")
    assert r.status_code == 200
    brokers = r.json().get("brokers", [])
    assert len(brokers) >= 4
    for b in brokers:
        assert "name" in b and "priority" in b and "available" in b


def test_strategies_endpoint_lists_engines(bypass_client):
    r = bypass_client.get("/strategies")
    assert r.status_code == 200
    strats = r.json().get("strategies", [])
    assert len(strats) >= 4
    names = {s["name"] for s in strats}
    assert "ASTRA.STAGE2" in names
    assert "ASTRA.PULLBACK" in names


def test_strategy_detail_endpoint_unknown_returns_404(bypass_client):
    r = bypass_client.get("/strategies/ASTRA.NONEXISTENT")
    assert r.status_code == 404


def test_upstox_status_endpoint_exists(bypass_client):
    r = bypass_client.get("/upstox/status")
    assert r.status_code == 200
    body = r.json()
    assert "available" in body
    assert "login_url" in body
