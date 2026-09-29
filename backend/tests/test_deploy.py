"""
Hosted-deploy safeguards (Vercel): closed registration, owner allowlist,
protected data endpoints, and the numpy Random Forest used without sklearn.
"""
import os

import numpy as np
import pytest


def test_auth_mode_endpoint_is_public(jwt_client):
    r = jwt_client.get("/api/auth/mode")
    assert r.status_code == 200
    assert r.json()["mode"] == "jwt"
    assert "registration_open" in r.json()


def test_hosted_registration_closes_after_first_user(jwt_client, monkeypatch, random_username):
    # The shared test DB already has users, so a hosted server must refuse new ones.
    jwt_client.post("/api/auth/register", json={"username": random_username, "password": "pw_test_123"})
    monkeypatch.setenv("ASTRA_OPEN_REGISTRATION", "false")
    r = jwt_client.post("/api/auth/register", json={"username": random_username + "x", "password": "pw_test_123"})
    assert r.status_code == 403
    assert jwt_client.get("/api/auth/mode").json()["registration_open"] is False


def test_allowlist_only_admits_listed_users(jwt_client, monkeypatch, random_username):
    monkeypatch.setenv("ASTRA_ALLOWED_USERS", f"{random_username}@example.com")
    bad = jwt_client.post("/api/auth/register", json={"username": "intruder@example.com", "password": "pw_test_123"})
    assert bad.status_code == 403
    ok = jwt_client.post("/api/auth/register", json={"username": f"{random_username}@EXAMPLE.com", "password": "pw_test_123"})
    assert ok.status_code == 200
    tok = jwt_client.post("/api/auth/token", data={"username": f"{random_username}@example.com", "password": "pw_test_123"})
    assert tok.status_code == 200


@pytest.mark.parametrize("path", ["/api/scan/universe", "/api/crypto/analyze/BTC-USD",
                                  "/api/quote/RELIANCE", "/api/market/pulse", "/api/settings"])
def test_data_endpoints_require_login(jwt_client, path):
    assert jwt_client.get(path).status_code == 401


def test_backtest_requires_login(jwt_client):
    assert jwt_client.post("/strategies/astra/backtest").status_code == 401


def test_numpy_forest_matches_sklearn(tmp_path):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor
    from app.services.tree_ensemble import export_forest, NumpyForestRegressor
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = X[:, 0] * 2 - X[:, 3] + rng.normal(scale=0.1, size=300)
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0).fit(X, y)
    path = tmp_path / "rf.npz"
    export_forest(rf, str(path))
    f = NumpyForestRegressor.load(str(path))
    Xt = rng.normal(size=(50, 6))
    assert np.allclose(rf.predict(Xt), f.predict(Xt), atol=1e-9)


def test_committed_numpy_forest_loads():
    from app.services.tree_ensemble import NumpyForestRegressor
    here = os.path.dirname(__file__)
    f = NumpyForestRegressor.load(os.path.join(here, "..", "app", "models", "saved_models", "astra_rf_trees.npz"))
    assert f.n_features_in_ == 20
    assert np.isfinite(f.predict(np.zeros((1, 20)))).all()
