"""
Auth tests — verify the JWT path is correctly gated and the bypass mode behaves.
"""

import pytest


def _register_and_login(client, username: str) -> str:
    r = client.post("/api/auth/register",
                    json={"username": username, "password": "pw_test_123"})
    assert r.status_code == 200, f"register failed: {r.text}"
    r = client.post("/api/auth/token",
                    data={"username": username, "password": "pw_test_123"})
    assert r.status_code == 200, f"login failed: {r.text}"
    return r.json()["access_token"]


def test_protected_endpoint_requires_token_in_jwt_mode(jwt_client):
    r = jwt_client.get("/api/emergency/status")
    assert r.status_code == 401
    assert "Missing" in r.json()["detail"] or "token" in r.json()["detail"].lower()


def test_protected_endpoint_rejects_garbage_token(jwt_client):
    r = jwt_client.get("/api/emergency/status",
                       headers={"Authorization": "Bearer junk.invalid.token"})
    assert r.status_code == 401
    assert "Invalid" in r.json()["detail"] or "expired" in r.json()["detail"].lower()


def test_protected_endpoint_accepts_valid_token(jwt_client, random_username):
    token = _register_and_login(jwt_client, random_username)
    r = jwt_client.get("/api/emergency/status",
                       headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["paper_trading"] is True


def test_register_rejects_duplicate_username(jwt_client, random_username):
    r1 = jwt_client.post("/api/auth/register",
                         json={"username": random_username, "password": "x" * 12})
    assert r1.status_code == 200
    r2 = jwt_client.post("/api/auth/register",
                         json={"username": random_username, "password": "y" * 12})
    assert r2.status_code == 400


def test_login_rejects_wrong_password(jwt_client, random_username):
    jwt_client.post("/api/auth/register",
                    json={"username": random_username, "password": "correct_pw_123"})
    r = jwt_client.post("/api/auth/token",
                        data={"username": random_username, "password": "WRONG_pw"})
    assert r.status_code != 200


def test_strategies_endpoint_is_public(jwt_client):
    """Strategy LISTING is public (marketplace browse). Backtest could be gated later."""
    r = jwt_client.get("/strategies")
    assert r.status_code == 200
    assert isinstance(r.json().get("strategies"), list)


def test_bypass_mode_allows_unauthenticated_access(bypass_client):
    """In bypass mode, protected endpoints return 200 without a token."""
    r = bypass_client.get("/api/emergency/status")
    assert r.status_code == 200
    assert r.json()["paper_trading"] is True
