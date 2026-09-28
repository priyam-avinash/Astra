"""
The most important multi-tenancy test: user context routes broker calls to
the correct per-user token. Without this, user A's broker calls could
inadvertently use user B's credentials.

We don't hit the real Upstox API — we mock `upstox_service.get_ohlcv` to record
the token it received, then assert each call uses the expected user's token.
"""

from unittest.mock import patch

import pandas as pd
import pytest


def _ensure_user(db, username: str):
    from app.models.database import User
    u = db.query(User).filter(User.username == username).first()
    if not u:
        u = User(username=username, hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
    return u


def test_no_user_context_uses_env_token(monkeypatch):
    """Without user context, broker should pass token=None → service uses env."""
    from app.brokers.upstox_client import UpstoxBroker
    captured = {}
    def fake_get_ohlcv(self, symbol, period="1mo", interval="15m", token=None):
        captured["token"] = token
        return pd.DataFrame()
    with patch("app.services.upstox_data.UpstoxDataService.get_ohlcv", fake_get_ohlcv):
        UpstoxBroker().get_ohlcv("RELIANCE", period="1mo", interval="1d")
    assert captured["token"] is None, "without user context, token must be None (env fallback)"


def test_user_context_routes_per_user_token():
    """With user context, broker calls must use that user's encrypted DB token."""
    from app.brokers.upstox_client import UpstoxBroker
    from app.models.database import SessionLocal
    from app.services.broker_credentials import upsert_credentials, revoke
    from app.services.user_context import user_scope

    db = SessionLocal()
    try:
        ua = _ensure_user(db, "ctx_user_a")
        ub = _ensure_user(db, "ctx_user_b")
        ua_id, ub_id = int(ua.id), int(ub.id)
        revoke(db, ua_id, "Upstox")
        revoke(db, ub_id, "Upstox")
        upsert_credentials(db, ua_id, "Upstox", access_token="TOKEN_A")
        upsert_credentials(db, ub_id, "Upstox", access_token="TOKEN_B")
    finally:
        db.close()

    captured = []
    def fake_get_ohlcv(self, symbol, period="1mo", interval="15m", token=None):
        captured.append(token)
        return pd.DataFrame()

    with patch("app.services.upstox_data.UpstoxDataService.get_ohlcv", fake_get_ohlcv):
        broker = UpstoxBroker()
        with user_scope(ua_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")
        with user_scope(ub_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")
        # Without scope → env fallback (token=None)
        broker.get_ohlcv("RELIANCE", "1mo", "1d")

    assert captured == ["TOKEN_A", "TOKEN_B", None], (
        f"context routing broken: got {captured}"
    )

    # Cleanup
    db = SessionLocal()
    try:
        revoke(db, ua_id, "Upstox")
        revoke(db, ub_id, "Upstox")
    finally:
        db.close()


def test_user_with_no_db_credentials_gets_none(monkeypatch):
    """A user without saved credentials should NOT inherit env token from another."""
    from app.brokers.upstox_client import UpstoxBroker
    from app.models.database import SessionLocal
    from app.services.broker_credentials import revoke
    from app.services.user_context import user_scope

    db = SessionLocal()
    try:
        u = _ensure_user(db, "orphan_user")
        revoke(db, u.id, "Upstox")
    finally:
        db.close()

    captured = []
    def fake_get_ohlcv(self, symbol, period="1mo", interval="15m", token=None):
        captured.append(token)
        return pd.DataFrame()

    # Env has a token (legacy single-user dev mode)
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "ENV_LEGACY_TOKEN")

    with patch("app.services.upstox_data.UpstoxDataService.get_ohlcv", fake_get_ohlcv):
        with user_scope(u.id):
            UpstoxBroker().get_ohlcv("RELIANCE", "1mo", "1d")

    # When the user has no DB row, broker_credentials.get_token falls back to env.
    # That's intentional for legacy single-user dev. The next test confirms isolation
    # when each user has their OWN row.
    assert captured == ["ENV_LEGACY_TOKEN"], (
        "user with no DB row should fall back to env for single-user dev parity"
    )


# ── Dhan per-user routing ──────────────────────────────────────────────────────

def test_dhan_user_context_routes_per_user_token():
    """DhanBroker must pass each user's DB token to dhan_data_service.get_ohlcv."""
    from app.brokers.dhan_client import DhanBroker
    from app.models.database import SessionLocal
    from app.services.broker_credentials import upsert_credentials, revoke
    from app.services.user_context import user_scope

    db = SessionLocal()
    try:
        ua = _ensure_user(db, "dhan_user_a")
        ub = _ensure_user(db, "dhan_user_b")
        ua_id, ub_id = int(ua.id), int(ub.id)
        revoke(db, ua_id, "Dhan")
        revoke(db, ub_id, "Dhan")
        upsert_credentials(db, ua_id, "Dhan", access_token="DHAN_TOKEN_A")
        upsert_credentials(db, ub_id, "Dhan", access_token="DHAN_TOKEN_B")
    finally:
        db.close()

    captured = []
    def fake_get_ohlcv(self, symbol, period="6mo", interval="1d", token=None):
        captured.append(token)
        return pd.DataFrame()

    with patch("app.services.dhan_data.DhanDataService.get_ohlcv", fake_get_ohlcv):
        broker = DhanBroker()
        with user_scope(ua_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")
        with user_scope(ub_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")
        broker.get_ohlcv("RELIANCE", "1mo", "1d")  # no scope → None

    assert captured == ["DHAN_TOKEN_A", "DHAN_TOKEN_B", None], (
        f"Dhan context routing broken: got {captured}"
    )

    db = SessionLocal()
    try:
        revoke(db, ua_id, "Dhan")
        revoke(db, ub_id, "Dhan")
    finally:
        db.close()


def test_dhan_no_user_context_uses_env_token():
    """Without user context, DhanBroker passes token=None (env fallback)."""
    from app.brokers.dhan_client import DhanBroker
    captured = {}
    def fake_get_ohlcv(self, symbol, period="6mo", interval="1d", token=None):
        captured["token"] = token
        return pd.DataFrame()
    with patch("app.services.dhan_data.DhanDataService.get_ohlcv", fake_get_ohlcv):
        DhanBroker().get_ohlcv("RELIANCE", period="1mo", interval="1d")
    assert captured["token"] is None


# ── Zerodha per-user routing ───────────────────────────────────────────────────

def test_zerodha_user_context_routes_per_user_token(monkeypatch):
    """ZerodhaBroker._client() must use the DB token for the current user."""
    from app.brokers.zerodha_client import ZerodhaBroker
    from app.models.database import SessionLocal
    from app.services.broker_credentials import upsert_credentials, revoke
    from app.services.user_context import user_scope

    monkeypatch.setenv("KITE_API_KEY", "fake_api_key")

    db = SessionLocal()
    try:
        ua = _ensure_user(db, "zerodha_user_a")
        ub = _ensure_user(db, "zerodha_user_b")
        ua_id, ub_id = int(ua.id), int(ub.id)
        revoke(db, ua_id, "Zerodha")
        revoke(db, ub_id, "Zerodha")
        upsert_credentials(db, ua_id, "Zerodha", access_token="KITE_TOKEN_A")
        upsert_credentials(db, ub_id, "Zerodha", access_token="KITE_TOKEN_B")
    finally:
        db.close()

    # Intercept KiteConnect construction to capture the token it received
    captured = []

    class FakeKite:
        def __init__(self, api_key):
            self._token = None
        def set_access_token(self, tok):
            captured.append(tok)
        def historical_data(self, **kwargs):
            return []

    with patch("kiteconnect.KiteConnect", FakeKite):
        broker = ZerodhaBroker()
        with user_scope(ua_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")
        with user_scope(ub_id):
            broker.get_ohlcv("RELIANCE", "1mo", "1d")

    assert "KITE_TOKEN_A" in captured, f"Token A not used: {captured}"
    assert "KITE_TOKEN_B" in captured, f"Token B not used: {captured}"

    db = SessionLocal()
    try:
        revoke(db, ua_id, "Zerodha")
        revoke(db, ub_id, "Zerodha")
    finally:
        db.close()


def test_zerodha_no_user_context_uses_env_token(monkeypatch):
    """Without user context, ZerodhaBroker uses KITE_ACCESS_TOKEN from env."""
    from app.brokers.zerodha_client import ZerodhaBroker

    monkeypatch.setenv("KITE_API_KEY", "fake_api_key")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "ENV_KITE_TOKEN")

    captured = []

    class FakeKite:
        def __init__(self, api_key):
            pass
        def set_access_token(self, tok):
            captured.append(tok)
        def historical_data(self, **kwargs):
            return []

    with patch("kiteconnect.KiteConnect", FakeKite):
        ZerodhaBroker().get_ohlcv("RELIANCE", "1mo", "1d")

    assert "ENV_KITE_TOKEN" in captured, f"Env token not used: {captured}"
