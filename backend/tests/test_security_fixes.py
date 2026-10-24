"""
Regression tests for the v1.14 security review:
  1. provider error text never carries API keys
  2. Upstox OAuth callback refuses missing/invalid/unbound state outside local
     bypass dev, and never writes the global token in that case
  4. WebSockets require a token (jwt mode) and an allowed Origin
  5. credential encryption keeps decrypting legacy ciphertext
  6. OAuth state tokens are never accepted as access tokens
  8. intervals can't inject path components into the disk cache
"""
import base64
import hashlib
import os

import pytest
from jose import JWTError
from starlette.websockets import WebSocketDisconnect


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_user(username: str) -> int:
    from app.models.database import SessionLocal, User
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).first()
        if u is None:
            u = User(username=username, hashed_password="x")
            db.add(u)
            db.commit()
            db.refresh(u)
        return u.id
    finally:
        db.close()


# ── 1. API key scrubbing ─────────────────────────────────────────────────────

def test_scrub_secrets_masks_query_params():
    from app.services.market_data import scrub_secrets
    s = ("HTTPSConnectionPool: Max retries exceeded with url: /time_series?symbol=AAPL"
         "&apikey=SECRET123&interval=1day and ?api_key=AK&token=TT&access_token=AT&key=KK")
    out = scrub_secrets(s)
    for secret in ("SECRET123", "=AK", "=TT", "=AT", "=KK"):
        assert secret not in out
    assert "apikey=***" in out and "symbol=AAPL" in out and "interval=1day" in out
    assert scrub_secrets(None) == ""


def test_provider_error_and_health_never_store_key(monkeypatch):
    import app.services.market_data as md

    class Boom:
        def get(self, url, params=None, headers=None, timeout=None):
            raise ConnectionError(f"Max retries exceeded with url: {url}?apikey=TOPSECRET&symbol=X")

    monkeypatch.setattr(md, "_session", lambda kind="browser": Boom())
    with pytest.raises(md.ProviderError) as e:
        md._http_get("https://api.example.com/q", params={"apikey": "TOPSECRET"})
    assert "TOPSECRET" not in e.value.message and "TOPSECRET" not in str(e.value)

    # Even a raw message handed to ProviderError / record_fail is scrubbed
    err = md.ProviderError("network", "GET https://x/y?apikey=LEAKY failed")
    assert "LEAKY" not in err.message
    md.HEALTH.reset()
    md.HEALTH.record_fail("twelve_data", err, "crypto")
    snap = repr(md.HEALTH.snapshot())
    assert "twelve_data" in snap and "LEAKY" not in snap
    md.HEALTH.reset()


def test_twelve_data_key_sent_as_header_not_query(monkeypatch):
    import app.services.market_data as md
    seen = {}

    class R:
        status_code = 200
        def json(self):
            return {"status": "error", "code": 400, "message": "nope"}

    def fake_get(url, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(md, "get_secret", lambda n: "KEY123" if n == "twelve_data_key" else None)
    monkeypatch.setattr(md, "_http_get", fake_get)
    with pytest.raises(md.ProviderError):
        md._twelve_data_ohlcv("AAPL", 31, "1d")
    assert "apikey" not in (seen.get("params") or {})
    assert seen["headers"]["Authorization"] == "apikey KEY123"


# ── 2. Upstox OAuth callback ─────────────────────────────────────────────────

@pytest.fixture()
def upstox_spy(monkeypatch):
    """Record token exchanges / global writes instead of hitting Upstox."""
    from app.services.upstox_data import upstox_service
    calls = {"exchange": 0, "global": 0}

    def fake_exchange(code):
        calls["exchange"] += 1
        return "tok_" + code

    def fake_global(token):
        calls["global"] += 1

    monkeypatch.setattr(upstox_service, "exchange_code", fake_exchange)
    monkeypatch.setattr(upstox_service, "persist_global_token", fake_global)
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "")
    return calls


def test_callback_without_state_in_jwt_mode_is_400_and_writes_nothing(jwt_client, upstox_spy):
    r = jwt_client.get("/upstox/callback", params={"code": "abc"})
    assert r.status_code == 400
    assert upstox_spy == {"exchange": 0, "global": 0}
    assert os.environ.get("UPSTOX_ACCESS_TOKEN", "") == ""


def test_callback_with_forged_state_is_400(jwt_client, upstox_spy):
    r = jwt_client.get("/upstox/callback", params={"code": "abc", "state": "forged.state.value"})
    assert r.status_code == 400
    assert upstox_spy == {"exchange": 0, "global": 0}


def test_callback_with_valid_state_but_no_browser_binding_is_400(jwt_client, upstox_spy):
    from app.services.oauth_state import sign_state
    state = sign_state(_make_user("sec_oauth_victim"), "Upstox")
    r = jwt_client.get("/upstox/callback", params={"code": "abc", "state": state})
    assert r.status_code == 400
    assert upstox_spy == {"exchange": 0, "global": 0}


def test_callback_with_bound_state_persists_per_user_only(jwt_client, upstox_spy):
    from app.models.database import SessionLocal
    from app.services.broker_credentials import (
        get_token, oauth_binding_cookie_name, oauth_binding_value, revoke,
    )
    from app.services.oauth_state import sign_state
    uid = _make_user("sec_oauth_owner")
    state = sign_state(uid, "Upstox")
    jwt_client.cookies.set(oauth_binding_cookie_name("Upstox"), oauth_binding_value(state))
    r = jwt_client.get("/upstox/callback", params={"code": "xyz<script>", "state": state})
    assert r.status_code == 200, r.text
    assert "<script>" not in r.text
    assert upstox_spy == {"exchange": 1, "global": 0}
    assert os.environ.get("UPSTOX_ACCESS_TOKEN", "") == ""
    db = SessionLocal()
    try:
        assert get_token(db, uid, "Upstox") == "tok_xyz<script>"
        revoke(db, uid, "Upstox")
    finally:
        db.close()


def test_login_url_sets_binding_cookie(jwt_client, random_username, monkeypatch):
    from app.brokers.registry import get_broker
    b = get_broker("Upstox")
    monkeypatch.setattr(type(b), "get_login_url", lambda self: "https://upstox.example/auth?x=1")
    jwt_client.post("/api/auth/register", json={"username": random_username, "password": "pw_test_123"})
    tok = jwt_client.post("/api/auth/token",
                          data={"username": random_username, "password": "pw_test_123"}).json()["access_token"]
    r = jwt_client.get("/brokers/Upstox/login-url", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    assert "astra_oauth_upstox" in r.headers.get("set-cookie", "")
    assert "httponly" in r.headers["set-cookie"].lower()


def test_callback_without_state_local_bypass_uses_global_path(bypass_client, upstox_spy, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "SERVERLESS", False)
    r = bypass_client.get("/upstox/callback", params={"code": "abc"})
    assert r.status_code == 200
    assert upstox_spy == {"exchange": 1, "global": 1}


def test_callback_without_state_serverless_bypass_is_400(bypass_client, upstox_spy, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "SERVERLESS", True)
    r = bypass_client.get("/upstox/callback", params={"code": "abc"})
    assert r.status_code == 400
    assert upstox_spy == {"exchange": 0, "global": 0}


def test_upstox_status_and_brokers_require_login_in_jwt_mode(jwt_client):
    assert jwt_client.get("/upstox/status").status_code == 401
    assert jwt_client.get("/upstox/login", follow_redirects=False).status_code == 401
    assert jwt_client.get("/brokers").status_code == 401


# ── 4. WebSockets ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/ws/prices", "/ws/market-data"])
def test_ws_without_token_rejected_in_jwt_mode(jwt_client, path):
    with pytest.raises(WebSocketDisconnect):
        with jwt_client.websocket_connect(path):
            pass


def test_ws_with_state_token_rejected(jwt_client):
    from app.services.oauth_state import sign_state
    state = sign_state(_make_user("sec_ws_user"), "Upstox")
    with pytest.raises(WebSocketDisconnect):
        with jwt_client.websocket_connect(f"/ws/prices?token={state}"):
            pass


def test_ws_bad_origin_rejected_even_in_bypass(bypass_client):
    with pytest.raises(WebSocketDisconnect):
        with bypass_client.websocket_connect("/ws/prices", headers={"origin": "https://evil.example"}):
            pass


class _FakeWS:
    def __init__(self, query=None, headers=None):
        self.query_params = query or {}
        self.headers = headers or {}

        class U:
            path = "/ws/prices"
        self.url = U()


def test_ws_auth_helper_accepts_valid_access_token(monkeypatch):
    import asyncio
    from app.api.websockets import _authenticate_ws
    from app.services.auth import create_access_token
    monkeypatch.setenv("AUTH_MODE", "jwt")
    _make_user("sec_ws_ok")
    tok = create_access_token({"sub": "sec_ws_ok"})
    ws = _FakeWS({"token": tok}, {"origin": "http://localhost:5173"})
    assert asyncio.run(_authenticate_ws(ws)) == "sec_ws_ok"
    assert asyncio.run(_authenticate_ws(_FakeWS({"token": tok}, {"origin": "https://evil.example"}))) is None
    assert asyncio.run(_authenticate_ws(_FakeWS({}, {}))) is None


# ── 5. credential encryption key ─────────────────────────────────────────────

def test_crypto_decrypts_legacy_ciphertext():
    from cryptography.fernet import Fernet
    from app.services.crypto import decrypt, encrypt
    seed = os.getenv("JWT_SECRET_KEY", "dev_fallback_seed_change_me")
    legacy = Fernet(base64.urlsafe_b64encode(hashlib.sha256(seed.encode() + b"::astra-cred-key").digest()))
    old = legacy.encrypt(b"old_token").decode()
    assert decrypt(old) == "old_token"
    assert decrypt(encrypt("new_token")) == "new_token"


def test_crypto_primary_key_not_public_seed(monkeypatch):
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    from app.services import crypto
    from app.services.auth import SECRET_KEY
    expected = base64.urlsafe_b64encode(hashlib.sha256(SECRET_KEY.encode() + b"::astra-cred-key").digest())
    assert crypto._resolve_key() == expected


# ── 6. JWT hardening ─────────────────────────────────────────────────────────

def test_access_token_has_typ_and_hs256():
    from jose import jwt
    from app.services.auth import ALGORITHM, SECRET_KEY, create_access_token, decode_access_token
    assert ALGORITHM == "HS256"
    tok = create_access_token({"sub": "alice", "purpose": "astra-oauth"})
    assert jwt.get_unverified_header(tok)["alg"] == "HS256"
    claims = jwt.decode(tok, SECRET_KEY, algorithms=["HS256"])
    assert claims["typ"] == "access" and "purpose" not in claims
    assert decode_access_token(tok)["sub"] == "alice"


def test_oauth_state_token_rejected_as_access_token():
    from app.services.auth import decode_access_token, is_access_payload
    from app.services.oauth_state import sign_state
    state = sign_state(1, "Upstox")
    with pytest.raises(JWTError):
        decode_access_token(state)
    with pytest.raises(JWTError):
        decode_access_token(state, allow_legacy=False)
    assert not is_access_payload({"sub": "1", "purpose": "astra-oauth"})
    assert not is_access_payload({"sub": "1", "typ": "refresh"})


def test_legacy_access_token_transition():
    from datetime import datetime, timedelta
    from jose import jwt
    from app.services.auth import SECRET_KEY, decode_access_token
    legacy = jwt.encode({"sub": "bob", "exp": datetime.utcnow() + timedelta(minutes=5)},
                        SECRET_KEY, algorithm="HS256")
    assert decode_access_token(legacy)["sub"] == "bob"
    with pytest.raises(JWTError):
        decode_access_token(legacy, allow_legacy=False)


# ── 8. interval whitelist ────────────────────────────────────────────────────

def test_interval_cannot_reach_filesystem_path():
    import app.services.market_data as md
    with pytest.raises(ValueError):
        md._disk_path("RELIANCE.NS", "../../etc/passwd")
    assert md._disk_get("RELIANCE.NS", "1d/../../x")[0].empty
    assert md.market_data.get_ohlcv("RELIANCE", period="1mo", interval="../../x").empty
    assert md._disk_path("RELIANCE.NS", "1d").parent == md.DISK_CACHE_DIR
