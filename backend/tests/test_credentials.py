"""
Tests for per-user encrypted broker credentials:
  - Fernet encrypt/decrypt round-trip
  - User A cannot see User B's credentials
  - OAuth state signing and verification
  - Env-var fallback when DB has no row
"""

import os

import pytest


# ── Crypto round-trip ──────────────────────────────────────────────────────

def test_crypto_round_trip():
    from app.services.crypto import encrypt, decrypt
    plain = "very_secret_access_token_xyz_123"
    cipher = encrypt(plain)
    assert cipher != plain
    assert decrypt(cipher) == plain


def test_crypto_decrypts_to_empty_on_invalid_token():
    from app.services.crypto import decrypt
    assert decrypt("garbage-not-a-real-fernet-token") == ""
    assert decrypt("") == ""


def test_crypto_handles_unicode():
    from app.services.crypto import encrypt, decrypt
    plain = "नमस्ते 🔐 token"
    assert decrypt(encrypt(plain)) == plain


# ── BrokerCredential service: data isolation ──────────────────────────────

def test_users_cannot_read_each_others_credentials():
    """Critical multi-tenancy test: user_a's token must NEVER be returned for user_b."""
    from app.models.database import SessionLocal, User
    from app.services.broker_credentials import upsert_credentials, get_token, revoke
    db = SessionLocal()
    try:
        # Use a fixed-prefix username to be idempotent across runs
        ua = db.query(User).filter(User.username == "isol_user_a").first() \
             or User(username="isol_user_a", hashed_password="x")
        ub = db.query(User).filter(User.username == "isol_user_b").first() \
             or User(username="isol_user_b", hashed_password="x")
        if ua.id is None:
            db.add(ua); db.commit(); db.refresh(ua)
        if ub.id is None:
            db.add(ub); db.commit(); db.refresh(ub)

        # Clean any prior state
        revoke(db, ua.id, "Upstox")
        revoke(db, ub.id, "Upstox")

        # Save different tokens
        upsert_credentials(db, ua.id, "Upstox", access_token="token_for_A")
        upsert_credentials(db, ub.id, "Upstox", access_token="token_for_B")

        assert get_token(db, ua.id, "Upstox") == "token_for_A"
        assert get_token(db, ub.id, "Upstox") == "token_for_B"

        # Cleanup
        revoke(db, ua.id, "Upstox")
        revoke(db, ub.id, "Upstox")
    finally:
        db.close()


def test_credentials_upsert_preserves_unspecified_fields():
    """Calling upsert with only access_token must not wipe refresh_token."""
    from app.models.database import SessionLocal, User
    from app.services.broker_credentials import upsert_credentials, revoke
    from app.services.crypto import decrypt
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "upsert_user").first() \
            or User(username="upsert_user", hashed_password="x")
        if u.id is None:
            db.add(u); db.commit(); db.refresh(u)
        revoke(db, u.id, "Upstox")

        # First write — both tokens
        upsert_credentials(db, u.id, "Upstox",
                           access_token="A1", refresh_token="R1")
        # Second write — only access_token; refresh should survive
        row = upsert_credentials(db, u.id, "Upstox", access_token="A2")
        assert decrypt(row.access_token_enc)  == "A2"
        assert decrypt(row.refresh_token_enc) == "R1"

        revoke(db, u.id, "Upstox")
    finally:
        db.close()


def test_get_token_falls_back_to_env_when_db_empty(monkeypatch):
    """If no DB row exists, the env-var fallback should serve the token."""
    from app.models.database import SessionLocal, User
    from app.services.broker_credentials import get_token, revoke
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == "fallback_user").first() \
            or User(username="fallback_user", hashed_password="x")
        if u.id is None:
            db.add(u); db.commit(); db.refresh(u)
        revoke(db, u.id, "Upstox")

        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "env_legacy_token_xyz")
        assert get_token(db, u.id, "Upstox") == "env_legacy_token_xyz"

        # And empty/placeholder env values do NOT count as available
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "")
        assert get_token(db, u.id, "Upstox") is None
    finally:
        db.close()


# ── OAuth state signing ────────────────────────────────────────────────────

def test_oauth_state_round_trip():
    from app.services.oauth_state import sign_state, verify_state
    s = sign_state(user_id=42, broker="Upstox")
    assert verify_state(s, "Upstox") == 42


def test_oauth_state_rejects_wrong_broker():
    from app.services.oauth_state import sign_state, verify_state
    s = sign_state(user_id=42, broker="Upstox")
    assert verify_state(s, "Dhan") is None


def test_oauth_state_rejects_garbage():
    from app.services.oauth_state import verify_state
    assert verify_state("not.a.real.jwt", "Upstox") is None
    assert verify_state("", "Upstox") is None


def test_oauth_state_rejects_state_from_other_purpose():
    """A regular auth JWT should NOT be usable as an OAuth state."""
    from app.services.auth import create_access_token
    from app.services.oauth_state import verify_state
    plain_jwt = create_access_token({"sub": "42"})
    assert verify_state(plain_jwt, "Upstox") is None
