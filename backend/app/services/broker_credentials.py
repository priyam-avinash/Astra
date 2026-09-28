"""
ASTRA Broker Credentials Service
==================================
Per-user broker token storage with Fernet encryption at rest.

Each (user_id, broker) tuple stores up to: access_token, refresh_token,
client_id, api_key, api_secret, expires_at. All sensitive strings are
encrypted before hitting the DB.

Used by:
  - /upstox/callback (and future /dhan/login, /zerodha/login) to persist
    the OAuth token to the current user
  - BrokerClient adapters to retrieve a user's tokens when serving requests

Design choice — fallback to env vars:
  In single-user dev mode, broker credentials may still live in .env. The
  resolver `get_user_token(user_id, broker)` first checks the DB and falls
  back to env. This lets the current dev setup keep working unchanged while
  multi-user SaaS reads from per-user rows.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.database import BrokerCredential
from app.services.crypto import decrypt, encrypt

logger = logging.getLogger(__name__)


# Map a broker name → env var that holds the access token (legacy single-user mode)
_ENV_TOKEN_FALLBACK = {
    "upstox":  "UPSTOX_ACCESS_TOKEN",
    "dhan":    "DHAN_ACCESS_TOKEN",
    "zerodha": "KITE_ACCESS_TOKEN",
}


def _normalize(broker: str) -> str:
    return broker.strip().lower()


def upsert_credentials(
    db: Session,
    user_id: int,
    broker: str,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    client_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> BrokerCredential:
    """Upsert per-user broker credentials. Encrypts before storing."""
    broker_lc = _normalize(broker)
    row = (
        db.query(BrokerCredential)
        .filter(BrokerCredential.user_id == user_id,
                BrokerCredential.broker == broker_lc)
        .first()
    )
    if row is None:
        row = BrokerCredential(user_id=user_id, broker=broker_lc)
        db.add(row)

    # Only overwrite fields we were given; preserve existing on None
    if access_token is not None:
        row.access_token_enc  = encrypt(access_token)
    if refresh_token is not None:
        row.refresh_token_enc = encrypt(refresh_token)
    if client_id is not None:
        row.client_id_enc     = encrypt(client_id)
    if api_key is not None:
        row.api_key_enc       = encrypt(api_key)
    if api_secret is not None:
        row.api_secret_enc    = encrypt(api_secret)
    if expires_at is not None:
        row.expires_at = expires_at

    db.commit()
    db.refresh(row)
    return row


def get_token(db: Session, user_id: int, broker: str) -> Optional[str]:
    """
    Return the access token for (user_id, broker).
    DB first, then env-var fallback for legacy single-user mode.
    Returns None if neither has a value.
    """
    broker_lc = _normalize(broker)
    row = (
        db.query(BrokerCredential)
        .filter(BrokerCredential.user_id == user_id,
                BrokerCredential.broker == broker_lc)
        .first()
    )
    if row and row.access_token_enc:
        token = decrypt(row.access_token_enc)
        if token:
            return token

    env_key = _ENV_TOKEN_FALLBACK.get(broker_lc)
    if env_key:
        env_val = os.getenv(env_key, "")
        if env_val and env_val.strip() not in ("", "None", "none"):
            return env_val
    return None


def list_user_brokers(db: Session, user_id: int) -> list[dict]:
    """Inspection helper — used by API to show which brokers this user has connected."""
    rows = db.query(BrokerCredential).filter(BrokerCredential.user_id == user_id).all()
    out = []
    for r in rows:
        out.append({
            "broker":      r.broker,
            "has_access":  bool(r.access_token_enc),
            "has_refresh": bool(r.refresh_token_enc),
            "expires_at":  r.expires_at.isoformat() if r.expires_at else None,
            "updated_at":  r.updated_at.isoformat() if r.updated_at else None,
        })
    return out


def revoke(db: Session, user_id: int, broker: str) -> bool:
    """Delete a user's credentials for one broker. Returns True if removed."""
    broker_lc = _normalize(broker)
    row = (
        db.query(BrokerCredential)
        .filter(BrokerCredential.user_id == user_id,
                BrokerCredential.broker == broker_lc)
        .first()
    )
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True
