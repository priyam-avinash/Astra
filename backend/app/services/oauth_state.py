"""
Signed OAuth `state` parameter — embeds the initiating user_id into the
broker OAuth flow so that on callback (browser redirect with no auth context)
we can attribute the new token to the correct user.

Uses the JWT machinery already present in app.services.auth.
Token expires in 10 minutes (OAuth round-trips are fast).
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

from app.services.auth import SECRET_KEY, ALGORITHM


_OAUTH_STATE_TTL_SECONDS = 600   # 10 min — plenty for an OAuth round trip
_OAUTH_PURPOSE_TAG = "astra-oauth"


def sign_state(user_id: int, broker: str) -> str:
    """Return a JWT-signed state string identifying the user + broker."""
    payload = {
        "sub":     str(user_id),
        "broker":  broker.strip().lower(),
        "purpose": _OAUTH_PURPOSE_TAG,
        "exp":     datetime.utcnow() + timedelta(seconds=_OAUTH_STATE_TTL_SECONDS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_state(state: str, expected_broker: str) -> Optional[int]:
    """
    Validate a state token. Returns user_id on success, None on any failure
    (expired, tampered, wrong broker, wrong purpose).
    """
    if not state:
        return None
    try:
        payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    if payload.get("purpose") != _OAUTH_PURPOSE_TAG:
        return None
    if payload.get("broker") != expected_broker.strip().lower():
        return None
    sub = payload.get("sub")
    try:
        return int(sub) if sub is not None else None
    except (TypeError, ValueError):
        return None
