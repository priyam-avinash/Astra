"""
Brokers API — list broker plugins, hand out per-user OAuth URLs,
inspect which brokers the current user has connected.

The `/login-url` endpoint requires auth so we can embed the user_id into the
OAuth `state` parameter. The callback uses that state to attribute the new
token to the correct user.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import User, get_db
from app.api.endpoints import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brokers", tags=["brokers"])


@router.get("")
async def list_all_brokers():
    """List all registered broker plugins + their availability status (public)."""
    from app.brokers.registry import list_brokers
    return {"brokers": list_brokers()}


@router.get("/{name}/login-url")
async def broker_login_url(name: str, current_user: User = Depends(get_current_user)):
    """
    Return the OAuth login URL for `name` with a signed `state` parameter
    encoding the current user. The callback uses this to attribute the token.
    """
    from app.brokers.registry import get_broker
    from app.services.oauth_state import sign_state

    b = get_broker(name)
    if b is None:
        raise HTTPException(status_code=404, detail=f"Broker '{name}' not registered")

    base_url = b.get_login_url()
    if not base_url:
        return {"broker": b.name, "login_url": None,
                "reason": f"{b.name} does not expose an OAuth flow yet"}

    state = sign_state(current_user.id, b.name)
    sep = "&" if "?" in base_url else "?"
    return {
        "broker":    b.name,
        "login_url": f"{base_url}{sep}state={state}",
        "user_id":   current_user.id,
    }


@router.get("/connected")
async def list_connected_brokers(current_user: User = Depends(get_current_user),
                                 db: Session = Depends(get_db)):
    """Return per-broker connection status + token expiry for the current user."""
    from app.services.broker_credentials import list_user_brokers
    return {"connected": list_user_brokers(db, current_user.id)}


@router.delete("/{name}")
async def revoke_broker(name: str,
                        current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Disconnect a broker for the current user — wipes their stored tokens."""
    from app.services.broker_credentials import revoke
    removed = revoke(db, current_user.id, name)
    return {"broker": name, "removed": removed}
