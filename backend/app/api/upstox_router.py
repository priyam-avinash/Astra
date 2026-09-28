"""
Upstox OAuth routes
====================
GET /upstox/login    → redirects browser to Upstox login (legacy single-user)
GET /upstox/callback → exchanges auth code for access token
                       If `state` is a valid signed JWT → persist token to the
                       user_id encoded in state (multi-tenant).
                       Otherwise → persist to .env (legacy single-user dev).
GET /upstox/status   → JSON: is token present and service available?
"""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from app.models.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upstox", tags=["upstox"])


@router.get("/login")
async def upstox_login():
    """Browser redirect to Upstox OAuth (legacy entry — prefer /brokers/Upstox/login-url)."""
    from app.services.upstox_data import upstox_service
    return RedirectResponse(url=upstox_service.get_login_url())


@router.get("/callback")
async def upstox_callback(code: str = "", state: str = "", db: Session = Depends(get_db)):
    """
    OAuth callback.

    Flow:
      1. Exchange `code` for access token via Upstox API
      2. If `state` is a valid signed JWT → store token in BrokerCredential
         for that user (multi-tenant SaaS path)
      3. Else → store token in .env (single-user dev path, preserves legacy)
    """
    from app.services.upstox_data import upstox_service
    from app.services.oauth_state import verify_state
    from app.services.broker_credentials import upsert_credentials

    if not code:
        return HTMLResponse(
            _html_page("❌ Error", "No authorization code received. Please try /upstox/login again."),
            status_code=400,
        )

    try:
        token = upstox_service.exchange_code(code)
    except Exception as e:
        return HTMLResponse(_html_page("❌ Token Exchange Failed", str(e)), status_code=500)

    # If state is present and verifies, persist per-user instead of env
    persisted_path = "env"
    user_id = verify_state(state, "Upstox") if state else None
    if user_id is not None:
        try:
            upsert_credentials(db, user_id=user_id, broker="Upstox",
                               access_token=token)
            persisted_path = f"user_id={user_id}"
        except Exception as e:
            logger.exception(f"Failed to persist Upstox token for user {user_id}")
            return HTMLResponse(_html_page("❌ Storage Failed", str(e)), status_code=500)

    return HTMLResponse(_html_page(
        "✅ Upstox Connected",
        f"Access token obtained and saved ({persisted_path}).<br><br>"
        "You can close this tab. ASTRA will now use Upstox for intraday data.<br><br>"
        "<strong>Token refreshes daily — reconnect each trading morning.</strong>"
    ))


@router.get("/status")
async def upstox_status():
    from app.services.upstox_data import upstox_service
    token_present = upstox_service.is_available()
    token_valid   = upstox_service.is_token_valid() if token_present else False
    return JSONResponse({
        "available":      token_valid,          # True only if live API probe passes
        "token_present":  token_present,        # True if string exists in env (may be expired)
        "token_valid":    token_valid,
        "login_url": "/upstox/login",
        "note": (
            "Token is live and valid."
            if token_valid else
            "Token missing or expired — visit /upstox/login to refresh (required daily)."
        ),
    })


def _html_page(title: str, body: str) -> str:
    return f"""
    <!DOCTYPE html><html><head><title>{title}</title>
    <style>body{{font-family:sans-serif;max-width:600px;margin:80px auto;padding:20px;
    background:#0f0f0f;color:#e0e0e0;}}h1{{color:#00e5ff;}}p{{line-height:1.6;}}</style>
    </head><body><h1>{title}</h1><p>{body}</p></body></html>
    """
