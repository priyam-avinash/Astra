"""
Upstox OAuth routes
====================
GET /upstox/login    → redirects browser to Upstox login (legacy single-user
                       dev entry; prefer /brokers/Upstox/login-url)
GET /upstox/callback → exchanges auth code for access token
                       * `state` present: must be a valid signed state AND
                         (outside AUTH_MODE=bypass) match the browser-binding
                         cookie set by /brokers/{name}/login-url → token is
                         stored for that user (BrokerCredential, encrypted).
                       * no `state`: only allowed locally (not serverless)
                         with AUTH_MODE=bypass → token goes to os.environ +
                         backend/.env (legacy single-user dev).
                       * anything else → 400, nothing is exchanged or written.
GET /upstox/status   → JSON: is a token present / valid for the current user?

Handlers are plain `def`: they make blocking HTTP calls (token exchange,
profile probe), so FastAPI runs them in its threadpool.
"""

import html
import logging
import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session

from app.api.endpoints import get_current_user
from app.models.database import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upstox", tags=["upstox"])

_BROKER = "Upstox"


def _auth_mode() -> str:
    return (os.getenv("AUTH_MODE", "jwt") or "jwt").strip().lower()


def _global_token_write_allowed() -> bool:
    """The process-wide token (os.environ + .env) is shared by every user, so
    it may only be written on a local single-user install."""
    from app.core import config
    return (not config.SERVERLESS) and _auth_mode() == "bypass"


@router.get("/login")
def upstox_login(current_user: User = Depends(get_current_user)):
    """Browser redirect to Upstox OAuth (legacy entry — prefer /brokers/Upstox/login-url)."""
    from app.services.upstox_data import upstox_service
    return RedirectResponse(url=upstox_service.get_login_url())


@router.get("/callback")
def upstox_callback(request: Request, code: str = "", state: str = "",
                    db: Session = Depends(get_db)):
    """OAuth callback. State/binding is verified BEFORE the code is exchanged."""
    from app.services.upstox_data import upstox_service
    from app.services.oauth_state import verify_state
    from app.services.broker_credentials import (
        oauth_binding_cookie_name, upsert_credentials, verify_oauth_binding,
    )

    cookie_name = oauth_binding_cookie_name(_BROKER)

    def page(title: str, body_html: str, status: int = 200) -> HTMLResponse:
        resp = HTMLResponse(_html_page(title, body_html), status_code=status)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.delete_cookie(cookie_name, path="/")
        return resp

    if not code:
        return page("Error", "No authorization code received. Start again from Settings → Brokers.", 400)

    user_id = None
    if state:
        user_id = verify_state(state, _BROKER)
        if user_id is None:
            return page("Error", "Invalid or expired login request. Start again from Settings → Brokers.", 400)
        if _auth_mode() != "bypass" and not verify_oauth_binding(request.cookies.get(cookie_name), state):
            logger.warning("Upstox callback: state not bound to this browser (user_id=%s)", user_id)
            return page("Error", "This login was started in a different browser or has expired. "
                                 "Start again from Settings → Brokers.", 400)
    elif not _global_token_write_allowed():
        return page("Error", "Missing login state. Start the Upstox connection from Settings → Brokers.", 400)

    try:
        token = upstox_service.exchange_code(code)
    except Exception as e:
        logger.warning("Upstox token exchange failed: %s: %s", type(e).__name__, e)
        return page("Token Exchange Failed",
                    f"Upstox did not issue a token ({html.escape(type(e).__name__)}). "
                    "The code may have expired — please try again.", 502)

    if user_id is not None:
        try:
            upsert_credentials(db, user_id=user_id, broker=_BROKER, access_token=token)
        except Exception:
            logger.exception("Failed to persist Upstox token for user %s", user_id)
            return page("Storage Failed", "The token could not be saved. Please try again.", 500)
        where = "to your account"
    else:
        upstox_service.persist_global_token(token)
        where = "for this local install"

    return page(
        "Upstox Connected",
        f"Access token obtained and saved {html.escape(where)}.<br><br>"
        "You can close this tab. ASTRA will now use Upstox for intraday data.<br><br>"
        "<strong>Token refreshes daily — reconnect each trading morning.</strong>",
    )


@router.get("/status")
def upstox_status(current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    from app.services.upstox_data import upstox_service
    from app.services.broker_credentials import get_token
    token = get_token(db, current_user.id, _BROKER)     # per-user row, else env (legacy)
    token_present = upstox_service.is_available(token=token or "")
    token_valid   = upstox_service.is_token_valid(token=token) if token_present else False
    return JSONResponse({
        "available":      token_valid,          # True only if live API probe passes
        "token_present":  token_present,        # True if a token string exists (may be expired)
        "token_valid":    token_valid,
        "login_url": "/upstox/login",
        "note": (
            "Token is live and valid."
            if token_valid else
            "Token missing or expired — reconnect Upstox from Settings → Brokers (required daily)."
        ),
    })


def _html_page(title: str, body_html: str) -> str:
    """`title` is escaped here; `body_html` is trusted markup — callers must
    html.escape() anything dynamic they interpolate into it."""
    t = html.escape(title)
    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>{t}</title>
    <style>body{{font-family:sans-serif;max-width:600px;margin:80px auto;padding:20px;
    background:#0f0f0f;color:#e0e0e0;}}h1{{color:#00e5ff;}}p{{line-height:1.6;}}</style>
    </head><body><h1>{t}</h1><p>{body_html}</p></body></html>
    """
