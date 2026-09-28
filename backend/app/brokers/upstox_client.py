"""
Upstox v2 broker plugin.

Behavior depends on whether a current_user_id is set in the request context:
  - With user context: looks up the user's encrypted token from BrokerCredential
    and uses it for this call. Other users' tokens are never touched.
  - Without user context (backtest scripts, etc.): falls back to env-var token.

This is the bridge between the per-user credential store and the actual
broker singleton (which holds the API key + base URLs).
"""

import logging
from typing import Optional

import pandas as pd

from app.brokers.base import BrokerClient

logger = logging.getLogger(__name__)


def _resolve_user_token() -> Optional[str]:
    """
    Return the access token for the current request's user, or None.
    Returns None on:
      - no user in context (script / unauthenticated call)
      - no DB row for this user + broker (then caller falls through to env)
    """
    try:
        from app.services.user_context import get_current_user_id
        user_id = get_current_user_id()
        if user_id is None:
            return None
        # Open a short-lived session for the token lookup
        from app.models.database import SessionLocal
        from app.services.broker_credentials import get_token
        db = SessionLocal()
        try:
            return get_token(db, user_id, "Upstox")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"UpstoxBroker user-token resolution failed: {e}")
        return None


class UpstoxBroker(BrokerClient):
    @property
    def name(self) -> str:
        return "Upstox"

    def is_available(self) -> bool:
        from app.services.upstox_data import upstox_service
        return upstox_service.is_available(token=_resolve_user_token())

    def get_ohlcv(self, symbol: str, period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
        from app.services.upstox_data import upstox_service
        try:
            return upstox_service.get_ohlcv(
                symbol, period=period, interval=interval,
                token=_resolve_user_token(),
            )
        except Exception:
            return pd.DataFrame()

    def get_login_url(self) -> Optional[str]:
        from app.services.upstox_data import upstox_service
        return upstox_service.get_login_url()

    def exchange_code(self, code: str) -> Optional[str]:
        from app.services.upstox_data import upstox_service
        return upstox_service.exchange_code(code)
