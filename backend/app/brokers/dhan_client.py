"""
Dhan HQ broker plugin.

Behavior depends on whether a current_user_id is set in the request context:
  - With user context: looks up the user's encrypted token from BrokerCredential
    and uses it for this call. Other users' tokens are never touched.
  - Without user context (backtest scripts, etc.): falls back to env-var token.
"""

import logging
from typing import Optional

import pandas as pd

from app.brokers.base import BrokerClient

logger = logging.getLogger(__name__)


def _resolve_user_token() -> Optional[str]:
    """
    Return the Dhan access token for the current request's user, or None.
    Returns None when no user is in context or no DB row exists (caller falls
    through to env-var token).
    """
    try:
        from app.services.user_context import get_current_user_id
        user_id = get_current_user_id()
        if user_id is None:
            return None
        from app.models.database import SessionLocal
        from app.services.broker_credentials import get_token
        db = SessionLocal()
        try:
            return get_token(db, user_id, "Dhan")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"DhanBroker user-token resolution failed: {e}")
        return None


class DhanBroker(BrokerClient):
    @property
    def name(self) -> str:
        return "Dhan"

    def is_available(self) -> bool:
        from app.services.dhan_data import dhan_data_service
        return dhan_data_service.is_available(token=_resolve_user_token())

    def get_ohlcv(self, symbol: str, period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
        from app.services.dhan_data import dhan_data_service
        try:
            return dhan_data_service.get_ohlcv(
                symbol, period=period, interval=interval,
                token=_resolve_user_token(),
            )
        except Exception:
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[float]:
        from app.services.dhan_data import dhan_data_service
        try:
            result = dhan_data_service.get_quote(symbol, token=_resolve_user_token())
            return result or None
        except Exception:
            return None
