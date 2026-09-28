"""
Zerodha Kite Connect broker plugin.
Requires: KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN in env (token daily refresh).

Setup steps for users:
  1. Create a Kite Connect app at https://developers.kite.trade
  2. Set KITE_API_KEY and KITE_API_SECRET in .env
  3. Visit /zerodha/login (when endpoint is wired) to do the daily OAuth flow
  4. Token gets saved to env and refreshed each trading day (SEBI rule)

Behavior depends on whether a current_user_id is set in the request context:
  - With user context: looks up the user's encrypted token from BrokerCredential.
  - Without user context: falls back to env KITE_ACCESS_TOKEN.

This is a *paper-safe* implementation: get_ohlcv works once token is set,
but place_order is deliberately not implemented — execution stays gated through
the broker_service singleton (PaperTradingEngine).
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from app.brokers.base import BrokerClient

logger = logging.getLogger(__name__)

# Kite uses different interval strings than yfinance
_INTERVAL_MAP = {
    "1m":   "minute",
    "5m":   "5minute",
    "15m":  "15minute",
    "1h":   "60minute",
    "1d":   "day",
    "daily": "day",
}

# period → days
_PERIOD_DAYS = {
    "5d":  5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
}


def _resolve_user_token() -> Optional[str]:
    """
    Return the Zerodha access token for the current request's user, or None.
    Returns None when no user is in context or no DB row exists (caller falls
    through to env KITE_ACCESS_TOKEN).
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
            return get_token(db, user_id, "Zerodha")
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"ZerodhaBroker user-token resolution failed: {e}")
        return None


class ZerodhaBroker(BrokerClient):
    @property
    def name(self) -> str:
        return "Zerodha"

    def is_available(self) -> bool:
        token = _resolve_user_token() or os.getenv("KITE_ACCESS_TOKEN")
        return bool(os.getenv("KITE_API_KEY")) and bool(token)

    def _client(self, token: Optional[str] = None):
        """Lazy-init Kite client, preferring the supplied token over env."""
        access_token = token or os.getenv("KITE_ACCESS_TOKEN")
        if not os.getenv("KITE_API_KEY") or not access_token:
            return None
        try:
            from kiteconnect import KiteConnect
            kc = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
            kc.set_access_token(access_token)
            return kc
        except Exception as e:
            logger.error(f"Zerodha client init failed: {e}")
            return None

    def get_ohlcv(self, symbol: str, period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
        kc = self._client(token=_resolve_user_token())
        if kc is None:
            return pd.DataFrame()
        kite_interval = _INTERVAL_MAP.get(interval, "day")
        days = _PERIOD_DAYS.get(period, 30)

        try:
            instrument_token = self._resolve_token(symbol, kc)
            if instrument_token is None:
                logger.warning(f"Zerodha: no token for {symbol}")
                return pd.DataFrame()

            to_dt   = datetime.now()
            from_dt = to_dt - timedelta(days=days)
            data = kc.historical_data(
                instrument_token=instrument_token,
                from_date=from_dt,
                to_date=to_dt,
                interval=kite_interval,
            )
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df = df.rename(columns={
                "date": "Datetime",
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            }).set_index("Datetime").sort_index()
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Kolkata")
            else:
                df.index = df.index.tz_convert("Asia/Kolkata")
            return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        except Exception as e:
            logger.error(f"Zerodha ohlcv failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[float]:
        kc = self._client(token=_resolve_user_token())
        if kc is None:
            return None
        try:
            tkn = self._resolve_token(symbol, kc)
            if tkn is None:
                return None
            quote = kc.ltp([f"NSE:{symbol}"])
            return float(quote[f"NSE:{symbol}"]["last_price"])
        except Exception:
            return None

    def get_login_url(self) -> Optional[str]:
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            return None
        return f"https://kite.trade/connect/login?api_key={api_key}&v=3"

    def exchange_code(self, request_token: str) -> Optional[str]:
        """Kite's request_token → access_token exchange. Token valid for 1 day."""
        api_key = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")
        if not api_key or not api_secret:
            return None
        try:
            from kiteconnect import KiteConnect
            kc = KiteConnect(api_key=api_key)
            data = kc.generate_session(request_token, api_secret=api_secret)
            token = data.get("access_token")
            if token:
                os.environ["KITE_ACCESS_TOKEN"] = token
                logger.info("Zerodha access token obtained and stored in env")
            return token
        except Exception as e:
            logger.error(f"Zerodha token exchange failed: {e}")
            return None

    # ── Internal ────────────────────────────────────────────────────────

    _SYMBOL_TOKEN_CACHE: dict = {}

    def _resolve_token(self, symbol: str, kc=None) -> Optional[int]:
        """
        Resolve symbol → Kite instrument_token. Lazy-loads the instruments dump.
        Accepts an optional pre-built kc client to avoid a second _client() call.
        """
        if not self.__class__._SYMBOL_TOKEN_CACHE:
            client = kc or self._client(token=_resolve_user_token())
            if client is None:
                return None
            try:
                instruments = client.instruments("NSE")
                self.__class__._SYMBOL_TOKEN_CACHE = {
                    i["tradingsymbol"]: int(i["instrument_token"])
                    for i in instruments
                    if i.get("instrument_type") == "EQ"
                }
                logger.info(f"Zerodha instruments cached: {len(self.__class__._SYMBOL_TOKEN_CACHE)} NSE EQ")
            except Exception as e:
                logger.warning(f"Zerodha instruments fetch failed: {e}")
                return None
        return self.__class__._SYMBOL_TOKEN_CACHE.get(symbol)
