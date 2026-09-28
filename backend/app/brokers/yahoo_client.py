"""
Yahoo Finance broker plugin — read-only. Used as universal fallback.
"""

from typing import Optional

import pandas as pd

from app.brokers.base import BrokerClient


class YahooBroker(BrokerClient):
    @property
    def name(self) -> str:
        return "Yahoo"

    def is_available(self) -> bool:
        # Yahoo is always "available" — graceful degradation on 429 happens inside
        return True

    def get_ohlcv(self, symbol: str, period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
        from app.services.yahoo_finance import yahoo_service
        # Yahoo expects .NS suffix for NSE equities
        sym = symbol if "." in symbol or symbol.startswith("^") else f"{symbol}.NS"
        try:
            return yahoo_service.get_ohlcv(sym, period=period, interval=interval)
        except Exception:
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[float]:
        from app.services.yahoo_finance import yahoo_service
        sym = symbol if "." in symbol or symbol.startswith("^") else f"{symbol}.NS"
        try:
            q = yahoo_service.get_quote(sym)
            return float(q) if q else None
        except Exception:
            return None
