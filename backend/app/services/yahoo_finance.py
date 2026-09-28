"""
Deprecated compatibility shim (v1.13).

The Yahoo/NSE code that lived here moved into app.services.market_data, which
adds Chrome TLS impersonation (curl_cffi), provider fallbacks, caching and
health tracking. These wrappers keep old imports working.
"""
import pandas as pd

from app.services.market_data import market_data, normalize_symbol


def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    return market_data.get_ohlcv(symbol, period=period, interval=interval)


def get_quote(symbol: str) -> float:
    return market_data.get_quote(symbol)


def get_nse_quote(symbol: str) -> dict:
    info = market_data.get_quote_info(symbol)
    return {"symbol": normalize_symbol(symbol), "lastPrice": info["price"], "source": info["source"]}


def get_nse_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    return market_data.get_ohlcv(symbol, period=f"{days}d", interval="1d")


def clear_cache(symbol=None) -> None:
    market_data.clear_cache(symbol)


class YahooFinanceService:
    get_ohlcv = staticmethod(get_ohlcv)
    get_quote = staticmethod(get_quote)
    get_nse_quote = staticmethod(get_nse_quote)
    get_nse_ohlcv = staticmethod(get_nse_ohlcv)

    @staticmethod
    def is_available() -> bool:
        return True


yahoo_service = YahooFinanceService()
