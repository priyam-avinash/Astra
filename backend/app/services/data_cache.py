"""
ASTRA Data Cache — Local disk cache for OHLCV data.
=====================================================
Eliminates yfinance rate-limit issues by caching OHLCV CSV files on disk.
Cache is valid for:
  - Daily data:    24 hours on weekdays, 72 hours on weekends (market closed)
  - Hourly data:   30 minutes
  - 4h/15m data:   60 minutes
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(symbol: str, interval: str) -> str:
    safe = symbol.replace("/", "_").replace("^", "IDX_").replace("=", "FX_")
    return os.path.join(CACHE_DIR, f"{safe}_{interval}.csv")


def _ttl_seconds(interval: str) -> int:
    """Return cache TTL in seconds based on interval."""
    now = datetime.now()
    is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6

    if interval == "1d":
        return 72 * 3600 if is_weekend else 8 * 3600   # 72h weekends, 8h weekdays
    elif interval in ("4h", "1h"):
        return 60 * 60      # 1 hour
    elif interval in ("30m", "15m"):
        return 30 * 60      # 30 minutes
    return 4 * 3600         # Default 4 hours


def get(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """
    Return cached DataFrame if fresh, otherwise returns empty DataFrame.
    Caller should check if the result is empty and fetch from API if so.
    """
    path = _cache_key(symbol, interval)
    if not os.path.exists(path):
        return pd.DataFrame()

    mtime = os.path.getmtime(path)
    age = datetime.now().timestamp() - mtime
    ttl = _ttl_seconds(interval)

    if age > ttl:
        logger.debug(f"Cache STALE for {symbol} ({interval}) — age {age/3600:.1f}h > TTL {ttl/3600:.1f}h")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty and len(df) > 10:
            logger.info(f"Cache HIT for {symbol} ({interval}): {len(df)} bars, age {age/60:.0f}m")
            return df
    except Exception as e:
        logger.debug(f"Cache read failed for {symbol}: {e}")

    return pd.DataFrame()


def put(symbol: str, df: pd.DataFrame, interval: str = "1d") -> None:
    """Save DataFrame to cache. Silently ignores errors."""
    if df is None or df.empty:
        return
    try:
        path = _cache_key(symbol, interval)
        df.to_csv(path)
        logger.debug(f"Cache WRITE for {symbol} ({interval}): {len(df)} bars → {path}")
    except Exception as e:
        logger.debug(f"Cache write failed for {symbol}: {e}")


def invalidate(symbol: str, interval: str = "1d") -> None:
    """Force-invalidate cache for a symbol (e.g., after a manual refresh)."""
    path = _cache_key(symbol, interval)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Cache INVALIDATED for {symbol} ({interval})")


def stats() -> dict:
    """Return cache statistics for monitoring."""
    if not os.path.exists(CACHE_DIR):
        return {"files": 0, "size_mb": 0}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    total_bytes = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files
    )
    return {
        "files": len(files),
        "size_mb": round(total_bytes / 1024 / 1024, 2),
        "cache_dir": CACHE_DIR,
    }
