"""
ASTRA Intraday Universe Scorer
================================
Each morning, rank NSE stocks by their suitability for each intraday strategy.

Scoring rationale (research-driven, not arbitrary):
  - ORB        → wants trending + volatile stocks (ADX > 22, ATR% in 1-3% band)
  - EMA_Cross  → wants directional trend (SMA50 slope > 0, price above SMA50)
  - Momentum   → wants high-beta volatile stocks (ATR% > 1.5%, recent breakouts)

Daily output: ranked list per strategy, top-N (default 20) used for the day.
Cached for the trading day (TTL: until date changes).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Universe — NIFTY-200 subset that has liquid intraday markets
# (Drawn from the larger NIFTY 500 list but trimmed to >₹50cr avg daily turnover)
INTRADAY_UNIVERSE = [
    # NIFTY 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "KOTAKBANK", "BHARTIARTL", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ADANIENT", "POWERGRID", "TECHM", "INDUSINDBK", "DIVISLAB", "JSWSTEEL",
    "BAJAJFINSV", "COALINDIA", "HCLTECH", "ONGC", "NTPC", "TATAMOTORS",
    "TATASTEEL", "CIPLA", "BRITANNIA", "DRREDDY", "BPCL", "HEROMOTOCO",
    "GRASIM", "EICHERMOT", "TATACONSUM", "SBILIFE", "HDFCLIFE",
    "APOLLOHOSP", "HINDALCO", "ADANIPORTS", "BAJAJ-AUTO", "ITC",
    # Top intraday movers from NIFTY NEXT 50
    "SIEMENS", "HAVELLS", "VOLTAS", "PIDILITIND", "TATAPOWER", "ADANIGREEN",
    "DMART", "INDIGO", "DLF", "GODREJCP", "CHOLAFIN", "TRENT",
    "AUROPHARMA", "LUPIN", "BIOCON", "BANDHANBNK", "IDFCFIRSTB", "PNB",
    "GAIL", "IOC", "RECLTD", "PFC", "IRFC", "RVNL",
    # High-beta midcaps
    "DIXON", "POLYCAB", "TIINDIA", "KPITTECH", "PERSISTENT", "LTIM",
    "MOTHERSON", "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR",
    "INDIANB", "CANBK", "BANKBARODA", "UNIONBANK",
]

# In-memory cache: date -> {strategy: ranked_list}
_CACHE: dict = {}
_CACHE_DATE: Optional[str] = None


def _fetch_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Try Upstox (no rate limit) first, then Yahoo. Returns 30+ daily bars or None."""
    # 1. Upstox — preferred, no rate limit, requires daily login
    try:
        from app.services.upstox_data import upstox_service
        if upstox_service.is_available():
            df = upstox_service.get_ohlcv(symbol, period="3mo", interval="1d")
            if df is not None and not df.empty and len(df) >= 30:
                return df
    except Exception as e:
        logger.debug(f"[universe] Upstox fetch failed for {symbol}: {e}")

    # 2. Yahoo — fallback (often 429'd during market hours)
    try:
        from app.services.yahoo_finance import yahoo_service
        df = yahoo_service.get_ohlcv(f"{symbol}.NS", period="3mo", interval="1d")
        if df is not None and not df.empty and len(df) >= 30:
            return df
    except Exception as e:
        logger.debug(f"[universe] Yahoo fetch failed for {symbol}: {e}")

    return None


def _score_single(symbol: str) -> Optional[dict]:
    """
    Fetch 30 days of daily bars, compute scoring metrics.
    Returns a dict with scores per strategy, or None on failure.
    """
    try:
        df = _fetch_daily(symbol)
        if df is None:
            return None

        df = df.tail(30).copy()
        df["TR"] = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_14 = df["TR"].rolling(14).mean().iloc[-1]
        close  = df["Close"].iloc[-1]
        atr_pct = (atr_14 / close * 100) if close > 0 else 0

        # SMA50 proxy on 30-day data using 20-period MA (we only have 30 daily bars)
        sma20 = df["Close"].rolling(20).mean()
        sma_slope = (sma20.iloc[-1] - sma20.iloc[-10]) / sma20.iloc[-10] * 100 if sma20.iloc[-10] > 0 else 0
        above_sma = close > sma20.iloc[-1]

        # ADX (14)
        up   = df["High"].diff()
        down = -df["Low"].diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr14 = df["TR"].rolling(14).sum()
        plus_di14  = 100 * pd.Series(plus_dm,  index=df.index).rolling(14).sum() / tr14
        minus_di14 = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
        dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14 + 1e-9)
        adx = dx.rolling(14).mean().iloc[-1] if len(dx) >= 14 else 15.0

        # Liquidity: 20-day avg turnover in ₹ crore
        avg_turnover_cr = (df["Close"] * df["Volume"]).tail(20).mean() / 1e7

        # 20-day return (momentum)
        ret_20d = (close / df["Close"].iloc[-20] - 1) * 100 if df["Close"].iloc[-20] > 0 else 0

        # ── Per-strategy scores (0–100) ──────────────────────────────────────
        # ORB: needs ADX > 22 AND ATR% in 1-3% AND liquid
        orb_score = 0.0
        if adx >= 22 and 1.0 <= atr_pct <= 3.5 and avg_turnover_cr >= 50:
            orb_score = min(100, adx * 1.5 + atr_pct * 8)

        # EMA_Cross: directional trend, price above SMA, slope positive
        ema_score = 0.0
        if above_sma and sma_slope > 1.0 and avg_turnover_cr >= 50:
            ema_score = min(100, sma_slope * 4 + adx)

        # Momentum: high recent return + volatility
        mom_score = 0.0
        if atr_pct >= 1.5 and abs(ret_20d) >= 3 and avg_turnover_cr >= 50:
            mom_score = min(100, abs(ret_20d) * 3 + atr_pct * 5)

        return {
            "symbol":    symbol,
            "adx":       round(float(adx), 1),
            "atr_pct":   round(float(atr_pct), 2),
            "sma_slope": round(float(sma_slope), 2),
            "ret_20d":   round(float(ret_20d), 2),
            "turnover":  round(float(avg_turnover_cr), 1),
            "scores":    {
                "ORB":       round(orb_score, 1),
                "EMA_Cross": round(ema_score, 1),
                "Momentum":  round(mom_score, 1),
            },
        }
    except Exception as e:
        logger.debug(f"[universe] {symbol} scoring failed: {e}")
        return None


def rank_universe(top_n: int = 20, max_workers: int = 6) -> dict:
    """
    Score the full intraday universe and return ranked lists per strategy.
    Cached for the current trading day.

    Returns:
        {
            "ORB":       ["RELIANCE", "INFY", ...],     # top_n by ORB score
            "EMA_Cross": [...],
            "Momentum":  [...],
            "scores":    {symbol: full_metrics_dict},   # for inspection
            "scored_at": ISO timestamp,
        }
    """
    global _CACHE, _CACHE_DATE
    today = datetime.now().strftime("%Y-%m-%d")
    if _CACHE_DATE == today and _CACHE:
        return _CACHE

    logger.info(f"[universe] Scoring {len(INTRADAY_UNIVERSE)} symbols...")
    start = time.time()

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_single, s): s for s in INTRADAY_UNIVERSE}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results[r["symbol"]] = r

    ranked: dict = {"scores": results, "scored_at": datetime.now().isoformat()}
    for strat in ("ORB", "EMA_Cross", "Momentum"):
        sorted_syms = sorted(
            results.values(),
            key=lambda x: x["scores"][strat],
            reverse=True,
        )
        # Only include stocks with non-zero score for that strategy
        ranked[strat] = [r["symbol"] for r in sorted_syms if r["scores"][strat] > 0][:top_n]

    _CACHE      = ranked
    _CACHE_DATE = today
    duration = round(time.time() - start, 1)
    logger.info(
        f"[universe] Scored {len(results)}/{len(INTRADAY_UNIVERSE)} in {duration}s — "
        f"ORB:{len(ranked['ORB'])} EMA:{len(ranked['EMA_Cross'])} MOM:{len(ranked['Momentum'])}"
    )
    return ranked
