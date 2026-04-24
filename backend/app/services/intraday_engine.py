"""
ASTRA Intraday Engine
======================
Generates intraday signals using 15-minute OHLCV data.
Supports Opening Range Breakout (ORB), VWAP Mean Reversion,
and momentum strategies.

Data source: yfinance (interval="15m", period="5d")
"""

import logging
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MARKET_OPEN      = "09:15"
MARKET_CLOSE     = "15:30"
ORB_WINDOW_MIN   = 30          # First 30 minutes define the Opening Range
INTRADAY_SL_PCT  = 0.003       # 0.3% stop-loss
INTRADAY_TP_PCT  = 0.006       # 0.6% take-profit
NO_TRADE_AFTER   = "14:45"     # No new entries after this time

_IST = pytz.timezone("Asia/Kolkata")
_UTC = pytz.utc


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fetch_intraday(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    Fetch 15-minute OHLCV bars for the last `days` trading days.
    Returns a cleaned DataFrame with timezone-aware IST index.
    """
    try:
        df = yf.download(
            symbol,
            period=f"{days}d",
            interval="15m",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
        if df is None or df.empty:
            logger.warning(f"[intraday] No data for {symbol}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is timezone-aware IST
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(_IST)
        else:
            df.index = df.index.tz_convert(_IST)

        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df[df["Volume"] > 0]
        return df

    except Exception as exc:
        logger.error(f"[intraday] _fetch_intraday({symbol}) failed: {exc}")
        return pd.DataFrame()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative VWAP: cumsum(typical_price * volume) / cumsum(volume).
    typical_price = (H + L + C) / 3
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tpv = (typical * df["Volume"]).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return vwap


def _compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.where(loss != 0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    rsi   = rsi.where(loss != 0, 100.0)
    return rsi


def _compute_orb(df: pd.DataFrame, date: datetime.date) -> tuple:
    """
    Compute the Opening Range (high/low) from the first ORB_WINDOW_MIN minutes
    of trading for the given date.

    Returns (orb_high, orb_low). Returns (None, None) if insufficient data.
    """
    # Filter to the target date
    day_df = df[df.index.date == date]
    if day_df.empty:
        return None, None

    market_open_time = dtime(9, 15)
    orb_end_time     = dtime(9, 15 + ORB_WINDOW_MIN)  # 09:45

    orb_df = day_df[
        (day_df.index.time >= market_open_time) &
        (day_df.index.time < orb_end_time)
    ]
    if orb_df.empty:
        return None, None

    return float(orb_df["High"].max()), float(orb_df["Low"].min())


# ── Strategy 1: Opening Range Breakout ───────────────────────────────────────

def orb_signal(df: pd.DataFrame, date: datetime.date) -> dict:
    """
    ORB strategy for a given date.

    BUY  if latest close breaks above ORB high with volume > 1.5× avg volume.
    SELL if latest close breaks below ORB low  with volume > 1.5× avg volume.

    Returns a signal dict with keys: signal, entry_price, sl, tp, strategy.
    """
    orb_high, orb_low = _compute_orb(df, date)
    if orb_high is None:
        return {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "ORB"}

    day_df = df[df.index.date == date].copy()
    avg_vol = day_df["Volume"].mean() if not day_df.empty else 0

    # Look at bars after the ORB window
    orb_end_time    = dtime(9, 15 + ORB_WINDOW_MIN)
    no_trade_time   = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])
    post_orb = day_df[
        (day_df.index.time >= orb_end_time) &
        (day_df.index.time <= no_trade_time)
    ]

    if post_orb.empty:
        return {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "ORB"}

    last     = post_orb.iloc[-1]
    close    = float(last["Close"])
    vol_ok   = float(last["Volume"]) > 1.5 * avg_vol if avg_vol > 0 else False

    if close > orb_high and vol_ok:
        entry = close
        return {
            "signal":      "BUY",
            "entry_price": round(entry, 2),
            "sl":          round(entry * (1 - INTRADAY_SL_PCT), 2),
            "tp":          round(entry * (1 + INTRADAY_TP_PCT), 2),
            "strategy":    "ORB",
        }
    elif close < orb_low and vol_ok:
        entry = close
        return {
            "signal":      "SELL",
            "entry_price": round(entry, 2),
            "sl":          round(entry * (1 + INTRADAY_SL_PCT), 2),
            "tp":          round(entry * (1 - INTRADAY_TP_PCT), 2),
            "strategy":    "ORB",
        }

    return {"signal": "HOLD", "entry_price": close, "sl": 0.0, "tp": 0.0, "strategy": "ORB"}


# ── Strategy 2: VWAP Mean Reversion ──────────────────────────────────────────

def vwap_signal(df: pd.DataFrame) -> dict:
    """
    VWAP Mean-Reversion strategy on the most recent bar.

    BUY  if price < VWAP - 0.5% AND RSI < 35  (oversold below VWAP → snap back)
    SELL if price > VWAP + 0.5% AND RSI > 65  (overbought above VWAP → mean revert)

    Returns a signal dict with keys: signal, entry_price, sl, tp, strategy.
    """
    if df.empty or len(df) < 15:
        return {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "VWAP_MR"}

    df = df.copy()
    df["VWAP"] = _compute_vwap(df)
    df["RSI"]  = _compute_rsi_series(df["Close"])

    last  = df.dropna(subset=["VWAP", "RSI"]).iloc[-1] if not df.dropna(subset=["VWAP", "RSI"]).empty else None
    if last is None:
        return {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "VWAP_MR"}

    close  = float(last["Close"])
    vwap   = float(last["VWAP"])
    rsi    = float(last["RSI"])

    vwap_lower = vwap * (1 - 0.005)   # VWAP - 0.5%
    vwap_upper = vwap * (1 + 0.005)   # VWAP + 0.5%

    if close < vwap_lower and rsi < 35:
        return {
            "signal":      "BUY",
            "entry_price": round(close, 2),
            "sl":          round(close * (1 - INTRADAY_SL_PCT), 2),
            "tp":          round(close * (1 + INTRADAY_TP_PCT), 2),
            "strategy":    "VWAP_MR",
        }
    elif close > vwap_upper and rsi > 65:
        return {
            "signal":      "SELL",
            "entry_price": round(close, 2),
            "sl":          round(close * (1 + INTRADAY_SL_PCT), 2),
            "tp":          round(close * (1 - INTRADAY_TP_PCT), 2),
            "strategy":    "VWAP_MR",
        }

    return {"signal": "HOLD", "entry_price": close, "sl": 0.0, "tp": 0.0, "strategy": "VWAP_MR"}


# ── IntradayEngine class ──────────────────────────────────────────────────────

class IntradayEngine:
    """
    Orchestrates intraday signal generation using ORB and VWAP strategies.
    """

    def __init__(self):
        logger.info("ASTRA IntradayEngine initialised (ORB + VWAP_MR strategies)")

    @staticmethod
    def _market_session() -> str:
        """Return current IST market session label."""
        now_ist  = datetime.now(_IST).time()
        open_t   = dtime(9, 15)
        pre_open = dtime(9, 0)
        close_t  = dtime(15, 30)
        if now_ist < pre_open:
            return "PRE_MARKET"
        elif now_ist < open_t:
            return "OPENING"
        elif now_ist <= close_t:
            return "MARKET_HOURS"
        return "AFTER_HOURS"

    @staticmethod
    def _time_remaining_min() -> int:
        """Minutes remaining until market close (15:30 IST). Negative if closed."""
        now_ist  = datetime.now(_IST)
        close_dt = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        delta    = (close_dt - now_ist).total_seconds() / 60
        return int(delta)

    def analyze_intraday(self, symbol: str) -> dict:
        """
        Full intraday analysis for a symbol.

        Steps:
          1. Fetch 15-minute OHLCV data (5 days).
          2. Compute VWAP and ORB for today's session.
          3. Run both ORB and VWAP strategies.
          4. Return consensus (both agree) or highest-confidence signal.

        Returns
        -------
        dict with keys:
          symbol, signal, confidence, strategy, entry_price, sl, tp,
          time_of_signal, market_session, time_remaining_min
        """
        now_ist  = datetime.now(_IST)
        today    = now_ist.date()

        base_result = {
            "symbol":           symbol,
            "signal":           "HOLD",
            "confidence":       0.0,
            "strategy":         "NONE",
            "entry_price":      0.0,
            "sl":               0.0,
            "tp":               0.0,
            "time_of_signal":   now_ist.strftime("%H:%M:%S"),
            "market_session":   self._market_session(),
            "time_remaining_min": self._time_remaining_min(),
        }

        try:
            df = _fetch_intraday(symbol, days=5)
            if df.empty:
                logger.warning(f"[intraday] No data for {symbol}")
                return base_result

            # Run strategies
            orb_res  = orb_signal(df, today)
            vwap_res = vwap_signal(df)

            # Consensus: both agree
            if orb_res["signal"] != "HOLD" and vwap_res["signal"] != "HOLD":
                if orb_res["signal"] == vwap_res["signal"]:
                    base_result.update({
                        "signal":     orb_res["signal"],
                        "confidence": 85.0,
                        "strategy":   "ORB+VWAP_MR",
                        "entry_price":orb_res["entry_price"],
                        "sl":         orb_res["sl"],
                        "tp":         orb_res["tp"],
                    })
                    return base_result

            # Fallback: whichever strategy has a non-HOLD signal
            if orb_res["signal"] != "HOLD":
                base_result.update({
                    "signal":     orb_res["signal"],
                    "confidence": 65.0,
                    "strategy":   "ORB",
                    "entry_price":orb_res["entry_price"],
                    "sl":         orb_res["sl"],
                    "tp":         orb_res["tp"],
                })
                return base_result

            if vwap_res["signal"] != "HOLD":
                base_result.update({
                    "signal":     vwap_res["signal"],
                    "confidence": 60.0,
                    "strategy":   "VWAP_MR",
                    "entry_price":vwap_res["entry_price"],
                    "sl":         vwap_res["sl"],
                    "tp":         vwap_res["tp"],
                })
                return base_result

        except Exception as exc:
            logger.error(f"[intraday] analyze_intraday({symbol}) error: {exc}", exc_info=True)

        return base_result


# ── Module-level singleton ────────────────────────────────────────────────────

intraday_engine = IntradayEngine()
