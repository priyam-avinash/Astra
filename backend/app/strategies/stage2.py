"""
ASTRA Stage-2 v2.0 — Trend Following (Weinstein/Minervini)
===========================================================
v2.0 changes: Supertrend bearish flip replaces Close < SMA150 exit;
ATR-based hard stop replaces fixed -7% stop.

Entry (unchanged from v1):
  - Close > SMA150 (≈30-week MA, defines Stage 2)
  - SMA150 slope > 0 over last 30 days (uptrend confirmed)
  - Close within 8% of 252-day high (price strength filter)
  - Volume > 1.5× 20-day avg (breakout volume confirmation)
  - 6-month return > +5% (already-trending stock)
  - NIFTY > NIFTY's own SMA150 (market regime filter)

Exit (first triggered):
  A. Supertrend(10, 3.0) flips bearish  ← replaces Close < SMA150
  B. Hard stop: entry − 2.5×ATR(14)     ← replaces fixed -7%
  C. Max stop width gate: reject signal if stop > 12% of entry
  D. Time stop: 40 trading days
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    Strategy, Signal, Position, ExitReason, SignalSide, StrategyTimeframe, StrategyMeta
)
from app.strategies.indicators import compute_atr, compute_daily_vwap, compute_hv_rank, compute_supertrend
from app.strategies.filters import passes_enhancement_filters

# ── Parameters ───────────────────────────────────────────────────────────────
SMA_PERIOD            = 150
SMA_SLOPE_LOOKBACK    = 30
HIGH_PROXIMITY_PCT    = 0.92
VOL_MULTIPLIER        = 1.5
MIN_6M_RETURN         = 0.05
USE_NIFTY_REGIME      = True

SUPERTREND_PERIOD     = 10
SUPERTREND_MULT       = 3.0
ATR_STOP_MULT         = 2.5
MAX_STOP_PCT          = 0.12        # reject if ATR stop > 12% away from entry
TIME_STOP_DAYS        = 40


class Stage2Strategy(Strategy):
    """Weinstein/Minervini Stage-2 trend following. v2.0: Supertrend exit + ATR stop."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.STAGE2",
            version="2.0.0",
            description=(
                "Stage-2 trend following: Supertrend exit replaces SMA150 breakdown; "
                "ATR-based stop replaces fixed -7%"
            ),
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["trend", "positional", "factor", "supertrend"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.sort_index().copy()
        df["SMA"]               = df["Close"].rolling(SMA_PERIOD).mean()
        df["SMA_slope"]         = df["SMA"].diff(SMA_SLOPE_LOOKBACK)
        df["high_252"]          = df["High"].rolling(252).max()
        df["vol_avg_20"]        = df["Volume"].rolling(20).mean()
        df["ret_6m"]            = df["Close"].pct_change(126)
        df["atr14"]             = compute_atr(df, 14)
        df["supertrend_bullish"] = compute_supertrend(df, atr_period=SUPERTREND_PERIOD,
                                                       multiplier=SUPERTREND_MULT)
        df["hv_rank"] = compute_hv_rank(df["Close"])
        df["vwap"]    = compute_daily_vwap(df)
        return df

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < SMA_PERIOD + 130:
            return None
        row = bars.iloc[-1]
        for col in ("SMA", "high_252", "vol_avg_20", "ret_6m", "atr14", "supertrend_bullish"):
            if pd.isna(row.get(col)):
                return None

        close = float(row["Close"])
        if not (close > row["SMA"]):
            return None
        if not (row["SMA_slope"] > 0):
            return None
        if not (close >= HIGH_PROXIMITY_PCT * row["high_252"]):
            return None
        if not (row["Volume"] > VOL_MULTIPLIER * row["vol_avg_20"]):
            return None
        if not (row["ret_6m"] > MIN_6M_RETURN):
            return None
        if USE_NIFTY_REGIME and not _nifty_in_stage2(bars.index[-1]):
            return None

        atr14 = float(row["atr14"])
        hard_stop = close - ATR_STOP_MULT * atr14

        # Reject if stop is too wide (high-ATR name — risk not manageable)
        if (close - hard_stop) / close > MAX_STOP_PCT:
            return None

        # Enhancement filters (HV rank, India VIX, VWAP) — active only when flags are ON
        if not passes_enhancement_filters(row):
            return None

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=None,
            trail_pct=0.0,
            trail_activate_pct=0.0,
            confidence=60.0,
            metadata={"atr14": round(atr14, 2)},
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        if len(bars) < 15:
            return None
        row = bars.iloc[-1]

        # Primary exit: Supertrend bearish flip
        if "supertrend_bullish" in row.index and not pd.isna(row["supertrend_bullish"]):
            if not bool(row["supertrend_bullish"]):
                return ExitReason.STRATEGY

        # Time stop
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= int(TIME_STOP_DAYS * 1.4):
            return ExitReason.TIME_STOP

        return None


# ── NIFTY regime lookup (lazy-loaded once per process) ─────────────────────
_NIFTY_REGIME: Optional[pd.DataFrame] = None


def _load_nifty_regime() -> Optional[pd.DataFrame]:
    global _NIFTY_REGIME
    if _NIFTY_REGIME is not None:
        return _NIFTY_REGIME
    # v1.13: shared market-data layer (fast-failing, cached)
    try:
        from app.services.market_data import market_data
        raw = market_data.get_ohlcv("^NSEI", period="2y", interval="1d")
        if len(raw) >= SMA_PERIOD + 30:
            df = raw[["Close"]].sort_index().copy()
            df["SMA"] = df["Close"].rolling(SMA_PERIOD).mean()
            _NIFTY_REGIME = df
            return _NIFTY_REGIME
    except Exception:
        pass
    try:
        from app.services.upstox_data import upstox_service
        if upstox_service.is_available():
            df = upstox_service.get_ohlcv("NIFTYBEES", period="2y", interval="1d")
            if df is not None and not df.empty and len(df) >= SMA_PERIOD + 30:
                df = df.sort_index().copy()
                df["SMA"] = df["Close"].rolling(SMA_PERIOD).mean()
                _NIFTY_REGIME = df
                return _NIFTY_REGIME
    except Exception:
        pass
    return None


def _nifty_in_stage2(date) -> bool:
    df = _load_nifty_regime()
    if df is None:
        return True
    try:
        idx = df.index.get_indexer([date], method="pad")[0]
        if idx < 0:
            return True
        row = df.iloc[idx]
        if pd.isna(row["SMA"]):
            return True
        return float(row["Close"]) > float(row["SMA"])
    except Exception:
        return True
