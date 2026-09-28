"""
ASTRA.PULLBACK v3.23 — Supertrend-Gated Pullback to EMA20 with RSI Crossover
=============================================================================
Key changes from v3.2:
  - Zone anchor: EMA50 → EMA20 (faster anchor = sharper bounces, higher WR)
  - RSI condition: sub-50 dip no longer required; RSI crossover ABOVE 55 is sufficient
    (strong stocks in healthy pullbacks dip only to 51-54; the sub-50 check was
     excluding the best setups)
  - Candle filter: close > open (intraday strength confirmation)
  - Zone: [EMA20×0.97, EMA20×1.04] (tight around EMA20)

Entry (6 conditions):
  1. EMA20 > EMA50 > EMA200       (uptrend structure)
  2. Supertrend(10,3) bullish     (trend confirmed)
  3. Close ∈ [EMA20×0.97, EMA20×1.04]  (EMA20 pullback zone)
  4. RSI(14) crosses above 55 today   (recovery confirmation — no sub-50 dip required)
  5. Close > yesterday's Close AND Close > Open  (strong bullish candle)
  6. Volume > 1.1× 20-day avg
  7. NIFTY > NIFTY EMA200             (market regime)

Exit (first triggered):
  A. Fixed target: entry + 4×ATR(14)   ← captures mean-reversion bounce
  B. Hard stop: entry − 2×ATR(14)
  C. Time stop: 40 trading days (56 calendar days)
"""
from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    ExitReason, Position, Signal, SignalSide,
    Strategy, StrategyMeta, StrategyTimeframe,
)
from app.strategies.indicators import compute_atr, compute_daily_vwap, compute_ema, compute_hv_rank, compute_supertrend
from app.strategies.filters import passes_enhancement_filters

# ── Parameters ────────────────────────────────────────────────────────────────
PULLBACK_ZONE_LOW    = 0.97        # allow slight dip below EMA anchor
PULLBACK_ZONE_HIGH   = 1.04        # allow up to 4% above EMA anchor

RSI_OVERSOLD_THRESH  = 50          # slightly stricter than v3.0 (was 55)
RSI_RECOVERY_THRESH  = 55          # stronger recovery signal required (was 50)
RSI_LOOKBACK_DAYS    = 20          # recent oversold dip required (was 30)
RSI_DIP_MIN_BARS     = 1           # RSI must be oversold for ≥ 1 bar

VOL_MULTIPLIER       = 1.1         # lowered: bounce is already confirmation
USE_NIFTY_REGIME     = True

ATR_STOP_MULT        = 2.0         # widened: fewer noise exits
TARGET_ATR_MULT      = 4.5         # wider target: higher-quality setups can sustain bigger moves
MIN_RR_GATE          = 1.5

TRAIL_ACTIVATE_PCT   = 0.04
TRAIL_DRAWDOWN_PCT   = 0.07
TIME_STOP_DAYS       = 40          # reverted to v3.2 best

WIN_PROB             = 0.45

# ── NIFTY regime cache ────────────────────────────────────────────────────────
_NIFTY_CACHE: Optional[pd.DataFrame] = None


def _nifty_above_ema200(date) -> bool:
    """True if NIFTY Close > EMA200 on or just before `date`. Fail-open."""
    global _NIFTY_CACHE
    if _NIFTY_CACHE is None:
        # Use yf.download() directly — avoids yahoo_service retry/backoff hanging
        try:
            import yfinance as yf
            raw = yf.download(
                "^NSEI", period="2y", interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
            if raw is not None and not raw.empty and len(raw) >= 210:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                df = raw[["Close"]].sort_index().copy()
                df["EMA200"] = compute_ema(df["Close"], 200)
                _NIFTY_CACHE = df
        except Exception:
            pass
        if _NIFTY_CACHE is None:
            try:
                from app.services.upstox_data import upstox_service
                if upstox_service.is_available():
                    df = upstox_service.get_ohlcv("NIFTYBEES", period="2y", interval="1d")
                    if df is not None and not df.empty and len(df) >= 210:
                        df = df.sort_index().copy()
                        df["EMA200"] = compute_ema(df["Close"], 200)
                        _NIFTY_CACHE = df
            except Exception:
                pass
    if _NIFTY_CACHE is None:
        return True
    try:
        idx = _NIFTY_CACHE.index.get_indexer([date], method="pad")[0]
        if idx < 0:
            return True
        row = _NIFTY_CACHE.iloc[idx]
        if pd.isna(row.get("EMA200")):
            return True
        return float(row["Close"]) > float(row["EMA200"])
    except Exception:
        return True


class PullbackStrategy(Strategy):
    """Buy pullbacks to EMA50 in Supertrend-confirmed uptrends. v3.0: wider zone, Supertrend exit."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.PULLBACK",
            version="3.23.0",
            description=(
                "EMA20-anchored pullback in Supertrend uptrend: RSI crossover at 55 "
                "as recovery signal, 4×ATR target, 2×ATR stop, bullish candle confirmation"
            ),
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["mean-reversion", "trend-following", "positional", "supertrend"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.sort_index().copy()
        df["EMA20"]  = compute_ema(df["Close"], 20)
        df["EMA50"]  = compute_ema(df["Close"], 50)
        df["EMA200"] = compute_ema(df["Close"], 200)
        df["ATR14"]  = compute_atr(df, 14)
        df["RSI"]    = _compute_rsi(df["Close"], 14)
        df["RSI_was_oversold"] = (
            (df["RSI"] < RSI_OVERSOLD_THRESH)
            .rolling(RSI_LOOKBACK_DAYS).max()
            .astype(float)
        )
        df["vol_avg_20"]        = df["Volume"].rolling(20).mean()
        df["supertrend_bullish"] = compute_supertrend(df, atr_period=10, multiplier=3.0)

        # Weekly EMA20 slope — resample to weekly, forward-fill to daily index
        weekly = df[["Open", "High", "Low", "Close", "Volume"]].resample("W").last().dropna()
        weekly["ema20_w"]       = compute_ema(weekly["Close"], 20)
        weekly["ema20_w_slope"] = weekly["ema20_w"] - weekly["ema20_w"].shift(1)
        df["weekly_ema20_slope"] = weekly["ema20_w_slope"].reindex(df.index, method="ffill")

        df["hv_rank"] = compute_hv_rank(df["Close"])
        df["vwap"]    = compute_daily_vwap(df)
        return df

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < 210:
            return None

        row  = bars.iloc[-1]
        prev = bars.iloc[-2]

        required = ("EMA20", "EMA50", "EMA200", "ATR14", "RSI",
                    "RSI_was_oversold", "vol_avg_20", "supertrend_bullish")
        for col in required:
            if pd.isna(row.get(col)):
                return None

        close  = float(row["Close"])
        ema20  = float(row["EMA20"])
        ema50  = float(row["EMA50"])
        ema200 = float(row["EMA200"])
        atr14  = float(row["ATR14"])

        # 1. EMA alignment — uptrend structure
        if not (ema20 > ema50 > ema200):
            return None

        # 2. Supertrend bullish — trend confirmed
        if not bool(row["supertrend_bullish"]):
            return None

        # 3. Pullback zone: price near EMA20 (short-term mean-reversion anchor)
        if not (ema20 * PULLBACK_ZONE_LOW <= close <= ema20 * PULLBACK_ZONE_HIGH):
            return None

        # 4. RSI recovery crossover: RSI must cross above RECOVERY_THRESH today
        #    (rsi_prev < 55, rsi_now >= 55 — the crossover is sufficient; no need for
        #    sub-50 dip as strong stocks may dip to 51-54 in healthy pullbacks)
        rsi_now  = float(row["RSI"])
        rsi_prev = float(bars["RSI"].iloc[-2])
        if not (rsi_now >= RSI_RECOVERY_THRESH and rsi_prev < RSI_RECOVERY_THRESH):
            return None

        # 6. Strong bullish candle: up-day + close above open (intraday strength)
        if not (close > float(prev["Close"]) and close > float(row["Open"])):
            return None

        # 7. Volume
        if not (float(row["Volume"]) > VOL_MULTIPLIER * float(row["vol_avg_20"])):
            return None

        # 8. NIFTY market regime
        if USE_NIFTY_REGIME and not _nifty_above_ema200(bars.index[-1]):
            return None

        # Fixed target + hard stop → mean-reversion bounce capture
        hard_stop  = close - ATR_STOP_MULT * atr14
        target     = close + TARGET_ATR_MULT * atr14

        # R:R gate
        stop_dist   = close - hard_stop
        target_dist = target - close
        if stop_dist <= 0 or (target_dist / stop_dist) < MIN_RR_GATE:
            return None

        # Enhancement filters (HV rank, India VIX, VWAP) — active only when flags ON
        if not passes_enhancement_filters(row):
            return None

        # Half-Kelly sizing
        rr = target_dist / stop_dist
        kelly_raw     = WIN_PROB - (1 - WIN_PROB) / rr
        size_fraction = max(0.5, min(1.5, max(0.0, kelly_raw / 2) / 0.10))

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,                 # fixed 4×ATR target captures bounce
            trail_pct=0.0,                 # no trail — target+stop manage exits cleanly
            trail_activate_pct=0.0,
            confidence=70.0,
            size_fraction=size_fraction,
            metadata={
                "ema50":         round(ema50, 2),
                "atr14":         round(atr14, 2),
                "rsi":           round(rsi_now, 1),
                "kelly_raw":     round(kelly_raw, 3),
                "size_fraction": round(size_fraction, 2),
            },
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        # Pure target+stop strategy: hard_stop and target on Signal handle exits.
        # Only time stop remains to prevent infinite holds.
        if len(bars) < 15:
            return None
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= int(TIME_STOP_DAYS * 1.4):
            return ExitReason.TIME_STOP
        return None


def _compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
