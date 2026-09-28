"""
ASTRA.MOMENTUM v3.1 — Cross-Sectional Momentum with Bigger Target
====================================================================
v3.2 fix (v3.1 R:R=1.46 — trail+rank exit firing before 5×ATR target reached):
  - REMOVED: trail stop (trail_pct=0) — was capping winners at ~8% before target
  - TOP_PCT_EXIT: 0.25 → 0.35 (let momentum run further before rank exit fires)
  - TARGET_ATR_MULT: 5.0 (unchanged from v3.1)
  - EMA_STACK_EXT_PCT: 0.10 (unchanged from v3.1)
  - Volume: 1.0× (unchanged from v3.1)

Core insight: v2.1 had 40% WR — the problem was exits cutting winners short.
With trail removed and rank exit widened, winners should run to 5×ATR more often.

Entry:
  - Symbol in top 10% momentum decile (6m skip-1 return)
  - ADX > 20 (trending, not choppy)
  - EMA20 > EMA50 (trend stack aligned — rising structure)
  - Close > EMA50 (price above medium-term trend)
  - Close ≤ EMA20 × 1.10 (not deeply extended)
  - Volume > 1.0× 20-day avg
  - Close > SMA200

Exit (first triggered):
  A. Target: entry + 5×ATR(14)   ← bigger target captures real momentum runs
  B. Rank drops below top 25%
  C. Hard stop: entry − 2×ATR(14)
  D. Trail: 8% from peak after +5% gain
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    ExitReason, Position, Signal, SignalSide, Strategy, StrategyMeta, StrategyTimeframe
)
from app.strategies.indicators import compute_adx, compute_atr, compute_daily_vwap, compute_ema, compute_hv_rank
from app.strategies.filters import passes_enhancement_filters

# ── Parameters ────────────────────────────────────────────────────────────────
LOOKBACK_DAYS         = 252
SKIP_DAYS             = 21
TOP_PCT_ENTRY         = 0.10
TOP_PCT_EXIT          = 0.35          # widened: let momentum run further before rank exit
SMA_TREND             = 200
VOL_MULTIPLIER        = 1.0          # lowered: rank filter is the quality gate

ADX_MIN_THRESHOLD     = 20           # trending filter
EMA_STACK_EXT_PCT     = 0.10         # slightly wider than v2.1 (was 0.05)
ATR_STOP_MULT         = 2.0          # stop below entry
TARGET_ATR_MULT       = 5.0          # bigger target: 5×ATR above entry (R:R = 2.5)

# ── Fallback relaxation (apply if walk-forward yields < 30 trades) ────────
# ADX_MIN_THRESHOLD = 15
# EMA_STACK_EXT_PCT = 0.15
# TOP_PCT_ENTRY     = 0.15

_MOMENTUM_RANK_CACHE: dict = {}


class MomentumStrategy(Strategy):
    """Cross-sectional momentum v3.0: Supertrend exit lets winners run."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.MOMENTUM",
            version="3.2.0",
            description=(
                "Cross-sectional momentum v3.1: 5×ATR target, wider EMA zone, "
                "rank-based exit, trail activates after +5%"
            ),
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["momentum", "factor", "positional", "supertrend"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.sort_index().copy()
        df["SMA"]        = df["Close"].rolling(SMA_TREND).mean()
        df["EMA20"]      = compute_ema(df["Close"], 20)
        df["EMA50"]      = compute_ema(df["Close"], 50)
        df["vol_avg_20"] = df["Volume"].rolling(20).mean()
        df["atr14"]      = compute_atr(df, 14)
        df["adx"]        = compute_adx(df, 14)["ADX"]
        df["mom_score"]  = df["Close"].shift(SKIP_DAYS) / df["Close"].shift(LOOKBACK_DAYS) - 1
        df["hv_rank"]    = compute_hv_rank(df["Close"])
        df["vwap"]       = compute_daily_vwap(df)
        return df

    def prepare_universe(self, universe_data: dict) -> None:
        clear_momentum_cache()
        build_universe_momentum_table(universe_data)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < LOOKBACK_DAYS + 30:
            return None
        row = bars.iloc[-1]
        for col in ("SMA", "EMA20", "EMA50", "vol_avg_20", "mom_score", "atr14", "adx"):
            if pd.isna(row.get(col)):
                return None

        date_key = bars.index[-1].date()
        rank_pct = _get_momentum_rank(date_key, symbol, float(row["mom_score"]))
        if rank_pct is None or rank_pct > TOP_PCT_ENTRY:
            return None

        close = float(row["Close"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        atr14 = float(row["atr14"])

        # ADX gate — trending market only
        if float(row["adx"]) < ADX_MIN_THRESHOLD:
            return None

        # EMA stack: EMA20 > EMA50 (rising structure confirmed)
        if not (ema20 > ema50):
            return None

        # Price must be above EMA50
        if not (close > ema50):
            return None

        # Price must not be extended more than EMA_STACK_EXT_PCT above EMA20
        if close > ema20 * (1 + EMA_STACK_EXT_PCT):
            return None

        # Basic trend & liquidity filters
        if not (close > float(row["SMA"])):
            return None
        if not (float(row["Volume"]) > VOL_MULTIPLIER * float(row["vol_avg_20"])):
            return None

        hard_stop    = close - ATR_STOP_MULT * atr14
        target_price = close + TARGET_ATR_MULT * atr14

        # Enhancement filters (HV rank, India VIX, VWAP) — active only when flags are ON
        if not passes_enhancement_filters(row):
            return None

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target_price,           # 5×ATR target — captures full momentum run
            trail_pct=0.0,                 # no trail — target+stop manage exits cleanly
            trail_activate_pct=0.0,
            confidence=72.0,
            metadata={
                "mom_rank_pct":   rank_pct,
                "ema20_at_entry": round(ema20, 2),
                "ema50_at_entry": round(ema50, 2),
                "atr14":          round(atr14, 2),
            },
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        """Exit if rank drops below top quartile."""
        if len(bars) < LOOKBACK_DAYS + 30:
            return None
        row = bars.iloc[-1]
        if pd.isna(row.get("mom_score")):
            return None
        date_key = bars.index[-1].date()
        rank_pct = _get_momentum_rank(date_key, position.symbol, float(row["mom_score"]))
        if rank_pct is None:
            return None
        if rank_pct > TOP_PCT_EXIT:
            return ExitReason.STRATEGY
        return None


# ── Universe momentum rank ─────────────────────────────────────────────────

def _get_momentum_rank(date_key, symbol: str, my_score: float) -> Optional[float]:
    cached = _MOMENTUM_RANK_CACHE.get(date_key)
    if cached is None:
        if my_score is None:
            return None
        return 0.05 if my_score > 0.2 else (0.5 if my_score > 0 else 0.9)
    return cached.get(symbol)


def build_universe_momentum_table(universe_data: dict) -> None:
    global _MOMENTUM_RANK_CACHE
    series_list = []
    syms = list(universe_data.keys())
    for sym in syms:
        df = universe_data[sym]
        if df is None or df.empty:
            continue
        s = df["Close"].copy(); s.name = sym; series_list.append(s)
    if not series_list:
        return
    wide = pd.concat(series_list, axis=1).sort_index().ffill(limit=3)
    mom = wide.shift(SKIP_DAYS) / wide.shift(LOOKBACK_DAYS) - 1
    for date, row in mom.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            continue
        ranks = valid.rank(ascending=False) / len(valid)
        _MOMENTUM_RANK_CACHE[date.date()] = ranks.to_dict()


def clear_momentum_cache() -> None:
    _MOMENTUM_RANK_CACHE.clear()
