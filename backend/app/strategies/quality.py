"""
ASTRA.QUALITY v2.0 — Quality + Low-Volatility Factor
=====================================================
v2.0 changes:
  - TOP_PCT_ENTRY 0.10 → 0.20 (2× wider universe → more trades)
  - ADX < 35 filter: exclude high-volatility names that score well on recent Sharpe
  - ATR-based stop: 2×ATR(14) below entry  (replaces fixed -10%)
  - ATR-based target: 3×ATR(14) above entry (R:R 1.5 by construction)
  - Supertrend bearish flip as hard exit (don't hold through trend reversal)
  - Rank exit widened: top 25% → top 30%

Entry:
  - Close > SMA200
  - ADX < 35 (quality names are not momentum rockets)
  - Top 20% composite quality score (Sharpe, inv-vol, % positive months, max drawdown)

Exit (priority order):
  A. ATR target hit (3×ATR above entry) — runner handles via Signal.target
  B. Supertrend(10, 3.0) bearish flip
  C. Hard stop: entry − 2×ATR(14)      — runner handles via Signal.hard_stop
  D. Rank drops below top 30%
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    ExitReason, Position, Signal, SignalSide, Strategy, StrategyMeta, StrategyTimeframe
)
from app.strategies.indicators import compute_adx, compute_atr, compute_daily_vwap, compute_hv_rank, compute_supertrend
from app.strategies.filters import passes_enhancement_filters

# ── Parameters ────────────────────────────────────────────────────────────
LOOKBACK_DAYS         = 252
RISK_FREE_ANNUAL      = 0.07
TOP_PCT_ENTRY         = 0.20              # widened from 0.10
TOP_PCT_EXIT          = 0.30              # widened from 0.25
SMA_TREND             = 200

ADX_MAX_THRESHOLD     = 35               # exclude high-ADX volatile names
ATR_STOP_MULT         = 2.0
TARGET_ATR_MULT       = 3.0
SUPERTREND_PERIOD     = 10
SUPERTREND_MULT       = 3.0

# ── Fallback relaxation (apply if walk-forward yields < 30 trades) ────────
# TOP_PCT_ENTRY     = 0.25
# ADX_MAX_THRESHOLD = 40
# TOP_PCT_EXIT      = 0.35

_QUALITY_RANK_CACHE: dict = {}


class QualityStrategy(Strategy):
    """Quality + Low-Vol composite v2.0: wider entry, ATR targets, Supertrend exit."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.QUALITY",
            version="2.0.0",
            description=(
                "Quality + Low-Vol composite v2.0: top-20% entry, ATR stops/targets, "
                "Supertrend exit, ADX < 35 filter"
            ),
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["quality", "low-vol", "factor", "positional", "defensive"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.sort_index().copy()
        df["ret"]              = df["Close"].pct_change()
        df["SMA200"]           = df["Close"].rolling(SMA_TREND).mean()
        df["atr14"]            = compute_atr(df, 14)
        df["adx"]              = compute_adx(df, 14)["ADX"]
        df["supertrend_bullish"] = compute_supertrend(df, atr_period=SUPERTREND_PERIOD,
                                                       multiplier=SUPERTREND_MULT)
        df["hv_rank"] = compute_hv_rank(df["Close"])
        df["vwap"]    = compute_daily_vwap(df)
        return df

    def prepare_universe(self, universe_data: dict) -> None:
        clear_quality_cache()
        build_universe_quality_table(universe_data)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < LOOKBACK_DAYS + 30:
            return None
        row = bars.iloc[-1]
        for col in ("SMA200", "atr14", "adx"):
            if pd.isna(row.get(col)):
                return None

        close = float(row["Close"])
        if not (close > float(row["SMA200"])):
            return None

        # ADX filter — exclude volatile high-momentum names
        if float(row["adx"]) > ADX_MAX_THRESHOLD:
            return None

        date_key = bars.index[-1].date()
        rank_pct = _get_quality_rank(date_key, symbol)
        if rank_pct is None or rank_pct > TOP_PCT_ENTRY:
            return None

        atr14        = float(row["atr14"])
        stop_price   = close - ATR_STOP_MULT * atr14
        target_price = close + TARGET_ATR_MULT * atr14

        # Enhancement filters (HV rank, India VIX, VWAP) — active only when flags are ON
        if not passes_enhancement_filters(row):
            return None

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=stop_price,
            target=target_price,
            trail_pct=0.0,
            trail_activate_pct=0.0,
            confidence=75.0,
            metadata={
                "quality_rank_pct": rank_pct,
                "atr14": round(atr14, 2),
            },
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        if len(bars) < 15:
            return None
        row = bars.iloc[-1]

        # Supertrend bearish flip — don't hold through trend reversal
        if "supertrend_bullish" in row.index and not pd.isna(row["supertrend_bullish"]):
            if not bool(row["supertrend_bullish"]):
                return ExitReason.STRATEGY

        # Rank-based exit: dropped below top 30%
        if len(bars) >= LOOKBACK_DAYS + 30:
            date_key = bars.index[-1].date()
            rank_pct = _get_quality_rank(date_key, position.symbol)
            if rank_pct is not None and rank_pct > TOP_PCT_EXIT:
                return ExitReason.STRATEGY

        return None


# ── Quality score: 4-factor composite, cross-sectionally ranked ───────────

def build_universe_quality_table(universe_data: dict) -> None:
    global _QUALITY_RANK_CACHE
    syms = [s for s, df in universe_data.items() if df is not None and not df.empty]
    if not syms:
        return

    def _named(series, name):
        s = series.copy(); s.name = name; return s

    rets_wide = pd.concat(
        [_named(universe_data[s]["Close"].pct_change(), s) for s in syms],
        axis=1
    ).sort_index()

    annual_factor = np.sqrt(252)
    rf_daily = RISK_FREE_ANNUAL / 252

    mean_252 = rets_wide.rolling(LOOKBACK_DAYS).mean()
    std_252  = rets_wide.rolling(LOOKBACK_DAYS).std()
    sharpe   = (mean_252 - rf_daily) / std_252 * annual_factor

    annvol  = std_252 * annual_factor
    inv_vol = 1.0 / annvol.replace(0, np.nan)

    pos21   = (rets_wide.rolling(21).sum() > 0).astype(float)
    pos_pct = pos21.rolling(LOOKBACK_DAYS).mean()

    close_wide  = pd.concat(
        [_named(universe_data[s]["Close"], s) for s in syms], axis=1
    ).sort_index()
    rolling_max = close_wide.rolling(LOOKBACK_DAYS, min_periods=60).max()
    drawdown    = (close_wide - rolling_max) / rolling_max
    dd_score    = 1.0 + drawdown

    def _per_date_rank(df):
        return df.rank(axis=1, ascending=False, pct=True)

    composite = (
        _per_date_rank(sharpe) +
        _per_date_rank(inv_vol) +
        _per_date_rank(pos_pct) +
        _per_date_rank(dd_score)
    ) / 4.0
    composite_rank = _per_date_rank(composite)

    for date, row in composite_rank.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            continue
        _QUALITY_RANK_CACHE[date.date()] = valid.to_dict()


def _get_quality_rank(date_key, symbol: str) -> Optional[float]:
    cached = _QUALITY_RANK_CACHE.get(date_key)
    if cached is None:
        return None
    return cached.get(symbol)


def clear_quality_cache() -> None:
    _QUALITY_RANK_CACHE.clear()
