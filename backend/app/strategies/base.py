"""
ASTRA Strategy Plugin Framework — Substrate
=============================================
Defines the abstract contract every trading strategy must implement.
Both ASTRA's first-party engines AND user-defined strategies share this contract.

A Strategy is stateless w.r.t. its own bars — it gets daily/intraday bars and must
emit signals + exits via pure functions of those bars. State (positions, cash) is
owned by the runner.

Backtest, paper-trading, and live execution all go through the same runner that
calls these methods identically. This is the substrate that makes the engine
track and user-strategy track share a single code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ── Enums ─────────────────────────────────────────────────────────────────

class SignalSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class ExitReason(str, Enum):
    TARGET     = "TARGET"
    HARD_STOP  = "HARD_STOP"
    TRAIL_STOP = "TRAIL_STOP"
    TIME_STOP  = "TIME_STOP"
    STRATEGY   = "STRATEGY"   # strategy-defined custom exit (e.g. SMA breakdown)
    EOD_FORCE  = "EOD_FORCE"  # forced exit at end of backtest window


class StrategyTimeframe(str, Enum):
    INTRADAY_15M = "15min"
    INTRADAY_1H  = "1h"
    DAILY        = "1d"
    WEEKLY       = "1wk"


# ── Data contracts ────────────────────────────────────────────────────────

@dataclass
class Signal:
    """Entry signal emitted by Strategy.signal()."""
    symbol:        str
    side:          SignalSide
    entry_price:   float
    timestamp:     pd.Timestamp
    hard_stop:     Optional[float] = None     # absolute price
    target:        Optional[float] = None
    trail_pct:     Optional[float] = None     # trailing stop %, activated after some gain
    trail_activate_pct: float = 0.0           # activate trail only after this unrealised gain
    confidence:    float = 50.0               # 0–100, optional
    size_fraction: float = 1.0               # Kelly-derived position scale (1.0 = full notional)
    metadata:      dict = field(default_factory=dict)


@dataclass
class Position:
    """A live position held by the runner."""
    symbol:        str
    side:          SignalSide
    entry_price:   float
    entry_ts:      pd.Timestamp
    quantity:      int
    hard_stop:     Optional[float]
    target:        Optional[float]
    trail_pct:     Optional[float]
    trail_activate_pct: float
    peak_price:    float                      # tracked by runner for trailing-stop
    metadata:      dict = field(default_factory=dict)


@dataclass
class Trade:
    """A closed trade in the backtest log."""
    symbol:       str
    side:         SignalSide
    entry_ts:     pd.Timestamp
    exit_ts:      pd.Timestamp
    entry_price:  float
    exit_price:   float
    peak_price:   float
    quantity:     int
    days_held:    int
    gross_pct:    float
    net_pct:      float          # after costs
    pnl_rs:       float
    exit_reason:  ExitReason
    strategy:     str            # strategy.name
    metadata:     dict = field(default_factory=dict)


@dataclass
class StrategyMeta:
    """Self-describing metadata for a strategy. Drives marketplace listing."""
    name:        str                              # canonical id, e.g. "ASTRA.MOMENTUM"
    version:     str                              # semver
    description: str
    timeframe:   StrategyTimeframe
    asset_class: str = "EQUITY"                   # EQUITY | OPTIONS | CRYPTO | COMMODITY
    long_only:   bool = True
    owner:       str = "ASTRA"                    # "ASTRA" or user_id
    tags:        list = field(default_factory=list)


@dataclass
class CostModel:
    """Realistic Indian-market cost model. Default = positional delivery on NSE."""
    slippage_per_side_pct: float = 0.0005   # 0.05%
    round_trip_pct:        float = 0.0025   # brokerage + STT + exch + GST + stamp


# ── Strategy ABC ──────────────────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Lifecycle (runner-driven):
        1. runner instantiates concrete Strategy
        2. runner asks strategy.universe() once per session
        3. for each timestamp t in chronological order:
             - if no position: runner calls strategy.signal(bars_up_to_t)
             - if    position: runner calls strategy.should_exit(position, bars_up_to_t)
        4. trade results stored in a Trade log
    """

    # ── Identity ──────────────────────────────────────────────────────────

    @abstractmethod
    def meta(self) -> StrategyMeta:
        """Self-describing metadata. Used by marketplace, evaluator, UI."""

    # ── Universe ──────────────────────────────────────────────────────────

    def universe(self) -> list[str]:
        """
        Symbols this strategy trades. Default = all NSE 200; override for narrower.
        Called once per session (cache-friendly).
        """
        from app.services.intraday_universe import INTRADAY_UNIVERSE
        return INTRADAY_UNIVERSE

    # ── Signal generation ─────────────────────────────────────────────────

    @abstractmethod
    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Decide whether to open a position. `bars` is OHLCV up to and INCLUDING
        the decision bar. Strategy must NOT look ahead.
        Return None to skip; return Signal to enter on next bar's open.
        """

    # ── Exit logic ────────────────────────────────────────────────────────

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        """
        Custom exit logic beyond the standard hard_stop / target / trail_stop.
        Default = never custom-exit; let the runner handle hard_stop/target/trail.
        Override for strategies with strategy-specific exits (e.g. Stage-2's SMA breakdown).
        Return None for "no custom exit"; return an ExitReason to exit at next bar open.
        """
        return None

    # ── Optional: pre-compute features once per symbol ────────────────────

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Optional: enrich the OHLCV frame with strategy-specific indicators
        ONCE per symbol (instead of recomputing every bar). Default = no-op.
        """
        return bars

    # ── Optional: pre-compute UNIVERSE-WIDE state ─────────────────────────

    def prepare_universe(self, universe_data: dict) -> None:
        """
        Optional: cross-sectional strategies override this to build universe-wide
        state (e.g. momentum ranks across all symbols on every date).
        Called ONCE per evaluator run, with {symbol: DataFrame} for all fetched symbols.
        Default = no-op (per-symbol strategies don't need this).
        """
        return None

    # ── Defaults ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.meta().name
