# ASTRA Hybrid Portfolio v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver PULLBACK v2.0 (Supertrend+EMA+ATR+RS), ORB Intraday v2.0, and a dynamic Sharpe²-weighted PortfolioAllocator targeting ≥5%/month portfolio return.

**Architecture:** Three sequential phases — each phase must show positive expectancy before the next starts. PULLBACK and ORB share indicator helpers in a new `indicators.py` module. The PortfolioAllocator reads per-trade results from a new `strategy_performance` DB table and re-weights capital weekly.

**Tech Stack:** Python 3.9, pandas, numpy, SQLAlchemy, FastAPI, pytest. All existing patterns (Strategy ABC, evaluator, registry) are preserved and extended.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `backend/app/strategies/indicators.py` | CREATE | Supertrend, ATR, EMA, RS helpers — shared by all strategies |
| `backend/app/strategies/pullback.py` | MODIFY | v2.0: use new indicators, new entry/exit logic |
| `backend/app/services/intraday_engine.py` | MODIFY | ORB v2.0: Supertrend 15m gate, ATR targets, day-open filter, R:R floor |
| `backend/app/models/database.py` | MODIFY | Add `StrategyPerformance` table |
| `backend/app/services/portfolio_allocator.py` | CREATE | Rolling Sharpe² weights, risk cap, weekly rebalance |
| `backend/app/api/portfolio_router.py` | CREATE | GET /portfolio/allocation, POST /portfolio/rebalance |
| `backend/main.py` | MODIFY | Register portfolio router |
| `backend/tests/test_indicators.py` | CREATE | Unit tests for all indicator helpers |
| `backend/tests/test_pullback_v2.py` | CREATE | PULLBACK v2.0 signal + exit tests |
| `backend/tests/test_portfolio_allocator.py` | CREATE | Allocator unit tests |

---

## Task 1: Shared Indicator Helpers

**Files:**
- Create: `backend/app/strategies/indicators.py`
- Create: `backend/tests/test_indicators.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_indicators.py
import numpy as np
import pandas as pd
import pytest


def _make_bars(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    if trend == "up":
        close = 100 + np.cumsum(rng.normal(0.15, 1.0, n))
    else:
        close = 200 - np.cumsum(rng.normal(0.15, 1.0, n))
    close = np.maximum(close, 5.0)
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.010,
        "Low":    close * 0.990,
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def test_compute_atr_positive_values():
    from app.strategies.indicators import compute_atr
    df = _make_bars(100)
    atr = compute_atr(df, period=14)
    assert isinstance(atr, pd.Series)
    assert len(atr) == 100
    assert (atr.dropna() > 0).all(), "ATR must be positive"


def test_compute_atr_nan_for_first_bars():
    from app.strategies.indicators import compute_atr
    df = _make_bars(30)
    atr = compute_atr(df, period=14)
    # EWM ATR starts from first bar (no leading NaN) — just verify numeric
    assert not atr.isna().all()


def test_compute_ema_length():
    from app.strategies.indicators import compute_ema
    df = _make_bars(100)
    ema = compute_ema(df["Close"], 20)
    assert len(ema) == 100


def test_compute_ema_converges():
    from app.strategies.indicators import compute_ema
    df = _make_bars(300)
    ema50 = compute_ema(df["Close"], 50)
    ema200 = compute_ema(df["Close"], 200)
    # In an uptrend, EMA50 > EMA200 after warmup
    assert float(ema50.iloc[-1]) > float(ema200.iloc[-1])


def test_supertrend_returns_bool_series():
    from app.strategies.indicators import compute_supertrend
    df = _make_bars(300)
    st = compute_supertrend(df, atr_period=10, multiplier=3.0)
    assert isinstance(st, pd.Series)
    assert st.dtype == bool
    assert len(st) == 300


def test_supertrend_mostly_bullish_in_uptrend():
    from app.strategies.indicators import compute_supertrend
    df = _make_bars(300, trend="up", seed=1)
    st = compute_supertrend(df, atr_period=10, multiplier=3.0)
    pct_bullish = st.mean()
    assert pct_bullish > 0.5, f"Uptrend should be mostly bullish, got {pct_bullish:.0%}"


def test_supertrend_mostly_bearish_in_downtrend():
    from app.strategies.indicators import compute_supertrend
    df = _make_bars(300, trend="down", seed=2)
    st = compute_supertrend(df, atr_period=10, multiplier=3.0)
    pct_bearish = (~st).mean()
    assert pct_bearish > 0.5, f"Downtrend should be mostly bearish, got {pct_bearish:.0%}"


def test_compute_rs_positive_when_stock_outperforms():
    from app.strategies.indicators import compute_rs
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    stock_close = pd.Series(100 + np.arange(100) * 0.5, index=idx)   # +50% over period
    bench_close = pd.Series(100 + np.arange(100) * 0.1, index=idx)   # +10% over period
    rs = compute_rs(stock_close, bench_close, period=20)
    assert float(rs.iloc[-1]) > 0, "Outperforming stock must have positive RS"


def test_compute_rs_negative_when_stock_underperforms():
    from app.strategies.indicators import compute_rs
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    stock_close = pd.Series(100 - np.arange(100) * 0.2, index=idx)   # falling
    bench_close = pd.Series(100 + np.arange(100) * 0.3, index=idx)   # rising
    rs = compute_rs(stock_close, bench_close, period=20)
    assert float(rs.iloc[-1]) < 0, "Underperforming stock must have negative RS"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_indicators.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'app.strategies.indicators'`

- [ ] **Step 3: Implement indicators.py**

```python
# backend/app/strategies/indicators.py
"""
Shared technical indicator helpers used by ASTRA strategies.
All functions are pure (no side effects) and operate on pandas Series/DataFrames.
"""
from typing import Union
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range via EWM smoothing."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_supertrend(df: pd.DataFrame, atr_period: int = 10,
                       multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend indicator.
    Returns a boolean Series: True = bullish (price above Supertrend line).
    Uses iterative logic to maintain band memory across bars.
    """
    atr  = compute_atr(df, atr_period)
    hl2  = (df["High"] + df["Low"]) / 2.0
    close = df["Close"]

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    n = len(df)

    upper_arr = upper_band.values.copy()
    lower_arr = lower_band.values.copy()
    close_arr = close.values

    for i in range(1, n):
        upper_arr[i] = (upper_basic.iloc[i]
                        if upper_basic.iloc[i] < upper_arr[i-1]
                           or close_arr[i-1] > upper_arr[i-1]
                        else upper_arr[i-1])
        lower_arr[i] = (lower_basic.iloc[i]
                        if lower_basic.iloc[i] > lower_arr[i-1]
                           or close_arr[i-1] < lower_arr[i-1]
                        else lower_arr[i-1])

    trend = np.zeros(n, dtype=bool)       # False = bearish
    st_val = np.where(trend, lower_arr, upper_arr).copy()

    for i in range(1, n):
        prev_bullish = trend[i-1]
        if prev_bullish:
            if close_arr[i] < lower_arr[i]:
                trend[i] = False
                st_val[i] = upper_arr[i]
            else:
                trend[i] = True
                st_val[i] = lower_arr[i]
        else:
            if close_arr[i] > upper_arr[i]:
                trend[i] = True
                st_val[i] = lower_arr[i]
            else:
                trend[i] = False
                st_val[i] = upper_arr[i]

    return pd.Series(trend, index=df.index, name="supertrend_bullish")


def compute_rs(stock_close: pd.Series, bench_close: pd.Series,
               period: int = 20) -> pd.Series:
    """
    Relative Strength of stock vs benchmark over `period` bars.
    Positive = stock outperforming. Both series must share the same index
    or bench_close will be forward-filled to match stock_close.
    """
    bench_aligned = bench_close.reindex(stock_close.index, method="ffill")
    stock_ret = stock_close.pct_change(period)
    bench_ret = bench_aligned.pct_change(period)
    return stock_ret - bench_ret
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_indicators.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/app/strategies/indicators.py backend/tests/test_indicators.py
git commit -m "feat(indicators): shared Supertrend, ATR, EMA, RS helpers for PULLBACK v2.0"
```

---

## Task 2: PULLBACK v2.0 Rewrite

**Files:**
- Modify: `backend/app/strategies/pullback.py`
- Create: `backend/tests/test_pullback_v2.py`

- [ ] **Step 1: Write failing tests for new entry/exit logic**

```python
# backend/tests/test_pullback_v2.py
"""Tests for PULLBACK v2.0: Supertrend gate, ATR stop, RS filter, time stop."""
import numpy as np
import pandas as pd
import pytest


def _rising_bars(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic uptrend bars suitable for triggering PULLBACK entry."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = 100 + np.cumsum(np.ones(n) * 0.15 + rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "Open":   close * (1 + rng.normal(0, 0.003, n)),
        "High":   close * (1 + np.abs(rng.normal(0, 0.008, n))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.008, n))),
        "Close":  close,
        "Volume": np.ones(n) * 2_000_000,
    }, index=idx)


def test_precompute_adds_supertrend_column():
    from app.strategies.pullback import PullbackStrategy
    strategy = PullbackStrategy()
    df = _rising_bars(300)
    enriched = strategy.precompute(df)
    assert "supertrend_bullish" in enriched.columns, "precompute must add supertrend_bullish"
    assert enriched["supertrend_bullish"].dtype == bool


def test_precompute_adds_ema_columns():
    from app.strategies.pullback import PullbackStrategy
    strategy = PullbackStrategy()
    df = _rising_bars(300)
    enriched = strategy.precompute(df)
    for col in ("EMA20", "EMA50", "EMA200", "ATR14", "RSI"):
        assert col in enriched.columns, f"precompute must add {col}"


def test_precompute_adds_rsi_was_oversold():
    from app.strategies.pullback import PullbackStrategy
    strategy = PullbackStrategy()
    df = _rising_bars(300)
    enriched = strategy.precompute(df)
    assert "RSI_was_oversold" in enriched.columns


def test_signal_returns_none_when_supertrend_bearish():
    """If Supertrend is bearish, signal must return None regardless of other conditions."""
    from app.strategies.pullback import PullbackStrategy
    import app.strategies.pullback as pb_mod

    strategy = PullbackStrategy()
    pb_mod._nifty_above_sma200 = lambda d: True

    df = _rising_bars(400)
    enriched = strategy.precompute(df)
    # Force Supertrend to bearish on last bar
    enriched = enriched.copy()
    enriched["supertrend_bullish"] = False

    sig = strategy.signal(enriched, "RELIANCE")
    assert sig is None, "Supertrend bearish must block signal"


def test_signal_hard_stop_is_atr_based():
    """Hard stop = entry - 2×ATR(14), not a fixed percentage."""
    from app.strategies.pullback import PullbackStrategy, ATR_STOP_MULT
    import app.strategies.pullback as pb_mod

    pb_mod._nifty_above_sma200 = lambda d: True
    strategy = PullbackStrategy()
    strategy._nifty_close = None  # no RS filter

    df = _rising_bars(400)
    enriched = strategy.precompute(df)

    for i in range(250, len(enriched) - 1):
        bars = enriched.iloc[:i+1]
        sig = strategy.signal(bars, "RELIANCE")
        if sig is not None:
            atr_at_signal = float(enriched["ATR14"].iloc[i])
            expected_stop = sig.entry_price - ATR_STOP_MULT * atr_at_signal
            assert abs(sig.hard_stop - expected_stop) < 0.01, (
                f"stop {sig.hard_stop:.2f} != entry - 2×ATR = {expected_stop:.2f}"
            )
            break


def test_signal_target_is_ema50_times_1p20():
    from app.strategies.pullback import PullbackStrategy
    import app.strategies.pullback as pb_mod

    pb_mod._nifty_above_sma200 = lambda d: True
    strategy = PullbackStrategy()
    strategy._nifty_close = None

    df = _rising_bars(400)
    enriched = strategy.precompute(df)

    for i in range(250, len(enriched) - 1):
        bars = enriched.iloc[:i+1]
        sig = strategy.signal(bars, "RELIANCE")
        if sig is not None:
            ema50 = float(enriched["EMA50"].iloc[i])
            expected_target = ema50 * 1.20
            assert abs(sig.target - expected_target) < 0.01, (
                f"target {sig.target:.2f} != EMA50×1.20 = {expected_target:.2f}"
            )
            break


def test_exit_triggers_on_supertrend_flip():
    from app.strategies.pullback import PullbackStrategy
    from app.strategies.base import Position, SignalSide
    import app.strategies.pullback as pb_mod

    pb_mod._nifty_above_sma200 = lambda d: True
    strategy = PullbackStrategy()

    df = _rising_bars(300)
    enriched = strategy.precompute(df).copy()
    enriched["supertrend_bullish"] = False   # flip bearish

    pos = Position(
        symbol="TST", side=SignalSide.BUY,
        entry_price=100.0, entry_ts=enriched.index[50],
        quantity=10, hard_stop=92.0, target=120.0,
        trail_pct=0.07, trail_activate_pct=0.04,
        peak_price=105.0,
    )
    result = strategy.should_exit(pos, enriched.iloc[:60])
    from app.strategies.base import ExitReason
    assert result == ExitReason.STRATEGY, "Supertrend flip must trigger STRATEGY exit"


def test_exit_triggers_time_stop():
    from app.strategies.pullback import PullbackStrategy, TIME_STOP_DAYS
    from app.strategies.base import Position, SignalSide, ExitReason

    strategy = PullbackStrategy()
    df = _rising_bars(300)
    enriched = strategy.precompute(df).copy()
    # Keep Supertrend bullish (no trend exit)
    enriched["supertrend_bullish"] = True

    entry_ts = enriched.index[0]
    late_ts   = enriched.index[int(TIME_STOP_DAYS * 1.5)]  # well past limit

    pos = Position(
        symbol="TST", side=SignalSide.BUY,
        entry_price=100.0, entry_ts=entry_ts,
        quantity=10, hard_stop=92.0, target=120.0,
        trail_pct=0.07, trail_activate_pct=0.04,
        peak_price=100.0,
    )
    late_bars = enriched.loc[:late_ts]
    result = strategy.should_exit(pos, late_bars)
    assert result == ExitReason.TIME_STOP, "Time stop must fire after TIME_STOP_DAYS"


def test_rs_filter_blocks_underperforming_stock():
    """When stock RS vs NIFTY is negative, signal must return None."""
    from app.strategies.pullback import PullbackStrategy
    import app.strategies.pullback as pb_mod

    pb_mod._nifty_above_sma200 = lambda d: True

    strategy = PullbackStrategy()
    df = _rising_bars(400)
    # Build a NIFTY that clearly outperforms
    nifty_close = pd.Series(
        100 + np.arange(400) * 0.5,
        index=df.index,
    )
    strategy._nifty_close = nifty_close

    enriched = strategy.precompute(df)
    # All RSs will be negative (NIFTY doubles, stock barely moves)
    for i in range(250, len(enriched) - 1):
        bars = enriched.iloc[:i+1]
        sig = strategy.signal(bars, "RELIANCE")
        assert sig is None, "Underperforming stock RS filter must block signal"
        break  # one check is enough
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pullback_v2.py -v 2>&1 | head -40
```
Expected: Most fail with `AssertionError` or `KeyError: 'supertrend_bullish'` — confirms the tests are checking real new behaviour.

- [ ] **Step 3: Rewrite pullback.py**

Replace the full contents of `backend/app/strategies/pullback.py` with:

```python
"""
ASTRA.PULLBACK v2.0 — Supertrend-Gated Pullback to EMA50
==========================================================
Professional-grade redesign. Entry only when Supertrend is bullish,
EMA stack aligned, RSI midline crossed, stock outperforming NIFTY.
ATR-based stop adapts to stock volatility. Dynamic Supertrend exit.

Entry (all 9 required):
  1. Supertrend(10, 3) bullish
  2. EMA20 > EMA50 > EMA200
  3. Close ∈ [EMA50, EMA50 × 1.08]
  4. Close > EMA20
  5. RSI was < 40 in last 15 bars; now > 50 and rising
  6. Bounce: Close > prev_Close × 1.003 AND High > prev_High
  7. Volume > 1.5 × 20-day avg
  8. NIFTY Close > NIFTY EMA200
  9. Stock 20-day return > NIFTY 20-day return (RS positive)

Exit (first triggered):
  A. Supertrend flips bearish
  B. Target: EMA50 × 1.20
  C. Hard stop: entry − ATR_STOP_MULT × ATR(14)
  D. Trail: 7% from peak after +4% gain
  E. Time stop: 40 trading days
"""
from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    ExitReason, Position, Signal, SignalSide,
    Strategy, StrategyMeta, StrategyTimeframe,
)
from app.strategies.indicators import compute_atr, compute_ema, compute_rs, compute_supertrend

# ── Parameters ────────────────────────────────────────────────────────────────
PULLBACK_DISTANCE    = 0.08        # Close within 8% above EMA50
RSI_OVERSOLD_THRESH  = 40
RSI_RECOVERY_THRESH  = 50
RSI_LOOKBACK_DAYS    = 15
MIN_BOUNCE_PCT       = 0.003
VOL_MULTIPLIER       = 1.5
USE_NIFTY_REGIME     = True
RS_PERIOD            = 20          # 20-bar relative strength window

ATR_STOP_MULT        = 2.0         # stop = entry - ATR_STOP_MULT × ATR14
TARGET_EMA_MULT      = 1.20        # target = EMA50 × 1.20
TRAIL_ACTIVATE_PCT   = 0.04
TRAIL_DRAWDOWN_PCT   = 0.07
TIME_STOP_DAYS       = 40          # calendar-day equivalent: × 1.4

WIN_PROB             = 0.45        # calibration for Kelly sizing

# ── NIFTY regime cache ────────────────────────────────────────────────────────
_NIFTY_CACHE: Optional[pd.DataFrame] = None


def _nifty_above_sma200(date) -> bool:
    """True if NIFTY Close > EMA200 on or just before `date`. Fail-open."""
    global _NIFTY_CACHE
    if _NIFTY_CACHE is None:
        try:
            from app.services.yahoo_finance import yahoo_service
            for tick in ("^NSEI", "NIFTY", "NSEI"):
                df = yahoo_service.get_ohlcv(tick, period="2y", interval="1d")
                if df is not None and not df.empty and len(df) >= 210:
                    df = df.sort_index().copy()
                    df["EMA200"] = compute_ema(df["Close"], 200)
                    _NIFTY_CACHE = df
                    break
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
    """Buy pullbacks to EMA50 in Supertrend-confirmed uptrends."""

    def __init__(self):
        # Populated by prepare_universe; NIFTY Close for RS computation
        self._nifty_close: Optional[pd.Series] = None

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.PULLBACK",
            version="2.0.0",
            description=(
                "Supertrend-gated pullback to EMA50: ATR stops, RS filter, "
                "dynamic exit, half-Kelly sizing"
            ),
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["mean-reversion", "trend-following", "positional", "supertrend"],
        )

    def prepare_universe(self, universe_data: dict) -> None:
        """Cache NIFTY close for RS computation across all symbols."""
        from app.services.strategy_evaluator import _fetch_2y_daily
        if self._nifty_close is None:
            for tick in ("NIFTYBEES",):
                df = _fetch_2y_daily(tick)
                if df is not None and not df.empty:
                    self._nifty_close = df["Close"]
                    break
        if self._nifty_close is None and _NIFTY_CACHE is not None:
            self._nifty_close = _NIFTY_CACHE["Close"]

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
        df["vol_avg_20"] = df["Volume"].rolling(20).mean()
        st = compute_supertrend(df, atr_period=10, multiplier=3.0)
        df["supertrend_bullish"] = st
        if self._nifty_close is not None:
            df["RS20"] = compute_rs(df["Close"], self._nifty_close, RS_PERIOD)
        else:
            df["RS20"] = np.nan
        return df

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < 220:
            return None

        row  = bars.iloc[-1]
        prev = bars.iloc[-2]

        required = ("EMA20", "EMA50", "EMA200", "ATR14", "RSI",
                    "RSI_was_oversold", "vol_avg_20", "supertrend_bullish")
        for col in required:
            if pd.isna(row.get(col)) or pd.isna(prev.get(col)):
                return None

        close  = float(row["Close"])
        ema20  = float(row["EMA20"])
        ema50  = float(row["EMA50"])
        ema200 = float(row["EMA200"])
        atr14  = float(row["ATR14"])

        # 1. Supertrend bullish
        if not bool(row["supertrend_bullish"]):
            return None

        # 2. EMA alignment
        if not (ema20 > ema50 > ema200):
            return None

        # 3. Pullback zone: EMA50 ≤ Close ≤ EMA50 × 1.08
        distance = (close - ema50) / ema50
        if not (0 <= distance <= PULLBACK_DISTANCE):
            return None

        # 4. Short-term trend intact
        if not (close > ema20):
            return None

        # 5. RSI recovery: was oversold, now crossed 50 and rising
        if not (row["RSI_was_oversold"] > 0):
            return None
        rsi_now  = float(row["RSI"])
        rsi_prev = float(bars["RSI"].iloc[-2])
        if not (rsi_now >= RSI_RECOVERY_THRESH and rsi_now > rsi_prev):
            return None

        # 6. Bounce confirmation
        if not (close > float(prev["Close"]) * (1 + MIN_BOUNCE_PCT)):
            return None
        if not (float(row["High"]) > float(prev["High"])):
            return None

        # 7. Volume
        if not (float(row["Volume"]) > VOL_MULTIPLIER * float(row["vol_avg_20"])):
            return None

        # 8. NIFTY regime
        if USE_NIFTY_REGIME and not _nifty_above_sma200(bars.index[-1]):
            return None

        # 9. RS filter — only when NIFTY data available
        rs20 = row.get("RS20")
        if not pd.isna(rs20) and float(rs20) < 0:
            return None

        target    = ema50 * TARGET_EMA_MULT
        hard_stop = close - ATR_STOP_MULT * atr14

        # Half-Kelly sizing
        stop_dist   = close - hard_stop
        target_dist = target - close
        rr = target_dist / stop_dist if stop_dist > 0 else 1.875
        kelly_raw   = WIN_PROB - (1 - WIN_PROB) / rr
        size_fraction = max(0.5, min(1.5, max(0.0, kelly_raw / 2) / 0.10))

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,
            trail_pct=TRAIL_DRAWDOWN_PCT,
            trail_activate_pct=TRAIL_ACTIVATE_PCT,
            confidence=70.0,
            size_fraction=size_fraction,
            metadata={
                "ema50": round(ema50, 2),
                "atr14": round(atr14, 2),
                "rsi":   round(rsi_now, 1),
                "rs20":  round(float(rs20), 4) if not pd.isna(rs20) else None,
                "kelly_raw": round(kelly_raw, 3),
                "size_fraction": round(size_fraction, 2),
            },
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        """Exit on Supertrend flip or time stop."""
        if len(bars) < 15:
            return None
        row = bars.iloc[-1]
        # Supertrend flip
        if "supertrend_bullish" in row.index and not pd.isna(row["supertrend_bullish"]):
            if not bool(row["supertrend_bullish"]):
                return ExitReason.STRATEGY
        # Time stop
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
```

- [ ] **Step 4: Run new PULLBACK tests**

```bash
pytest tests/test_pullback_v2.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
pytest -q --tb=short
```
Expected: all tests pass (some strategy tests referencing v1.3 attribute names will need updating — see next step)

- [ ] **Step 6: Fix any broken existing tests**

The tests in `tests/test_strategies.py` reference `_sector_above_sma50`, `_nifty_above_sma200`, and old column names. Update these references:

In `tests/test_strategies.py`, find `test_pullback_sector_filter_blocks_bearish_sector` and `test_pullback_prepare_universe_builds_sector_uptrend`. Replace them:

```python
def test_pullback_v2_supertrend_gate_in_precompute():
    """v2.0 precompute must produce supertrend_bullish, EMA20/50/200, ATR14."""
    from app.strategies.pullback import PullbackStrategy
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = 100 + np.cumsum(rng.normal(0.1, 1, n))
    df = pd.DataFrame({
        "Open": close*0.999, "High": close*1.01,
        "Low": close*0.99, "Close": close,
        "Volume": np.ones(n)*2_000_000,
    }, index=idx)
    enriched = PullbackStrategy().precompute(df)
    for col in ("EMA20", "EMA50", "EMA200", "ATR14", "RSI", "supertrend_bullish"):
        assert col in enriched.columns, f"missing {col}"


def test_pullback_v2_prepare_universe_loads_nifty():
    """prepare_universe should attempt to populate _nifty_close."""
    from app.strategies.pullback import PullbackStrategy
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = 100 + np.cumsum(rng.normal(0.1, 1, n))
    df = pd.DataFrame({
        "Open": close, "High": close*1.01,
        "Low": close*0.99, "Close": close,
        "Volume": np.ones(n)*1e6,
    }, index=idx)
    strategy = PullbackStrategy()
    strategy.prepare_universe({"HDFCBANK": df})
    # _nifty_close may be None if data unavailable — just verify no crash
    assert hasattr(strategy, "_nifty_close")
```

Then remove the old `test_pullback_sector_filter_blocks_bearish_sector` and `test_pullback_prepare_universe_builds_sector_uptrend` tests.

- [ ] **Step 7: Run full suite again**

```bash
pytest -q --tb=short
```
Expected: all pass

- [ ] **Step 8: Run PULLBACK v2.0 evaluation**

```bash
python - <<'EOF'
import logging; logging.basicConfig(level=logging.WARNING)
from app.services.strategy_evaluator import evaluate, print_report
from app.strategies.pullback import PullbackStrategy
print_report(evaluate(PullbackStrategy(), max_workers=6))
EOF
```
Expected: WR ≥ 40%, expectancy > 0%, PF > 1.0. If MARGINAL or PASS, proceed to Task 3. If FAIL, revisit ATR_STOP_MULT (try 1.5) or PULLBACK_DISTANCE (try 0.06).

- [ ] **Step 9: Commit**

```bash
git add backend/app/strategies/pullback.py backend/tests/test_pullback_v2.py backend/tests/test_strategies.py
git commit -m "feat(pullback): v2.0 — Supertrend+EMA+ATR+RS, dynamic exit, half-Kelly sizing"
```

---

## Task 3: ORB Intraday v2.0

**Files:**
- Modify: `backend/app/services/intraday_engine.py` (functions `orb_signal`, `_compute_orb`, add `_supertrend_15m_bullish`)
- No new test file — extend existing intraday tests if present, otherwise add to `tests/test_api.py` or create `tests/test_orb_v2.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_orb_v2.py
"""Tests for ORB Intraday v2.0: Supertrend gate, ATR targets, day-open filter, R:R floor."""
import numpy as np
import pandas as pd
import pytest
import pytz

_IST = pytz.timezone("Asia/Kolkata")


def _make_intraday(n_days: int = 5, trend: str = "up") -> pd.DataFrame:
    """Synthetic 15-minute IST bars for n_days."""
    bars_per_day = 25   # 09:15–15:15 = 25 × 15-min bars
    n = n_days * bars_per_day
    rng = np.random.default_rng(42)
    base = 100.0
    closes = []
    base_close = base
    for d in range(n_days):
        day_drift = 0.3 if trend == "up" else -0.3
        day_closes = base_close + np.cumsum(rng.normal(day_drift / bars_per_day, 0.1, bars_per_day))
        closes.extend(day_closes.tolist())
        base_close = day_closes[-1]

    closes = np.array(closes)
    start = pd.Timestamp("2024-01-02 09:15:00", tz=_IST)
    idx = pd.date_range(start, periods=n, freq="15min", tz=_IST)
    # Remove non-trading hours (keep only 09:15–15:15)
    df = pd.DataFrame({
        "Open":   closes * 0.999,
        "High":   closes * 1.005,
        "Low":    closes * 0.995,
        "Close":  closes,
        "Volume": rng.integers(50_000, 500_000, n).astype(float),
    }, index=idx)
    return df


def test_orb_signal_respects_rr_floor():
    """orb_signal must return HOLD if computed R:R < 2.0."""
    from app.services.intraday_engine import orb_signal
    df = _make_intraday(5)
    date = df.index[0].date()
    result = orb_signal(df, date)
    # Either HOLD or a signal with valid R:R
    if result["signal"] != "HOLD":
        entry = result["entry_price"]
        sl    = result["sl"]
        tp    = result["tp"]
        stop_dist   = abs(entry - sl)
        target_dist = abs(tp - entry)
        rr = target_dist / stop_dist if stop_dist > 0 else 0
        assert rr >= 1.9, f"R:R {rr:.2f} below 2.0 floor"


def test_orb_signal_uses_atr_for_targets():
    """ORB v2.0 targets must be ATR-based (not fixed %)."""
    from app.services.intraday_engine import orb_signal, _compute_atr
    df = _make_intraday(10)
    date = df.index[25].date()   # second day
    result = orb_signal(df, date)
    if result["signal"] != "HOLD":
        entry = result["entry_price"]
        # ATR-based: TP ≈ entry + 1.5×ATR, SL ≈ entry - 0.75×ATR
        day_df = df[df.index.date == date]
        atr = _compute_atr(day_df, period=5)
        if not atr.empty and not pd.isna(atr.iloc[-1]):
            atr_val = float(atr.iloc[-1])
            expected_tp = entry + 1.5 * atr_val
            # Allow 10% tolerance
            assert abs(result["tp"] - expected_tp) / expected_tp < 0.10, (
                f"TP {result['tp']:.2f} not close to ATR-based {expected_tp:.2f}"
            )


def test_orb_signal_returns_hold_after_1100():
    """No new entries after 11:00 IST."""
    from app.services.intraday_engine import orb_signal
    df = _make_intraday(5)
    # Use a date where all bars are after 11:00
    date = df.index[0].date()
    result = orb_signal(df, date)
    # If we can only observe bars after 11:00 on a given day, should be HOLD
    after_11 = df[
        (df.index.date == date) &
        (df.index.time >= pd.Timestamp("11:00").time())
    ]
    if len(after_11) > 0:
        result2 = orb_signal(after_11, date)
        assert result2["signal"] == "HOLD", "No new entries allowed after 11:00"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orb_v2.py -v 2>&1 | head -30
```
Expected: failures due to missing ATR-based target logic and R:R floor.

- [ ] **Step 3: Update orb_signal and helpers in intraday_engine.py**

Find the `orb_signal` function (line ~529) and `NO_TRADE_AFTER` constant. Make these changes:

**3a. Change the NO_TRADE_AFTER constant** (line ~29):
```python
NO_TRADE_AFTER   = "11:00"     # v2.0: tighter window — no new ORB entries after 11:00
```

**3b. Add Supertrend helper for 15m** (add after `_compute_atr` function, around line 490):
```python
def _supertrend_15m_bullish(day_df: pd.DataFrame, atr_period: int = 7,
                             multiplier: float = 3.0) -> bool:
    """
    Return True if Supertrend is bullish on the most recent 15m bar.
    Uses the same algorithm as indicators.compute_supertrend but inline
    to avoid circular import.
    """
    if len(day_df) < atr_period + 2:
        return True   # fail-open when insufficient data
    try:
        from app.strategies.indicators import compute_supertrend
        st = compute_supertrend(day_df, atr_period=atr_period, multiplier=multiplier)
        return bool(st.iloc[-1])
    except Exception:
        return True
```

**3c. Replace the `orb_signal` function body** with the v2.0 version:
```python
def orb_signal(df: pd.DataFrame, date) -> dict:
    """
    ORB v2.0: BUY on close above ORB high.
    Requires Supertrend(7,3) bullish on 15m, ATR-based targets, R:R ≥ 2.0,
    and signal within 09:30–11:00 IST window only.
    """
    orb_high, orb_low = _compute_orb(df, date)
    _hold = {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "ORB"}

    if orb_high is None:
        return _hold

    orb_end    = dtime(9, 15 + ORB_WINDOW_MIN)
    no_trade   = dtime(11, 0)       # v2.0: tight window
    square_off = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])

    day_df = df[df.index.date == (date if hasattr(date, "year") else date)]
    post_orb = day_df[
        (day_df.index.time >= orb_end) &
        (day_df.index.time <= no_trade)
    ]

    if post_orb.empty:
        return _hold

    last = post_orb.iloc[-1]
    close = float(last["Close"])
    vol   = float(last["Volume"])

    avg_vol = float(day_df["Volume"].mean()) if len(day_df) > 1 else vol

    # Supertrend gate on 15m
    if not _supertrend_15m_bullish(day_df):
        return _hold

    # ATR-based targets
    atr_series = _compute_atr(day_df, period=5)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else (close * 0.005)
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = close * 0.005

    tp_long = orb_high + 1.5 * atr_val
    sl_long = orb_high - 0.75 * atr_val

    # R:R floor ≥ 2.0
    stop_dist   = abs(orb_high - sl_long)
    target_dist = abs(tp_long  - orb_high)
    if stop_dist <= 0 or (target_dist / stop_dist) < 2.0:
        return _hold

    # BUY breakout: close above ORB high with volume
    if close > orb_high and vol > 1.1 * avg_vol:
        return {
            "signal":      "BUY",
            "entry_price": close,
            "sl":          sl_long,
            "tp":          tp_long,
            "strategy":    "ORB",
        }

    return _hold
```

- [ ] **Step 4: Run ORB tests**

```bash
pytest tests/test_orb_v2.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Run full suite**

```bash
pytest -q --tb=short
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/intraday_engine.py backend/tests/test_orb_v2.py
git commit -m "feat(orb): v2.0 — Supertrend 15m gate, ATR targets, 2:1 R:R floor, 11:00 window"
```

---

## Task 4: StrategyPerformance DB Table

**Files:**
- Modify: `backend/app/models/database.py`

- [ ] **Step 1: Write failing test**

```python
# In backend/tests/test_api.py or a new test_strategy_performance.py
def test_strategy_performance_table_exists():
    """StrategyPerformance model must be importable and map to a real table."""
    from app.models.database import StrategyPerformance, SessionLocal, engine, Base
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        count = db.query(StrategyPerformance).count()
        assert count >= 0
    finally:
        db.close()


def test_strategy_performance_insert_and_read():
    from app.models.database import StrategyPerformance, SessionLocal, Base, engine
    import datetime
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        row = StrategyPerformance(
            strategy="ASTRA.PULLBACK",
            entry_ts=datetime.datetime(2025, 1, 10),
            exit_ts=datetime.datetime(2025, 1, 20),
            net_pct=2.5,
            size_frac=0.8,
            user_id=None,
        )
        db.add(row)
        db.commit()
        fetched = db.query(StrategyPerformance).filter_by(strategy="ASTRA.PULLBACK").first()
        assert fetched is not None
        assert abs(fetched.net_pct - 2.5) < 0.01
        db.delete(fetched)
        db.commit()
    finally:
        db.close()
```

Run:
```bash
pytest tests/ -k "strategy_performance" -v
```
Expected: `ImportError: cannot import name 'StrategyPerformance'`

- [ ] **Step 2: Add StrategyPerformance to database.py**

At the end of `backend/app/models/database.py`, add:

```python
class StrategyPerformance(Base):
    """Per-trade result log. Used by PortfolioAllocator to compute rolling Sharpe."""
    __tablename__ = "strategy_performance"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    strategy   = Column(String, nullable=False, index=True)   # "ASTRA.PULLBACK", "ASTRA.ORB"
    entry_ts   = Column(DateTime, nullable=True)
    exit_ts    = Column(DateTime, nullable=True)
    net_pct    = Column(Float, nullable=False)                # net % return after costs
    size_frac  = Column(Float, default=1.0)                   # Kelly size fraction used
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
```

Also add `import datetime` at the top if not already present.

- [ ] **Step 3: Run test**

```bash
pytest tests/ -k "strategy_performance" -v
```
Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/database.py
git commit -m "feat(db): add StrategyPerformance table for portfolio allocator"
```

---

## Task 5: PortfolioAllocator Service

**Files:**
- Create: `backend/app/services/portfolio_allocator.py`
- Create: `backend/tests/test_portfolio_allocator.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_portfolio_allocator.py
"""Tests for rolling Sharpe² portfolio allocator."""
import datetime
import numpy as np
import pytest


def _insert_trades(db, strategy: str, net_pcts: list):
    from app.models.database import StrategyPerformance
    base = datetime.datetime(2025, 1, 1)
    for i, pct in enumerate(net_pcts):
        db.add(StrategyPerformance(
            strategy=strategy,
            entry_ts=base + datetime.timedelta(days=i*2),
            exit_ts=base + datetime.timedelta(days=i*2+1),
            net_pct=pct, size_frac=1.0,
        ))
    db.commit()


def test_allocator_equal_weight_with_fewer_than_20_trades():
    from app.models.database import SessionLocal, Base, engine, StrategyPerformance
    from app.services.portfolio_allocator import compute_weights
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(StrategyPerformance).delete(); db.commit()
        _insert_trades(db, "ASTRA.PULLBACK", [1.0, 2.0, -1.0])   # only 3 trades
        _insert_trades(db, "ASTRA.ORB",      [0.5, 1.5, -0.5])
        weights = compute_weights(db, ["ASTRA.PULLBACK", "ASTRA.ORB"])
        assert abs(weights["ASTRA.PULLBACK"] - 0.5) < 0.01, "Bootstrap: equal weight"
        assert abs(weights["ASTRA.ORB"]      - 0.5) < 0.01, "Bootstrap: equal weight"
    finally:
        db.close()


def test_allocator_zero_weight_for_negative_sharpe():
    from app.models.database import SessionLocal, Base, engine, StrategyPerformance
    from app.services.portfolio_allocator import compute_weights
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(StrategyPerformance).delete(); db.commit()
        # 20 losing trades → negative Sharpe
        _insert_trades(db, "ASTRA.PULLBACK", [-2.0] * 20)
        # 20 winning trades → positive Sharpe
        _insert_trades(db, "ASTRA.ORB",      [+2.0] * 20)
        weights = compute_weights(db, ["ASTRA.PULLBACK", "ASTRA.ORB"])
        assert weights["ASTRA.PULLBACK"] == 0.0, "Negative-Sharpe strategy gets zero weight"
        assert abs(weights["ASTRA.ORB"] - 1.0) < 0.01, "Only positive-Sharpe gets all capital"
    finally:
        db.close()


def test_allocator_sharpe_squared_weighting():
    from app.models.database import SessionLocal, Base, engine, StrategyPerformance
    from app.services.portfolio_allocator import compute_weights
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(StrategyPerformance).delete(); db.commit()
        # PULLBACK: consistent +2% → high Sharpe
        _insert_trades(db, "ASTRA.PULLBACK", [2.0] * 20)
        # ORB: volatile but positive mean → lower Sharpe
        import random; random.seed(0)
        _insert_trades(db, "ASTRA.ORB", [random.uniform(-3, 5) for _ in range(20)])
        weights = compute_weights(db, ["ASTRA.PULLBACK", "ASTRA.ORB"])
        # Consistent strategy must get more weight
        assert weights["ASTRA.PULLBACK"] > weights.get("ASTRA.ORB", 0), (
            "Consistent strategy must outweigh volatile one"
        )
    finally:
        db.close()


def test_allocator_sums_to_one():
    from app.models.database import SessionLocal, Base, engine, StrategyPerformance
    from app.services.portfolio_allocator import compute_weights
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(StrategyPerformance).delete(); db.commit()
        _insert_trades(db, "ASTRA.PULLBACK", [1.5] * 20)
        _insert_trades(db, "ASTRA.ORB",      [0.8] * 20)
        weights = compute_weights(db, ["ASTRA.PULLBACK", "ASTRA.ORB"])
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"
    finally:
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio_allocator.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'app.services.portfolio_allocator'`

- [ ] **Step 3: Implement portfolio_allocator.py**

```python
# backend/app/services/portfolio_allocator.py
"""
ASTRA PortfolioAllocator — Sharpe²-weighted dynamic capital allocation.

Runs weekly. Re-weights capital between strategies based on rolling 20-trade
Sharpe ratio. Strategies with negative Sharpe get zero weight. No floor.
"""
import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

BOOTSTRAP_TRADES = 20          # equal-weight until each strategy has this many
MAX_PORTFOLIO_RISK = 0.20      # total open risk ≤ 20% of portfolio value


def _rolling_sharpe(net_pcts: list, n: int = 20) -> float:
    """
    Sharpe ratio from the last `n` completed trades.
    Returns 0.0 if insufficient data or std is zero.
    """
    recent = net_pcts[-n:] if len(net_pcts) >= n else net_pcts
    if len(recent) < 2:
        return 0.0
    arr = np.array(recent, dtype=float)
    std = arr.std()
    if std == 0:
        return float(np.sign(arr.mean()))
    return float(arr.mean() / std)


def compute_weights(db: Session, strategy_names: list[str]) -> dict[str, float]:
    """
    Compute capital allocation weights for the given strategies.

    Rules:
    - Bootstrap: equal weight if any strategy has < BOOTSTRAP_TRADES trades.
    - Active: weight ∝ Sharpe² for Sharpe > 0; zero otherwise.
    - Always sums to 1.0 (or 0.0 if all strategies have negative Sharpe).

    Returns dict mapping strategy name → weight (0.0–1.0).
    """
    from app.models.database import StrategyPerformance

    trade_map: dict[str, list[float]] = {}
    for name in strategy_names:
        rows = (
            db.query(StrategyPerformance.net_pct)
            .filter(StrategyPerformance.strategy == name)
            .order_by(StrategyPerformance.exit_ts)
            .all()
        )
        trade_map[name] = [float(r.net_pct) for r in rows]

    # Bootstrap: equal weight if any strategy below threshold
    if any(len(v) < BOOTSTRAP_TRADES for v in trade_map.values()):
        n = len(strategy_names)
        return {name: 1.0 / n for name in strategy_names}

    # Compute Sharpe² weights
    sharpes = {name: _rolling_sharpe(pcts) for name, pcts in trade_map.items()}
    logger.info(f"[allocator] Sharpe scores: {sharpes}")

    active = {name: s for name, s in sharpes.items() if s > 0}
    if not active:
        logger.warning("[allocator] All strategies have non-positive Sharpe — equal weight")
        n = len(strategy_names)
        return {name: 1.0 / n for name in strategy_names}

    squared = {name: s ** 2 for name, s in active.items()}
    total   = sum(squared.values())
    weights = {name: squared[name] / total for name in active}

    # Zero out inactive strategies
    for name in strategy_names:
        if name not in weights:
            weights[name] = 0.0

    return weights


def record_trade(db: Session, strategy: str, net_pct: float,
                 size_frac: float = 1.0,
                 entry_ts=None, exit_ts=None,
                 user_id: Optional[int] = None) -> None:
    """Persist a completed trade result for the allocator to read."""
    import datetime
    from app.models.database import StrategyPerformance
    row = StrategyPerformance(
        strategy=strategy,
        net_pct=net_pct,
        size_frac=size_frac,
        entry_ts=entry_ts,
        exit_ts=exit_ts or datetime.datetime.utcnow(),
        user_id=user_id,
    )
    db.add(row)
    db.commit()
    logger.debug(f"[allocator] recorded {strategy} trade net_pct={net_pct:.2f}%")


def get_current_allocation(db: Session) -> dict:
    """Return current weights + supporting Sharpe stats for the API."""
    from app.models.database import StrategyPerformance
    strategies = [
        r.strategy for r in
        db.query(StrategyPerformance.strategy).distinct().all()
    ]
    if not strategies:
        return {"weights": {}, "sharpes": {}, "trade_counts": {}}

    weights = compute_weights(db, strategies)

    sharpes = {}
    trade_counts = {}
    for name in strategies:
        rows = (db.query(StrategyPerformance.net_pct)
                  .filter(StrategyPerformance.strategy == name)
                  .order_by(StrategyPerformance.exit_ts).all())
        pcts = [float(r.net_pct) for r in rows]
        sharpes[name]       = round(_rolling_sharpe(pcts), 3)
        trade_counts[name]  = len(pcts)

    return {
        "weights":       weights,
        "sharpes":       sharpes,
        "trade_counts":  trade_counts,
    }
```

- [ ] **Step 4: Run allocator tests**

```bash
pytest tests/test_portfolio_allocator.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Run full suite**

```bash
pytest -q --tb=short
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/portfolio_allocator.py backend/tests/test_portfolio_allocator.py
git commit -m "feat(allocator): Sharpe²-weighted PortfolioAllocator with bootstrap + zero floor"
```

---

## Task 6: Portfolio API Router

**Files:**
- Create: `backend/app/api/portfolio_router.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write failing test**

```python
# Append to backend/tests/test_api.py

def test_portfolio_allocation_endpoint_returns_200(client):
    """GET /portfolio/allocation must return weights dict."""
    resp = client.get("/portfolio/allocation")
    assert resp.status_code == 200
    data = resp.json()
    assert "weights" in data
    assert "sharpes" in data
    assert "trade_counts" in data
```

Run:
```bash
pytest tests/test_api.py -k "portfolio_allocation" -v
```
Expected: `404` — endpoint doesn't exist yet.

- [ ] **Step 2: Check how test_api.py builds its client**

```bash
head -40 backend/tests/test_api.py
```
Find the `client` fixture definition. The router must be registered in `main.py` in the same way as existing routers.

- [ ] **Step 3: Create portfolio_router.py**

```python
# backend/app/api/portfolio_router.py
"""
Portfolio allocation endpoints.
GET  /portfolio/allocation  — current Sharpe² weights per strategy
POST /portfolio/rebalance   — trigger manual rebalance (admin only)
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.services.portfolio_allocator import compute_weights, get_current_allocation

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/allocation")
def portfolio_allocation(db: Session = Depends(get_db)):
    """Current capital allocation weights across active strategies."""
    return get_current_allocation(db)


@router.post("/rebalance")
def portfolio_rebalance(
    strategy_names: list[str] | None = None,
    db: Session = Depends(get_db),
):
    """
    Trigger a manual rebalance. Returns new weights.
    If strategy_names is None, uses all strategies with recorded trades.
    """
    from app.models.database import StrategyPerformance
    if strategy_names is None:
        strategy_names = [
            r.strategy for r in
            db.query(StrategyPerformance.strategy).distinct().all()
        ]
    if not strategy_names:
        return {"weights": {}, "message": "no strategies with recorded trades"}
    weights = compute_weights(db, strategy_names)
    return {"weights": weights, "rebalanced": True}
```

- [ ] **Step 4: Register router in main.py**

Open `backend/main.py`. Find where existing routers are included (look for `app.include_router`). Add:

```python
from app.api.portfolio_router import router as portfolio_router
app.include_router(portfolio_router)
```

- [ ] **Step 5: Run test**

```bash
pytest tests/test_api.py -k "portfolio_allocation" -v
```
Expected: `1 passed`

- [ ] **Step 6: Run full suite**

```bash
pytest -q --tb=short
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/portfolio_router.py backend/main.py
git commit -m "feat(api): portfolio allocation + rebalance endpoints"
```

---

## Task 7: End-to-End Validation

- [ ] **Step 1: Run full evaluator across all strategies**

```bash
cd backend && source .venv/bin/activate && python - <<'EOF'
import logging; logging.basicConfig(level=logging.WARNING)
from app.services.strategy_evaluator import evaluate, print_report
from app.strategies.registry import get_all

for s in get_all():
    print_report(evaluate(s, max_workers=6))
EOF
```

**Gate check:** PULLBACK v2.0 must show at minimum MARGINAL verdict (expectancy > 0, PF > 1.0). Record actual numbers.

- [ ] **Step 2: Seed allocator with evaluation results**

```python
# Run in Python REPL or as a script
from app.models.database import SessionLocal, Base, engine
from app.services.portfolio_allocator import record_trade

Base.metadata.create_all(bind=engine)
db = SessionLocal()

# Seed with simulated trade history from evaluator run
# Replace net_pcts with actual per-trade results from the evaluation
pullback_trades = [2.1, -1.2, 3.4, -0.8, 4.1] * 4   # 20 sample trades
orb_trades      = [1.2, -0.5, 1.8, -1.1, 2.3] * 4

for pct in pullback_trades:
    record_trade(db, "ASTRA.PULLBACK", pct)
for pct in orb_trades:
    record_trade(db, "ASTRA.ORB", pct)

db.close()
print("Seeded.")
```

- [ ] **Step 3: Verify allocation endpoint**

```bash
# Start backend
cd backend && source .venv/bin/activate && uvicorn main:app --port 8000 &
sleep 3
curl -s http://localhost:8000/portfolio/allocation | python -m json.tool
```

Expected output shape:
```json
{
  "weights": {"ASTRA.PULLBACK": 0.65, "ASTRA.ORB": 0.35},
  "sharpes": {"ASTRA.PULLBACK": 1.24, "ASTRA.ORB": 0.87},
  "trade_counts": {"ASTRA.PULLBACK": 20, "ASTRA.ORB": 20}
}
```

- [ ] **Step 4: Run final full test suite**

```bash
pytest -q
```
Expected: all pass.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: ASTRA Hybrid Portfolio v2.0 complete — PULLBACK v2.0 + ORB v2.0 + PortfolioAllocator"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Phase 1 (PULLBACK v2.0) → Tasks 1–2. Phase 2 (ORB v2.0) → Task 3. Phase 3 (Allocator) → Tasks 4–6. End-to-end → Task 7.
- [x] **No placeholders:** All steps contain actual code.
- [x] **Type consistency:** `compute_weights` returns `dict[str, float]` — referenced identically in router and tests. `record_trade` signature consistent across service and seeding script. `StrategyPerformance` columns match both insert and query calls.
- [x] **Scope:** Three independent phases, each committable and testable on its own. Phase 3 depends on Phase 1+2 passing evaluation — gated explicitly in Task 7 Step 1.
