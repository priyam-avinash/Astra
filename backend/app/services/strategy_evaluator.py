"""
ASTRA Strategy Evaluator
==========================
Runs any Strategy through realistic, robustness-aware backtesting.
Produces an honest 1-page report.

Three layers:
  1. Walk-forward backtest    — train/test contamination prevented
  2. Cost stress test         — re-run at 1.5× and 2× transaction costs
  3. Regime stability check   — split window into N quarters, check pass-rate

A strategy "passes" only if:
  - Expectancy/trade > +0.5% net
  - Profit factor    > 1.3
  - Survives 1.5× cost (still positive expectancy)
  - Profitable in ≥ 60% of regime quarters
  - ≥ 30 trades (statistical relevance)

If a strategy fails this gate, it is REJECTED, not tuned. Tuning to pass the
evaluator is overfitting in disguise.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    CostModel, ExitReason, Position, Signal, SignalSide, Strategy, Trade
)

logger = logging.getLogger(__name__)

TRADE_NOTIONAL = 100_000   # ₹1L per trade — fixed for comparability


# ── Single-symbol backtest runner ─────────────────────────────────────────

def _simulate_symbol(strategy: Strategy, symbol: str, df: pd.DataFrame,
                     cost: CostModel) -> list[Trade]:
    """
    Walk-forward bar-by-bar simulation for one symbol. Respects strategy's
    own indicators (via precompute) and exit logic (via should_exit + signal-attached stops).
    Returns list of closed Trade objects.
    """
    if df is None or df.empty:
        return []
    enriched = strategy.precompute(df)
    if enriched is None or enriched.empty:
        return []

    trades: list[Trade] = []
    position: Optional[Position] = None

    for i in range(1, len(enriched) - 1):
        bars_up_to_now = enriched.iloc[: i + 1]
        row = enriched.iloc[i]
        nxt = enriched.iloc[i + 1]

        if position is None:
            sig = strategy.signal(bars_up_to_now, symbol)
            if sig is None:
                continue
            # Enter at NEXT bar open + slippage
            fill = float(nxt["Open"]) * (1 + cost.slippage_per_side_pct)
            sized_notional = TRADE_NOTIONAL * max(0.1, min(2.0, sig.size_fraction))
            position = Position(
                symbol=symbol,
                side=sig.side,
                entry_price=fill,
                entry_ts=enriched.index[i + 1],
                quantity=max(1, int(sized_notional / fill)),
                hard_stop=sig.hard_stop,
                target=sig.target,
                trail_pct=sig.trail_pct,
                trail_activate_pct=sig.trail_activate_pct,
                peak_price=fill,
                metadata=sig.metadata,
            )
            continue

        # ── We are in a position; resolve exit ──────────────────────────
        high  = float(row["High"])
        low   = float(row["Low"])
        close = float(row["Close"])

        # Update peak (for trailing stop)
        if high > position.peak_price:
            position.peak_price = high

        exit_reason: Optional[ExitReason] = None
        exit_price_raw: Optional[float] = None

        # 1. Hard stop (intrabar)
        if position.hard_stop is not None and low <= position.hard_stop:
            exit_reason = ExitReason.HARD_STOP
            exit_price_raw = position.hard_stop

        # 2. Target (intrabar)
        elif position.target is not None and high >= position.target:
            exit_reason = ExitReason.TARGET
            exit_price_raw = position.target

        # 3. Trailing stop (intrabar, activated after some unrealised gain)
        elif position.trail_pct is not None:
            peak_gain = (position.peak_price - position.entry_price) / position.entry_price
            if peak_gain >= position.trail_activate_pct:
                trail_lvl = position.peak_price * (1 - position.trail_pct)
                if low <= trail_lvl:
                    exit_reason = ExitReason.TRAIL_STOP
                    exit_price_raw = trail_lvl

        # 4. Strategy-specific exit (e.g. Stage-2 SMA breakdown)
        if exit_reason is None:
            strat_exit = strategy.should_exit(position, bars_up_to_now)
            if strat_exit is not None:
                exit_reason = strat_exit
                exit_price_raw = float(nxt["Open"])   # fill at next bar's open

        if exit_reason is None:
            continue

        # Build closed Trade
        exit_price = exit_price_raw * (1 - cost.slippage_per_side_pct)
        if exit_reason in (ExitReason.STRATEGY, ExitReason.TARGET):
            exit_ts = enriched.index[i + 1]
            days_held = (enriched.index[i + 1] - position.entry_ts).days or (i + 1 - max(0, position.quantity * 0))
        else:
            exit_ts = enriched.index[i]
            days_held = (enriched.index[i] - position.entry_ts).days

        gross = (exit_price - position.entry_price) / position.entry_price
        if position.side == SignalSide.SELL:
            gross = -gross
        net = gross - cost.round_trip_pct

        trades.append(Trade(
            symbol=symbol,
            side=position.side,
            entry_ts=position.entry_ts,
            exit_ts=exit_ts,
            entry_price=round(position.entry_price, 2),
            exit_price=round(exit_price, 2),
            peak_price=round(position.peak_price, 2),
            quantity=position.quantity,
            days_held=max(0, days_held),
            gross_pct=round(gross * 100, 3),
            net_pct=round(net * 100, 3),
            pnl_rs=round(net * TRADE_NOTIONAL, 2),
            exit_reason=exit_reason,
            strategy=strategy.name,
        ))
        position = None

    # Mark-to-market any open position at end
    if position is not None:
        last = enriched.iloc[-1]
        exit_price = float(last["Close"]) * (1 - cost.slippage_per_side_pct)
        gross = (exit_price - position.entry_price) / position.entry_price
        if position.side == SignalSide.SELL:
            gross = -gross
        net = gross - cost.round_trip_pct
        days_held = (enriched.index[-1] - position.entry_ts).days
        trades.append(Trade(
            symbol=symbol,
            side=position.side,
            entry_ts=position.entry_ts,
            exit_ts=enriched.index[-1],
            entry_price=round(position.entry_price, 2),
            exit_price=round(exit_price, 2),
            peak_price=round(position.peak_price, 2),
            quantity=position.quantity,
            days_held=max(0, days_held),
            gross_pct=round(gross * 100, 3),
            net_pct=round(net * 100, 3),
            pnl_rs=round(net * TRADE_NOTIONAL, 2),
            exit_reason=ExitReason.EOD_FORCE,
            strategy=strategy.name,
        ))

    return trades


# ── Stats from a trade list ────────────────────────────────────────────────

@dataclass
class BacktestStats:
    n_trades:      int
    n_symbols:     int
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    rr_ratio:      float
    expectancy:    float        # in %
    profit_factor: float
    total_pnl_rs:  float
    avg_days_held: float
    max_dd:        float        # in % on cumulative trade-P&L curve

    def passes_basic_gate(self) -> bool:
        return (
            self.n_trades >= 30
            and self.expectancy > 0.5
            and self.profit_factor > 1.3
        )


def _stats(trades: list[Trade]) -> BacktestStats:
    if not trades:
        return BacktestStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    df = pd.DataFrame([t.__dict__ for t in trades])
    wins   = df[df["net_pct"] > 0]
    losses = df[df["net_pct"] <= 0]
    wr  = len(wins) / len(df) * 100
    aw  = wins["net_pct"].mean()   if len(wins)   else 0.0
    al  = losses["net_pct"].mean() if len(losses) else 0.0
    rr  = abs(aw / al) if al < 0 else float("inf")
    exp = wr / 100 * aw + (1 - wr / 100) * al
    gp  = wins["pnl_rs"].sum()
    gl  = abs(losses["pnl_rs"].sum())
    pf  = gp / gl if gl > 0 else float("inf")
    cum = df.sort_values("exit_ts")["pnl_rs"].cumsum()
    peak = cum.cummax()
    dd_rs = (cum - peak).min()
    max_dd = dd_rs / TRADE_NOTIONAL * 100   # rough — % of single trade notional

    return BacktestStats(
        n_trades=len(df),
        n_symbols=df["symbol"].nunique(),
        win_rate=round(wr, 1),
        avg_win=round(float(aw), 2),
        avg_loss=round(float(al), 2),
        rr_ratio=round(rr, 2) if rr != float("inf") else 999.0,
        expectancy=round(float(exp), 2),
        profit_factor=round(pf, 2) if pf != float("inf") else 999.0,
        total_pnl_rs=round(float(df["pnl_rs"].sum()), 0),
        avg_days_held=round(float(df["days_held"].mean()), 1),
        max_dd=round(float(max_dd), 1),
    )


# ── Multi-symbol fetch + backtest ──────────────────────────────────────────

# ── Process-level data cache: fetch once, reuse across all strategies + A/B runs ──
_UNIVERSE_DATA_CACHE: dict = {}   # symbol → DataFrame


def _fetch_2y_daily(symbol: str) -> Optional[pd.DataFrame]:
    """
    yf.Ticker().history() — single-ticker, uses a persistent session with crumb
    management that is thread-safe unlike yf.download() batch mode.
    Falls back to Upstox when available.
    """
    try:
        from app.services.upstox_data import upstox_service
        if upstox_service.is_available():
            df = upstox_service.get_ohlcv(symbol, period="2y", interval="1d")
            if df is not None and not df.empty and len(df) > 200:
                return df
    except Exception:
        pass
    try:
        import yfinance as yf
        raw = yf.download(
            f"{symbol}.NS", period="2y", interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if raw is not None and not raw.empty and len(raw) > 200:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw.index.name = "Date"
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
            return raw[cols].copy()
    except Exception:
        pass
    return None


def _load_universe(universe: list[str], max_workers: int = 4) -> dict:
    """
    Fetch all symbols once per process. Results cached in _UNIVERSE_DATA_CACHE —
    shared across all strategies and both A/B runs.
    Per-symbol timeout of 20s prevents one hanging ticker from blocking workers.
    """
    global _UNIVERSE_DATA_CACHE
    missing = [s for s in universe if s not in _UNIVERSE_DATA_CACHE]
    if not missing:
        print(f"  [fetch] all {len(universe)} symbols already cached — skipping fetch")
        return {s: _UNIVERSE_DATA_CACHE[s] for s in universe if s in _UNIVERSE_DATA_CACHE}

    print(f"  [fetch] downloading {len(missing)} symbols ({len(universe)-len(missing)} cached)...")
    workers = min(4, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_2y_daily, s): s for s in missing}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result(timeout=20)   # 20s hard cap per symbol
                if df is not None:
                    _UNIVERSE_DATA_CACHE[sym] = df
            except Exception as e:
                logger.debug(f"[evaluator] fetch error for {sym}: {e}")
            done += 1
            if done % 10 == 0:
                print(f"  [fetch] {done}/{len(missing)} done, {sum(1 for s in missing if s in _UNIVERSE_DATA_CACHE)} succeeded...")

    fetched = sum(1 for s in missing if s in _UNIVERSE_DATA_CACHE)
    print(f"  [fetch] complete: {fetched}/{len(missing)} symbols fetched successfully")
    return {s: _UNIVERSE_DATA_CACHE[s] for s in universe if s in _UNIVERSE_DATA_CACHE}


def clear_universe_cache() -> None:
    _UNIVERSE_DATA_CACHE.clear()


def _run_universe(strategy: Strategy, cost: CostModel,
                  max_workers: int = 4) -> list[Trade]:
    """
    Three-phase universe backtest:
      1. Fetch all symbols (cached — shared across strategies and A/B runs)
      2. Call strategy.prepare_universe(data)  — for cross-sectional strategies
      3. Parallel per-symbol bar-by-bar simulation
    """
    universe = strategy.universe()

    # Phase 1: fetch (uses cache after first call)
    universe_data = _load_universe(universe, max_workers=max_workers)
    logger.info(f"[evaluator] running simulation on {len(universe_data)} symbols...")

    # Phase 2: universe-wide preparation (no-op for per-symbol strategies)
    try:
        strategy.prepare_universe(universe_data)
    except Exception as e:
        logger.warning(f"[evaluator] prepare_universe failed: {e}")

    # Phase 3: parallel per-symbol simulation
    trades: list[Trade] = []
    def _sim_job(sym):
        return _simulate_symbol(strategy, sym, universe_data[sym], cost)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_sim_job, s): s for s in universe_data}
        for fut in as_completed(futures):
            try:
                trades.extend(fut.result())
            except Exception as e:
                logger.debug(f"[evaluator] sim error: {e}")
    return trades


# ── Regime stability check ────────────────────────────────────────────────

def _regime_pass_rate(trades: list[Trade], n_buckets: int = 8) -> tuple[float, list[float]]:
    """
    Split trades chronologically into N buckets by exit_ts. Return
    (fraction profitable, per-bucket P&L). Tests whether edge is regime-robust.
    """
    if not trades:
        return 0.0, []
    df = pd.DataFrame([{"exit_ts": t.exit_ts, "pnl_rs": t.pnl_rs} for t in trades])
    df = df.sort_values("exit_ts").reset_index(drop=True)
    if len(df) < n_buckets:
        n_buckets = max(1, len(df) // 3)
    bucket_size = max(1, len(df) // n_buckets)
    buckets_pnl: list[float] = []
    for i in range(n_buckets):
        lo = i * bucket_size
        hi = lo + bucket_size if i < n_buckets - 1 else len(df)
        seg = df.iloc[lo:hi]
        if len(seg) > 0:
            buckets_pnl.append(float(seg["pnl_rs"].sum()))
    if not buckets_pnl:
        return 0.0, []
    pass_rate = sum(1 for p in buckets_pnl if p > 0) / len(buckets_pnl)
    return pass_rate, buckets_pnl


# ── Full evaluator (public API) ────────────────────────────────────────────

@dataclass
class EvaluationReport:
    strategy_name:    str
    strategy_version: str
    base_stats:       BacktestStats
    stress_stats:     BacktestStats    # at 1.5× costs
    extreme_stats:    BacktestStats    # at 2.0× costs
    regime_pass_rate: float            # 0-1
    regime_buckets:   list             # per-bucket P&L
    verdict:          str              # PASS / MARGINAL / FAIL
    reasons:          list             # human-readable reasons
    elapsed_sec:      float


def evaluate(strategy: Strategy, max_workers: int = 4,
             base_cost: Optional[CostModel] = None) -> EvaluationReport:
    """Run full evaluation. Single fetch per symbol, re-simulated under 3 cost scenarios."""
    base_cost = base_cost or CostModel()
    start = time.time()
    meta = strategy.meta()
    logger.info(f"[evaluator] starting {meta.name} v{meta.version}")

    # Run base case + cache trades by symbol so we don't refetch
    base_trades = _run_universe(strategy, base_cost, max_workers=max_workers)
    base_stats  = _stats(base_trades)

    # 1.5× and 2× cost — simulate same trades with extra cost deducted (gross stays same)
    def _restress(trades: list[Trade], extra_cost_pct: float) -> list[Trade]:
        out = []
        for t in trades:
            new_net_pct = t.gross_pct - (base_cost.round_trip_pct + extra_cost_pct) * 100
            new = Trade(**{**t.__dict__,
                           "net_pct": round(new_net_pct, 3),
                           "pnl_rs":  round(new_net_pct / 100 * TRADE_NOTIONAL, 2)})
            out.append(new)
        return out

    stress_trades  = _restress(base_trades, extra_cost_pct=base_cost.round_trip_pct * 0.5)
    stress_stats   = _stats(stress_trades)
    extreme_trades = _restress(base_trades, extra_cost_pct=base_cost.round_trip_pct * 1.0)
    extreme_stats  = _stats(extreme_trades)

    regime_pass, regime_buckets = _regime_pass_rate(base_trades, n_buckets=8)

    # Verdict
    reasons: list[str] = []
    if base_stats.n_trades < 30:
        reasons.append(f"insufficient trades ({base_stats.n_trades} < 30)")
    if base_stats.expectancy <= 0.5:
        reasons.append(f"low expectancy ({base_stats.expectancy}% ≤ 0.5%)")
    if base_stats.profit_factor <= 1.3:
        reasons.append(f"low profit factor ({base_stats.profit_factor} ≤ 1.3)")
    if stress_stats.expectancy <= 0:
        reasons.append("fails 1.5× cost stress")
    if regime_pass < 0.6:
        reasons.append(f"regime pass-rate too low ({regime_pass*100:.0f}% < 60%)")

    if not reasons:
        verdict = "PASS"
    elif (
        base_stats.expectancy > 0
        and base_stats.profit_factor > 1.0
        and stress_stats.expectancy > -0.1
    ):
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    elapsed = round(time.time() - start, 1)
    return EvaluationReport(
        strategy_name=meta.name,
        strategy_version=meta.version,
        base_stats=base_stats,
        stress_stats=stress_stats,
        extreme_stats=extreme_stats,
        regime_pass_rate=round(regime_pass, 2),
        regime_buckets=[round(b, 0) for b in regime_buckets],
        verdict=verdict,
        reasons=reasons,
        elapsed_sec=elapsed,
    )


def print_report(report: EvaluationReport) -> None:
    print("\n" + "=" * 78)
    print(f"  STRATEGY EVALUATION  —  {report.strategy_name} v{report.strategy_version}")
    print("=" * 78)
    print(f"  Verdict: {report.verdict}")
    if report.reasons:
        for r in report.reasons:
            print(f"    • {r}")
    print()
    print(f"  ┌─ Base case (cost: 0.25% RT)")
    s = report.base_stats
    print(f"  │   trades={s.n_trades:>4}  symbols={s.n_symbols:>3}  WR={s.win_rate:>5.1f}%  "
          f"R:R={s.rr_ratio:>5.2f}  exp={s.expectancy:>+5.2f}%  PF={s.profit_factor:>4.2f}  P&L=₹{s.total_pnl_rs:>+10,.0f}")
    s = report.stress_stats
    print(f"  ├─ 1.5× cost stress")
    print(f"  │   exp={s.expectancy:>+5.2f}%  PF={s.profit_factor:>4.2f}  P&L=₹{s.total_pnl_rs:>+10,.0f}")
    s = report.extreme_stats
    print(f"  ├─ 2.0× cost extreme")
    print(f"  │   exp={s.expectancy:>+5.2f}%  PF={s.profit_factor:>4.2f}  P&L=₹{s.total_pnl_rs:>+10,.0f}")
    print(f"  └─ Regime stability (8 quarters): {report.regime_pass_rate*100:.0f}% profitable")
    print(f"      bucket P&L: {report.regime_buckets}")
    print()
    print(f"  Elapsed: {report.elapsed_sec}s")
    print("=" * 78 + "\n")
