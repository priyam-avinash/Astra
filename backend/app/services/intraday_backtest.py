"""
ASTRA Intraday Backtest Engine
================================
Walk-forward bar-by-bar simulation of ORB, EMA_Cross, and Momentum strategies
on historical 15-minute OHLCV data.

Usage:
    from app.services.intraday_backtest import run_intraday_backtest
    result = run_intraday_backtest("RELIANCE", days=30)
"""

import logging
from datetime import datetime, time as dtime
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from app.services.intraday_engine import (
    _fetch_intraday_extended,
    _compute_vwap,
    _compute_daily_vwap,
    _compute_rsi_series,
    _compute_macd,
    _compute_atr,
    _atr_sl_tp,
    _compute_orb,
    _trend_aligned,
    INTRADAY_SL_PCT,
    INTRADAY_TP_PCT,
    NO_TRADE_AFTER,
    SQUARE_OFF_TIME,
    ORB_WINDOW_MIN,
)

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

# Assume a fixed notional trade size for P&L calculation
TRADE_NOTIONAL  = 100_000  # ₹1 lakh per trade
SLIPPAGE_PCT    = 0.001    # 0.1% per side (NSE liquid stock intraday slippage)

# Realistic NSE intraday transaction costs (round-trip, deducted at exit):
#   Brokerage (Zerodha intraday):    ~0.04% (₹40 on ₹1L round-trip; flat ₹20/leg cap)
#   STT (sell side, intraday equity): 0.025%
#   Exchange transaction charges:    ~0.007% round-trip
#   SEBI + stamp duty:               ~0.005% round-trip
#   GST (18% on brokerage + exch):   ~0.009% round-trip
#                                    ─────────
#   TOTAL FIXED COSTS:                0.086% round-trip
# Plus 0.2% round-trip from SLIPPAGE_PCT (applied separately at fill).
# Net realistic round-trip cost: ~0.29% on ₹1L intraday positions.
TRANSACTION_COST_PCT = 0.00086   # 0.086% round-trip, applied to pnl_pct at exit


# ── Per-trade simulation ──────────────────────────────────────────────────────

def _simulate_trade(signal: str, entry_price: float, sl: float, tp: float,
                    future_bars: pd.DataFrame, entry_slipped: bool = False) -> dict:
    """
    Bar-by-bar simulation with no intrabar look-ahead:
    - entry fill = entry_price ± 0.1% slippage (skipped if `entry_slipped`)
    - each bar: the stop set on PREVIOUS bars is checked first, then the target
      (if both are touched in one bar the stop wins — conservative)
    - gap-aware stop fills: BUY exits at min(stop, Open), SELL at max(stop, Open);
      stop exits pay exit slippage, targets fill at the target price
    - trailing stop (breakeven at 50% of the move to TP, 50%-of-TP trail at 75%)
      is updated at the END of a bar, so a new level only applies from the next bar
    - auto square-off at the OPEN of the 15:15 bar (its close is 15 minutes later)
    """
    sq_off = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])
    if entry_slipped:
        fill_price = entry_price
    else:
        fill_price = entry_price * (1 + SLIPPAGE_PCT) if signal == "BUY" else entry_price * (1 - SLIPPAGE_PCT)

    def _pnl(exit_px):
        return (exit_px - fill_price) / fill_price if signal == "BUY" else (fill_price - exit_px) / fill_price

    def _slip_exit(px):
        return px * (1 - SLIPPAGE_PCT) if signal == "BUY" else px * (1 + SLIPPAGE_PCT)

    trailing_sl = sl
    move_to_tp = abs(tp - fill_price)

    for i, (ts, bar) in enumerate(future_bars.iterrows()):
        bar_time = ts.time() if hasattr(ts, "time") else ts.to_pydatetime().time()
        o, high, low = float(bar["Open"]), float(bar["High"]), float(bar["Low"])

        if bar_time >= sq_off:
            px = _slip_exit(o)
            return _trade_result("SQ_OFF", px, _pnl(px), i + 1)

        if signal == "BUY":
            if low <= trailing_sl:
                px = _slip_exit(min(trailing_sl, o))
                return _trade_result("SL", px, _pnl(px), i + 1)
            if high >= tp:
                return _trade_result("TP", tp, _pnl(tp), i + 1)
        else:
            if high >= trailing_sl:
                px = _slip_exit(max(trailing_sl, o))
                return _trade_result("SL", px, _pnl(px), i + 1)
            if low <= tp:
                return _trade_result("TP", tp, _pnl(tp), i + 1)

        # end of bar: raise the trail for the NEXT bar
        if move_to_tp > 0:
            if signal == "BUY":
                progress = (high - fill_price) / move_to_tp
                if progress >= 0.75:
                    trailing_sl = max(trailing_sl, fill_price + move_to_tp * 0.50)
                elif progress >= 0.50:
                    trailing_sl = max(trailing_sl, fill_price)
            else:
                progress = (fill_price - low) / move_to_tp
                if progress >= 0.75:
                    trailing_sl = min(trailing_sl, fill_price - move_to_tp * 0.50)
                elif progress >= 0.50:
                    trailing_sl = min(trailing_sl, fill_price)

    if not future_bars.empty:
        px = _slip_exit(float(future_bars.iloc[-1]["Close"]))
        return _trade_result("SQ_OFF", px, _pnl(px), len(future_bars))
    return _trade_result("NO_EXIT", entry_price, 0.0, 0)


def _execute_signal(sig: dict, day_df: pd.DataFrame) -> Optional[dict]:
    """
    Enter at the NEXT bar's open after the signal bar (the signal is only known
    once that bar has closed). SL/TP keep their distance from the signal price,
    re-anchored to the actual entry. No trade if the next bar is at/after 15:15.
    """
    sq_off = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])
    after = day_df[day_df.index > sig["ts"]]
    if after.empty:
        return None
    entry_ts = after.index[0]
    if entry_ts.time() >= sq_off:
        return None
    raw_open = float(after.iloc[0]["Open"])
    fill = raw_open * (1 + SLIPPAGE_PCT) if sig["signal"] == "BUY" else raw_open * (1 - SLIPPAGE_PCT)
    shift = fill - sig["entry_price"]
    sl, tp = sig["sl"] + shift, sig["tp"] + shift
    res = _simulate_trade(sig["signal"], fill, sl, tp, after, entry_slipped=True)
    return {"signal": sig["signal"], "signal_time": str(sig["ts"]), "signal_price": sig["entry_price"],
            "entry_time": str(entry_ts), "entry_price": round(fill, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            **res}


def _pick_combined_signal(signals: list) -> Optional[dict]:
    """
    Combined strategy without look-ahead: take the EARLIEST signal bar of the
    day; consensus is counted only among signals on that same bar. A tie with
    no majority → no trade; otherwise the best reward:risk wins.
    """
    sigs = [s for s in signals if s is not None]
    if not sigs:
        return None
    first_ts = min(s["ts"] for s in sigs)
    same = [s for s in sigs if s["ts"] == first_ts]
    buys = [s for s in same if s["signal"] == "BUY"]
    sells = [s for s in same if s["signal"] == "SELL"]
    if buys and sells and len(buys) == len(sells):
        return None
    dominant = buys if len(buys) > len(sells) else sells

    def _rr(sig):
        risk = abs(sig["entry_price"] - sig["sl"])
        return abs(sig["tp"] - sig["entry_price"]) / risk if risk > 0 else 0.0
    return max(dominant, key=_rr)


def _trade_result(outcome: str, exit_price: float, pnl_pct: float, bars_held: int) -> dict:
    # Deduct realistic round-trip transaction costs (brokerage + STT + exchange + GST + stamp).
    # SLIPPAGE_PCT is already baked into the entry/exit fill prices upstream.
    net_pnl_pct = pnl_pct - TRANSACTION_COST_PCT
    pnl_rs      = round(net_pnl_pct * TRADE_NOTIONAL, 2)
    return {
        "outcome":    outcome,
        "exit_price": round(exit_price, 2),
        "pnl_pct":    round(net_pnl_pct * 100, 3),   # in %, net of costs
        "pnl_rs":     pnl_rs,
        "bars_held":  bars_held,
        "gross_pnl_pct": round(pnl_pct * 100, 3),    # pre-cost gross %, for inspection
    }


# ── Strategy signal generators (bar-by-bar, up to each bar) ──────────────────

def _get_orb_signal_for_day(day_df: pd.DataFrame, date) -> Optional[dict]:
    """
    Return the FIRST actionable ORB signal for the day, or None if HOLD throughout.
    """
    orb_high, orb_low = _compute_orb(day_df, date)
    if orb_high is None:
        return None

    orb_end  = dtime(9, 15 + ORB_WINDOW_MIN)
    no_trade = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])

    post_orb = day_df[(day_df.index.time >= orb_end) & (day_df.index.time <= no_trade)]

    for ts, bar in post_orb.iterrows():
        close  = float(bar["Close"])
        avg_vol = float(day_df.loc[:ts, "Volume"].mean())   # only bars seen so far
        # Require at least 1.1× average volume — mild surge confirms genuine breakout
        # (1.5× was too strict: fires 0 times on GBM synthetic data due to uniform lognormal vol)
        vol_ok = float(bar["Volume"]) >= 1.1 * avg_vol if avg_vol > 0 else True
        if close > orb_high and vol_ok:
            sl, tp = _atr_sl_tp(day_df.loc[:ts], close, "BUY")
            return {"signal": "BUY",  "entry_price": close,
                    "sl": sl, "tp": tp, "entry": close, "ts": ts}
        elif close < orb_low and vol_ok:
            sl, tp = _atr_sl_tp(day_df.loc[:ts], close, "SELL")
            return {"signal": "SELL", "entry_price": close,
                    "sl": sl, "tp": tp, "entry": close, "ts": ts}
    return None


def _get_ema_cross_signal_for_day(day_df: pd.DataFrame, full_df: pd.DataFrame) -> Optional[dict]:
    """
    EMA 5/13 crossover signal for one day (replaces VWAP_MR).

    BUY  when EMA5 crosses above EMA13 AND RSI > 50 AND Close > VWAP AND trend BULL
    SELL when EMA5 crosses below EMA13 AND RSI < 50 AND Close < VWAP AND trend BEAR

    Returns the FIRST such crossover within the trading session, or None.
    """
    if len(day_df) < 15:
        return None

    df = day_df.copy()
    df["EMA5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["RSI"]   = _compute_rsi_series(df["Close"])
    df["VWAP"]  = _compute_daily_vwap(df)
    no_trade    = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])

    rows = list(df.iterrows())
    for i in range(1, len(rows)):
        ts, row       = rows[i]
        _, prev_row   = rows[i - 1]
        if ts.time() > no_trade:
            break
        if any(pd.isna(row.get(c)) for c in ["EMA5", "EMA13", "RSI", "VWAP"]):
            continue

        close = float(row["Close"])
        rsi   = float(row["RSI"])
        vwap  = float(row["VWAP"])

        bull_cross = float(prev_row["EMA5"]) <= float(prev_row["EMA13"]) and float(row["EMA5"]) > float(row["EMA13"])
        bear_cross = float(prev_row["EMA5"]) >= float(prev_row["EMA13"]) and float(row["EMA5"]) < float(row["EMA13"])

        if bull_cross and rsi > 50 and close > vwap:
            if not _trend_aligned(full_df.loc[:ts], "BUY"):
                continue
            sl, tp = _atr_sl_tp(df.iloc[:i+1], close, "BUY")
            return {"signal": "BUY",  "entry_price": close, "sl": sl, "tp": tp, "ts": ts}

        if bear_cross and rsi < 50 and close < vwap:
            if not _trend_aligned(full_df.loc[:ts], "SELL"):
                continue
            sl, tp = _atr_sl_tp(df.iloc[:i+1], close, "SELL")
            return {"signal": "SELL", "entry_price": close, "sl": sl, "tp": tp, "ts": ts}
    return None


def _get_momentum_signal_for_day(day_df: pd.DataFrame, full_df: pd.DataFrame) -> Optional[dict]:
    """Return first actionable Momentum signal for the day (trend-filtered)."""
    if len(day_df) < 15:   # need at least 15 bars for MACD (12+3 warmup)
        return None
    df = day_df.copy()
    df["RSI"] = _compute_rsi_series(df["Close"], 14)
    macd_line, signal_line, histogram = _compute_macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Sig"] = signal_line
    df["MACD_Hist"] = histogram
    no_trade = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])

    prev_row = None
    for ts, row in df.iterrows():
        if ts.time() > no_trade:
            break
        if pd.isna(row.get("RSI")) or pd.isna(row.get("MACD")):
            prev_row = row
            continue
        if prev_row is not None and not pd.isna(prev_row.get("MACD")):
            close = float(row["Close"])
            rsi   = float(row["RSI"])
            hist  = float(row["MACD_Hist"])
            bull_cross = float(prev_row["MACD"]) <= float(prev_row["MACD_Sig"]) and float(row["MACD"]) > float(row["MACD_Sig"])
            bear_cross = float(prev_row["MACD"]) >= float(prev_row["MACD_Sig"]) and float(row["MACD"]) < float(row["MACD_Sig"])
            if rsi > 55 and hist > 0 and bull_cross:
                if not _trend_aligned(full_df.loc[:ts], "BUY"):
                    prev_row = row
                    continue
                sl, tp = _atr_sl_tp(df.loc[:ts], close, "BUY")
                return {"signal": "BUY",  "entry_price": close, "sl": sl, "tp": tp, "ts": ts}
            elif rsi < 45 and hist < 0 and bear_cross:
                if not _trend_aligned(full_df.loc[:ts], "SELL"):
                    prev_row = row
                    continue
                sl, tp = _atr_sl_tp(df.loc[:ts], close, "SELL")
                return {"signal": "SELL", "entry_price": close, "sl": sl, "tp": tp, "ts": ts}
        prev_row = row
    return None


# ── Strategy stats builder ────────────────────────────────────────────────────

def _build_stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "total_pnl_rs": 0.0, "avg_pnl_rs": 0.0,
                "max_drawdown_rs": 0.0, "sharpe": None, "profit_factor": None}

    wins   = [t for t in trades if t["pnl_rs"] > 0]
    losses = [t for t in trades if t["pnl_rs"] <= 0]
    pnls   = [t["pnl_rs"] for t in trades]

    # Drawdown on cumulative trade P&L (in ₹)
    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak
    max_dd  = float(dd.min()) if len(dd) > 0 else 0.0

    # Correct Sharpe: aggregate to DAILY P&L first, then annualise
    sharpe = None
    try:
        daily_map: dict = {}
        for t in trades:
            d = t.get("date", "unknown")
            daily_map[d] = daily_map.get(d, 0.0) + t["pnl_rs"]
        daily_pnls = list(daily_map.values())
        if len(daily_pnls) >= 2:
            arr  = np.array(daily_pnls, dtype=float)
            std  = arr.std()
            if std > 0:
                sharpe = round(float(arr.mean() / std * np.sqrt(252)), 2)
    except Exception:
        pass

    gross_profit  = sum(p for p in pnls if p > 0)
    gross_loss    = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    return {
        "trades":          len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate_pct":    round(len(wins) / len(trades) * 100, 1),
        "total_pnl_rs":    round(float(np.sum(pnls)), 2),
        "avg_pnl_rs":      round(float(np.mean(pnls)), 2),
        "max_drawdown_rs": round(max_dd, 2),
        "sharpe":          sharpe,
        "profit_factor":   profit_factor,
    }


# ── Main backtest function ────────────────────────────────────────────────────

def run_intraday_backtest(symbol: str, days: int = 30) -> dict:
    """
    Walk-forward backtest over `days` of historical 15m data.

    Returns:
    {
        "symbol": str,
        "period_days": int,
        "trading_days_tested": int,
        "data_bars": int,
        "strategies": {
            "ORB":        { trades, wins, losses, win_rate_pct, total_pnl_rs, ... },
            "EMA_Cross":  { ... },
            "Momentum": { ... },
            "Combined": { ... },   # best signal each day (consensus preferred)
        },
        "trade_log": [ ... ],      # all individual trades
        "equity_curve": [ ... ],   # cumulative P&L per day (Combined strategy)
        "generated_at": str,
    }
    """
    clean = symbol.strip().replace(".NS", "").replace(".BSE", "").upper()
    logger.info(f"[backtest] Starting {days}d backtest for {clean}")

    df = _fetch_intraday_extended(clean, days=days)
    if df.empty:
        return {"error": (f"No real 15-minute data available for {clean}. "
                          "Check /api/data/health (Yahoo reachable? Dhan configured?)."),
                "symbol": clean}
    data_source = df.attrs.get("source") or "unknown"
    is_synthetic = data_source == "synthetic"

    # Ensure IST timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    trading_dates = sorted(set(df.index.date))
    logger.info(f"[backtest] {len(trading_dates)} trading days, {len(df)} bars for {clean}")

    orb_trades  = []
    ema_trades = []
    mom_trades  = []
    combined_trades = []
    trade_log   = []
    equity_curve = []
    cum_combined = 0.0

    for date in trading_dates:
        day_df = df[df.index.date == date].copy()
        if len(day_df) < 10:
            continue

        # Get first signal from each strategy on this day
        orb_sig  = _get_orb_signal_for_day(day_df, date)
        ema_sig  = _get_ema_cross_signal_for_day(day_df, df)
        mom_sig  = _get_momentum_signal_for_day(day_df, df)

        def _simulate_if_signal(sig, strategy_name, trade_list, log=True):
            if sig is None:
                return None
            res = _execute_signal(sig, day_df)
            if res is None:
                return None
            record = {"date": str(date), "strategy": strategy_name, **res}
            trade_list.append(record)
            if log:
                trade_log.append(record)
            return record

        _simulate_if_signal(orb_sig,  "ORB",      orb_trades)
        _simulate_if_signal(ema_sig,  "EMA_Cross", ema_trades)
        _simulate_if_signal(mom_sig,  "Momentum", mom_trades)

        rec = _simulate_if_signal(_pick_combined_signal([orb_sig, ema_sig, mom_sig]), "Combined",
                                  combined_trades, log=False)
        if rec is not None:
            cum_combined += rec["pnl_rs"]
            equity_curve.append({"date": str(date), "pnl_rs": round(rec["pnl_rs"], 2),
                                 "cum_pnl_rs": round(cum_combined, 2)})

    return {
        "symbol":               clean,
        "period_days":          days,
        "trading_days_tested":  len(trading_dates),
        "data_bars":            len(df),
        "trade_notional_rs":    TRADE_NOTIONAL,
        "strategies": {
            "ORB":      _build_stats(orb_trades),
            "EMA_Cross": _build_stats(ema_trades),
            "Momentum": _build_stats(mom_trades),
            "Combined": _build_stats(combined_trades),
        },
        "trade_log":    trade_log,
        "combined_log": combined_trades,
        "equity_curve": equity_curve,
        "data_source": data_source,
        "synthetic_data": is_synthetic,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Backtest uses ₹1L notional per trade. Entry at the next bar's open after the signal bar. "
            "ATR-based SL/TP (1× ATR SL, 3× ATR TP, floor 0.3%/0.9%). Stop checked before target, "
            "gap-aware fills; trailing SL (breakeven at 50% to TP, trail at 75%) applies from the next bar. "
            "Slippage: 0.1% per side + 0.086% fees. ORB volume filter: 1.1× the day's average so far. "
            "Sharpe: daily-aggregated returns. Auto square-off at 15:15 IST open."
        ),
    }


# ─── Universe-driven backtest (Task 3 — dynamic stock selection + Layer-1) ───

def run_universe_backtest(days: int = 30, top_n_per_strategy: int = 15,
                          use_weights: bool = True, freeze_weights: bool = True,
                          symbols: Optional[list] = None) -> dict:
    """
    Backtest every symbol of the intraday universe (or `symbols`).

    By default (`freeze_weights=True`) the run is out-of-sample: nothing is
    learned from the backtest trades and no Layer-1 weight filter is applied,
    and stocks are NOT pre-selected with rank_universe() (that ranks on today's
    data, which overlaps the test window). `top_n_per_strategy` and
    `use_weights` are reported only.

    `freeze_weights=False` restores the old learn-and-filter behaviour — the
    result is then flagged `in_sample: True`.
    """
    from app.services.intraday_universe import INTRADAY_UNIVERSE
    from app.services.signal_weights import signal_weights

    strategies = ["ORB", "EMA_Cross", "Momentum", "Combined"]
    syms = list(symbols or INTRADAY_UNIVERSE)
    filter_on = use_weights and not freeze_weights
    logger.info(f"[universe-backtest] days={days} symbols={len(syms)} freeze={freeze_weights}")

    per_strategy_trades: dict = {s: [] for s in strategies}
    per_symbol_results: dict = {}
    skipped_by_weight = 0

    for sym in syms:
        res = run_intraday_backtest(sym, days=days)
        if not res or "error" in res:
            continue
        logs = {"Combined": []}
        for t in res["trade_log"]:
            logs.setdefault(t["strategy"], []).append(t)
        for strategy in strategies:
            if filter_on and strategy != "Combined" and signal_weights.weight(strategy, sym) < 0.3:
                skipped_by_weight += 1
                continue
            stats = res["strategies"].get(strategy, {})
            per_symbol_results.setdefault(sym, {})[strategy] = {
                "trades": stats.get("trades", 0), "win_rate": stats.get("win_rate_pct", 0),
                "pnl_rs": stats.get("total_pnl_rs", 0)}
            for t in res.get("combined_log", []) if strategy == "Combined" else logs.get(strategy, []):
                per_strategy_trades[strategy].append({**t, "symbol": sym})
                if not freeze_weights and strategy != "Combined":
                    signal_weights.update(strategy, sym, t["pnl_rs"])

    aggregate: dict = {}
    for strat, trades in per_strategy_trades.items():
        if not trades:
            aggregate[strat] = {"trades": 0, "win_rate_pct": 0, "total_pnl_rs": 0}
            continue
        wins = [t for t in trades if t["pnl_rs"] > 0]
        total = sum(t["pnl_rs"] for t in trades)
        aggregate[strat] = {
            "trades": len(trades), "wins": len(wins), "losses": len(trades) - len(wins),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "total_pnl_rs": round(total, 2), "avg_pnl_rs": round(total / len(trades), 2),
            "avg_pnl_pct": round(sum(t["pnl_pct"] for t in trades) / len(trades), 3),
            "gross_avg_pnl_pct": round(sum(t.get("gross_pnl_pct", 0) for t in trades) / len(trades), 3),
            "symbols_traded": len({t["symbol"] for t in trades}),
        }

    return {
        "period_days": days,
        "top_n_per_strategy": top_n_per_strategy,
        "use_weights": filter_on,
        "use_weights_requested": use_weights,
        "freeze_weights": freeze_weights,
        "in_sample": not freeze_weights,
        "skipped_by_weight": skipped_by_weight,
        "universe_used": {s: syms for s in strategies},
        "aggregate": aggregate,
        "per_symbol": per_symbol_results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
