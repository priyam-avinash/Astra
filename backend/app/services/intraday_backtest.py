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
SLIPPAGE_PCT    = 0.001    # 0.1% per side (realistic NSE intraday slippage)


# ── Per-trade simulation ──────────────────────────────────────────────────────

def _simulate_trade(signal: str, entry_price: float, sl: float, tp: float,
                    future_bars: pd.DataFrame) -> dict:
    """
    Walk-forward bar-by-bar simulation with:
    - Slippage (0.1% per side) applied at entry
    - Trailing SL: breakeven after 50% of move to TP, then 50%-of-TP trail after 75%
    - SL checked before TP on each bar (conservative, worst-case intrabar)
    - Auto square-off at 15:15 IST

    Returns dict with: outcome, exit_price, pnl_pct, pnl_rs, bars_held
    """
    sq_off = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])

    # Apply slippage at entry
    if signal == "BUY":
        fill_price = entry_price * (1 + SLIPPAGE_PCT)
    else:
        fill_price = entry_price * (1 - SLIPPAGE_PCT)

    # Dynamic trailing SL (starts at original SL, tracks profit)
    trailing_sl = sl
    move_to_tp  = abs(tp - fill_price)   # total move from fill to TP

    for i, (ts, bar) in enumerate(future_bars.iterrows()):
        bar_time = ts.time() if hasattr(ts, "time") else ts.to_pydatetime().time()

        high  = float(bar["High"])
        low   = float(bar["Low"])
        close = float(bar["Close"])

        # Update trailing SL based on how far price has moved toward TP
        if move_to_tp > 0:
            if signal == "BUY":
                progress = (high - fill_price) / move_to_tp
                if progress >= 0.75:
                    new_trail = fill_price + move_to_tp * 0.50   # trail to 50% of TP
                    trailing_sl = max(trailing_sl, new_trail)
                elif progress >= 0.50:
                    trailing_sl = max(trailing_sl, fill_price)    # breakeven
            else:  # SELL
                progress = (fill_price - low) / move_to_tp
                if progress >= 0.75:
                    new_trail = fill_price - move_to_tp * 0.50
                    trailing_sl = min(trailing_sl, new_trail)
                elif progress >= 0.50:
                    trailing_sl = min(trailing_sl, fill_price)    # breakeven

        if signal == "BUY":
            if low <= trailing_sl:
                exit_price = trailing_sl
                pnl_pct    = (exit_price - fill_price) / fill_price
                return _trade_result("SL", exit_price, pnl_pct, i + 1)
            if high >= tp:
                exit_price = tp
                pnl_pct    = (exit_price - fill_price) / fill_price
                return _trade_result("TP", exit_price, pnl_pct, i + 1)
        else:  # SELL
            if high >= trailing_sl:
                exit_price = trailing_sl
                pnl_pct    = (fill_price - exit_price) / fill_price
                return _trade_result("SL", exit_price, pnl_pct, i + 1)
            if low <= tp:
                exit_price = tp
                pnl_pct    = (fill_price - exit_price) / fill_price
                return _trade_result("TP", exit_price, pnl_pct, i + 1)

        # Auto square-off at 15:15
        if bar_time >= sq_off:
            exit_price = close * (1 - SLIPPAGE_PCT) if signal == "BUY" else close * (1 + SLIPPAGE_PCT)
            if signal == "BUY":
                pnl_pct = (exit_price - fill_price) / fill_price
            else:
                pnl_pct = (fill_price - exit_price) / fill_price
            return _trade_result("SQ_OFF", exit_price, pnl_pct, i + 1)

    # End of data without resolution → square-off at last bar close
    if not future_bars.empty:
        raw_exit   = float(future_bars.iloc[-1]["Close"])
        exit_price = raw_exit * (1 - SLIPPAGE_PCT) if signal == "BUY" else raw_exit * (1 + SLIPPAGE_PCT)
        pnl_pct    = (exit_price - fill_price) / fill_price if signal == "BUY" else (fill_price - exit_price) / fill_price
        return _trade_result("SQ_OFF", exit_price, pnl_pct, len(future_bars))

    return _trade_result("NO_EXIT", entry_price, 0.0, 0)


def _trade_result(outcome: str, exit_price: float, pnl_pct: float, bars_held: int) -> dict:
    shares   = int(TRADE_NOTIONAL / max(exit_price, 1))
    pnl_rs   = round(pnl_pct * TRADE_NOTIONAL, 2)
    return {
        "outcome":    outcome,
        "exit_price": round(exit_price, 2),
        "pnl_pct":    round(pnl_pct * 100, 3),   # in %
        "pnl_rs":     pnl_rs,
        "bars_held":  bars_held,
    }


# ── Strategy signal generators (bar-by-bar, up to each bar) ──────────────────

def _get_orb_signal_for_day(day_df: pd.DataFrame, date) -> Optional[dict]:
    """
    Return the FIRST actionable ORB signal for the day, or None if HOLD throughout.
    """
    orb_high, orb_low = _compute_orb(day_df, date)
    if orb_high is None:
        return None

    avg_vol  = day_df["Volume"].mean() if not day_df.empty else 0
    orb_end  = dtime(9, 15 + ORB_WINDOW_MIN)
    no_trade = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])

    post_orb = day_df[(day_df.index.time >= orb_end) & (day_df.index.time <= no_trade)]

    for ts, bar in post_orb.iterrows():
        close  = float(bar["Close"])
        # Require at least 1.5× average volume — surge volume confirms genuine breakout
        vol_ok = float(bar["Volume"]) >= 1.5 * avg_vol if avg_vol > 0 else True
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
    df["VWAP"]  = _compute_vwap(df)
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
            if not _trend_aligned(full_df, "BUY"):
                continue
            sl, tp = _atr_sl_tp(df.iloc[:i+1], close, "BUY")
            return {"signal": "BUY",  "entry_price": close, "sl": sl, "tp": tp, "ts": ts}

        if bear_cross and rsi < 50 and close < vwap:
            if not _trend_aligned(full_df, "SELL"):
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
                if not _trend_aligned(full_df, "BUY"):
                    prev_row = row
                    continue
                sl, tp = _atr_sl_tp(df.loc[:ts], close, "BUY")
                return {"signal": "BUY",  "entry_price": close, "sl": sl, "tp": tp, "ts": ts}
            elif rsi < 45 and hist < 0 and bear_cross:
                if not _trend_aligned(full_df, "SELL"):
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
        return {"error": f"No intraday data available for {clean}", "symbol": clean}

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

        def _simulate_if_signal(sig, strategy_name, trade_list):
            if sig is None:
                return
            entry_ts = sig["ts"]
            future   = day_df[day_df.index > entry_ts]
            if future.empty:
                return
            res = _simulate_trade(sig["signal"], sig["entry_price"], sig["sl"], sig["tp"], future)
            record = {
                "date":       str(date),
                "strategy":   strategy_name,
                "signal":     sig["signal"],
                "entry_price":sig["entry_price"],
                "sl":         sig["sl"],
                "tp":         sig["tp"],
                **res,
            }
            trade_list.append(record)
            trade_log.append(record)

        _simulate_if_signal(orb_sig,  "ORB",      orb_trades)
        _simulate_if_signal(ema_sig,  "EMA_Cross", ema_trades)
        _simulate_if_signal(mom_sig,  "Momentum", mom_trades)

        # Combined: prefer consensus direction, then pick highest-quality signal
        signals_today = [s for s in [orb_sig, ema_sig, mom_sig] if s is not None]
        if signals_today:
            buys  = [s for s in signals_today if s["signal"] == "BUY"]
            sells = [s for s in signals_today if s["signal"] == "SELL"]
            dominant = buys if len(buys) >= len(sells) else sells

            def _signal_score(sig: dict) -> float:
                """Higher R:R and later timestamp (more confirmation) = better signal."""
                entry = sig.get("entry_price", 1.0)
                sl    = sig.get("sl", entry)
                tp    = sig.get("tp", entry)
                risk  = abs(entry - sl)
                reward = abs(tp - entry)
                rr    = reward / risk if risk > 0 else 0.0
                # Prefer higher R:R; tiebreak: later entry (more bars of confirmation)
                ts_score = sig.get("ts", pd.Timestamp.min)
                return (rr, ts_score)

            chosen = max(dominant, key=_signal_score)
            future = day_df[day_df.index > chosen["ts"]]
            if not future.empty:
                res = _simulate_trade(chosen["signal"], chosen["entry_price"],
                                      chosen["sl"], chosen["tp"], future)
                record = {
                    "date":       str(date),
                    "strategy":   "Combined",
                    "signal":     chosen["signal"],
                    "entry_price":chosen["entry_price"],
                    "sl":         chosen["sl"],
                    "tp":         chosen["tp"],
                    **res,
                }
                combined_trades.append(record)
                cum_combined += res["pnl_rs"]
                equity_curve.append({
                    "date":       str(date),
                    "pnl_rs":     round(res["pnl_rs"], 2),
                    "cum_pnl_rs": round(cum_combined, 2),
                })

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
        "equity_curve": equity_curve,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Backtest uses ₹1L notional per trade. ATR-based SL/TP (1× ATR SL, 3× ATR TP, "
            "floor 0.3%/0.9%). Trailing SL: breakeven at 50% to TP, trail at 75%. "
            "Slippage: 0.1% per side. ORB volume filter: 1.5× avg. "
            "Sharpe: daily-aggregated returns. Auto square-off 15:15 IST."
        ),
    }
