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
TRADE_NOTIONAL = 100_000  # ₹1 lakh per trade


# ── Per-trade simulation ──────────────────────────────────────────────────────

def _simulate_trade(signal: str, entry_price: float, sl: float, tp: float,
                    future_bars: pd.DataFrame) -> dict:
    """
    Given a signal and the bars AFTER the entry bar, simulate:
    - Hit TP  → win (return TP pct gain)
    - Hit SL  → loss
    - Hit 15:15 square-off → scratch (close at that bar's close)

    Returns dict with: outcome, exit_price, pnl_pct, pnl_rs, bars_held
    """
    sq_off = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])

    for i, (ts, bar) in enumerate(future_bars.iterrows()):
        bar_time = ts.time() if hasattr(ts, "time") else ts.to_pydatetime().time()

        high  = float(bar["High"])
        low   = float(bar["Low"])
        close = float(bar["Close"])

        if signal == "BUY":
            # Check SL first (intrabar worst-case)
            if low <= sl:
                exit_price = sl
                pnl_pct    = (exit_price - entry_price) / entry_price
                return _trade_result("SL", exit_price, pnl_pct, i + 1)
            if high >= tp:
                exit_price = tp
                pnl_pct    = (exit_price - entry_price) / entry_price
                return _trade_result("TP", exit_price, pnl_pct, i + 1)
        else:  # SELL
            if high >= sl:
                exit_price = sl
                pnl_pct    = (entry_price - exit_price) / entry_price
                return _trade_result("SL", exit_price, pnl_pct, i + 1)
            if low <= tp:
                exit_price = tp
                pnl_pct    = (entry_price - exit_price) / entry_price
                return _trade_result("TP", exit_price, pnl_pct, i + 1)

        # Auto square-off at 15:15
        if bar_time >= sq_off:
            exit_price = close
            if signal == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            return _trade_result("SQ_OFF", exit_price, pnl_pct, i + 1)

    # End of data without resolution → square-off at last bar close
    if not future_bars.empty:
        exit_price = float(future_bars.iloc[-1]["Close"])
        pnl_pct = (exit_price - entry_price) / entry_price if signal == "BUY" else (entry_price - exit_price) / entry_price
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
        # Require at least 0.8× average volume (relaxed from 1.5×: mid-session
        # breakouts are valid even without surge volume; 1.5× was too strict)
        vol_ok = float(bar["Volume"]) >= 0.8 * avg_vol if avg_vol > 0 else True
        if close > orb_high and vol_ok:
            return {"signal": "BUY",
                    "entry_price": close,
                    "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                    "tp": round(close * (1 + INTRADAY_TP_PCT), 2),
                    "entry": close, "ts": ts}
        elif close < orb_low and vol_ok:
            return {"signal": "SELL",
                    "entry_price": close,
                    "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                    "tp": round(close * (1 - INTRADAY_TP_PCT), 2),
                    "entry": close, "ts": ts}
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
            return {"signal": "BUY",  "entry_price": close,
                    "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                    "tp": round(close * (1 + INTRADAY_TP_PCT), 2), "ts": ts}

        if bear_cross and rsi < 50 and close < vwap:
            if not _trend_aligned(full_df, "SELL"):
                continue
            return {"signal": "SELL", "entry_price": close,
                    "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                    "tp": round(close * (1 - INTRADAY_TP_PCT), 2), "ts": ts}
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
                return {"signal": "BUY", "entry_price": close, "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                        "tp": round(close * (1 + INTRADAY_TP_PCT), 2), "ts": ts}
            elif rsi < 45 and hist < 0 and bear_cross:
                if not _trend_aligned(full_df, "SELL"):
                    prev_row = row
                    continue
                return {"signal": "SELL", "entry_price": close, "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                        "tp": round(close * (1 - INTRADAY_TP_PCT), 2), "ts": ts}
        prev_row = row
    return None


# ── Strategy stats builder ────────────────────────────────────────────────────

def _build_stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "total_pnl_rs": 0.0, "avg_pnl_rs": 0.0,
                "max_drawdown_pct": 0.0, "sharpe": None, "profit_factor": None}

    wins    = [t for t in trades if t["pnl_rs"] > 0]
    losses  = [t for t in trades if t["pnl_rs"] <= 0]
    pnls    = [t["pnl_rs"] for t in trades]
    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = (cum_pnl - peak)
    max_dd  = float(dd.min()) if len(dd) > 0 else 0.0

    # Sharpe (annualised, daily returns proxy)
    pnl_arr = np.array(pnls)
    sharpe  = None
    if pnl_arr.std() > 0:
        sharpe = round(float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)), 2)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
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

        # Combined: prefer consensus, else first available
        signals_today = [s for s in [orb_sig, ema_sig, mom_sig] if s is not None]
        if signals_today:
            buys  = [s for s in signals_today if s["signal"] == "BUY"]
            sells = [s for s in signals_today if s["signal"] == "SELL"]
            dominant = buys if len(buys) >= len(sells) else sells
            chosen = dominant[0]  # earliest-entry among dominant
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
            "Backtest uses ₹1L notional per trade, 0.3% SL, 0.6% TP, "
            "auto square-off at 15:15 IST. No brokerage/slippage costs included."
        ),
    }
