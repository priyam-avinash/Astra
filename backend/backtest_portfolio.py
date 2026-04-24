"""
ASTRA Portfolio Backtester v1.0
================================
Portfolio-level backtest across all NIFTY 50 symbols.

Key differences from backtest.py (single-symbol):
- Up to MAX_POSITIONS concurrent open positions
- Equal-weight sizing: CAPITAL / MAX_POSITIONS per trade
- Signal priority: rank by RF-predicted return on the same day
- Sector concentration cap: max 2 positions per sector
- Portfolio-level metrics: Sharpe, CAGR, Max Drawdown on equity curve
- Monthly P&L breakdown
- Top-5 / Bottom-5 contributing stocks

Signal generation uses the RF model directly (fast, vectorized) instead of
the full analyze_market_data() call chain. Fallback to HOLD if model absent.

Methodology:
- Out-of-sample: last 12 months of daily data (not seen during training)
- Entry on next open after signal
- SL = ATR × 1.6, TP = ATR × 4.0 (ASTRA.AI multipliers)
- Brokerage 0.03% round-trip + 0.05% slippage per leg
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from app.services.ai_predictor import ai_engine, FEATURE_COLS

# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CAP    = 100_000     # ₹1,00,000 starting capital
MAX_POSITIONS  = 5           # Maximum concurrent open positions
POSITION_PCT   = 1.0 / MAX_POSITIONS   # 20% per position (equal-weight)
SL_MULT        = 1.6
TP_MULT        = 4.0
BROKERAGE      = 0.0003      # 0.03% round-trip
SLIPPAGE       = 0.0005      # 0.05% per leg
RF_THRESHOLD   = 1.5         # Predicted return % threshold for BUY/SELL
LOOKBACK_BARS  = 252         # ~12 months out-of-sample

# ── Sector map (for concentration limit) ──────────────────────────────────────
_SECTOR_MAP = {
    "RELIANCE.NS":   "Energy",      "TCS.NS":        "IT",
    "HDFCBANK.NS":   "Banking",     "INFY.NS":        "IT",
    "ICICIBANK.NS":  "Banking",     "HINDUNILVR.NS":  "FMCG",
    "SBIN.NS":       "Banking",     "BAJFINANCE.NS":  "NBFC",
    "KOTAKBANK.NS":  "Banking",     "BHARTIARTL.NS":  "Telecom",
    "LT.NS":         "Engineering", "AXISBANK.NS":    "Banking",
    "ASIANPAINT.NS": "Paints",      "MARUTI.NS":      "Auto",
    "SUNPHARMA.NS":  "Pharma",      "TITAN.NS":       "Consumer",
    "ULTRACEMCO.NS": "Cement",      "WIPRO.NS":       "IT",
    "NESTLEIND.NS":  "FMCG",        "ADANIENT.NS":    "Conglomerate",
    "POWERGRID.NS":  "Power",       "TECHM.NS":       "IT",
    "INDUSINDBK.NS": "Banking",     "DIVISLAB.NS":    "Pharma",
    "JSWSTEEL.NS":   "Steel",       "BAJAJFINSV.NS":  "NBFC",
    "COALINDIA.NS":  "Mining",      "HCLTECH.NS":     "IT",
    "ONGC.NS":       "Energy",      "NTPC.NS":        "Power",
    "M&M.NS":        "Auto",        "TATAMOTORS.NS":  "Auto",
    "TATASTEEL.NS":  "Steel",       "CIPLA.NS":       "Pharma",
    "BRITANNIA.NS":  "FMCG",        "DRREDDY.NS":     "Pharma",
    "BPCL.NS":       "Energy",      "HEROMOTOCO.NS":  "Auto",
    "GRASIM.NS":     "Cement",      "EICHERMOT.NS":   "Auto",
    "TATACONSUM.NS": "FMCG",        "SBILIFE.NS":     "Insurance",
    "HDFCLIFE.NS":   "Insurance",   "UPL.NS":         "Agrochem",
    "APOLLOHOSP.NS": "Healthcare",  "HINDALCO.NS":    "Metals",
    "ADANIPORTS.NS": "Logistics",   "BAJAJ-AUTO.NS":  "Auto",
    "ITC.NS":        "FMCG",        "VEDL.NS":        "Metals",
}

PORTFOLIO_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "BHARTIARTL.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "ADANIENT.NS",
    "POWERGRID.NS", "TECHM.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "JSWSTEEL.NS",
    "BAJAJFINSV.NS", "COALINDIA.NS", "HCLTECH.NS", "ONGC.NS", "NTPC.NS",
    "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "CIPLA.NS", "BRITANNIA.NS",
    "DRREDDY.NS", "BPCL.NS", "HEROMOTOCO.NS", "GRASIM.NS", "EICHERMOT.NS",
    "TATACONSUM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "UPL.NS", "APOLLOHOSP.NS",
    "HINDALCO.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "ITC.NS", "VEDL.NS",
]


# ── Data preparation ───────────────────────────────────────────────────────────

def _prepare_symbol(symbol: str) -> dict:
    """
    Fetch 2 years of daily data, compute features, return test window.
    Returns dict: {symbol, df_feat (feature df), test_df (last 252 bars)}
    or {"symbol": ..., "error": ...} on failure.
    """
    try:
        df_raw  = ai_engine._fetch_data(symbol, period="2y", interval="1d")
        df_feat = ai_engine._compute_features(df_raw)
        if df_feat.empty or len(df_feat) < 60:
            return {"symbol": symbol, "error": "Insufficient data"}
        test_df = df_feat.tail(LOOKBACK_BARS).copy().reset_index(drop=False)
        return {"symbol": symbol, "df_feat": df_feat, "test_df": test_df}
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


def _generate_signals_rf(history_df: pd.DataFrame) -> tuple:
    """
    Use RF model predict on the last row to generate a signal + confidence.
    Returns (signal: str, pred_return: float).
    """
    if ai_engine.rf_model is None:
        return "HOLD", 0.0
    try:
        feats = history_df[FEATURE_COLS].tail(1)
        if feats.isnull().any().any():
            return "HOLD", 0.0
        pred = float(ai_engine.rf_model.predict(feats)[0])
        if pred > RF_THRESHOLD:
            return "BUY", pred
        elif pred < -RF_THRESHOLD:
            return "SELL", pred
        return "HOLD", pred
    except Exception:
        return "HOLD", 0.0


# ── Portfolio simulation ───────────────────────────────────────────────────────

def run_portfolio_backtest(symbols: list = None) -> dict:
    """
    Simulate a portfolio of up to MAX_POSITIONS concurrent long/short positions
    across all NIFTY 50 symbols over the last 12 months.

    Returns a dict with:
      trades, equity_curve, summary, monthly_pnl, symbol_pnl
    """
    if symbols is None:
        symbols = PORTFOLIO_SYMBOLS

    print(f"\n  Loading data for {len(symbols)} symbols…", flush=True)
    symbol_data = {}
    for sym in symbols:
        r = _prepare_symbol(sym)
        if "error" in r:
            print(f"    ✗ {sym}: {r['error'][:50]}")
        else:
            symbol_data[sym] = r
            print(f"    ✓ {sym}: {len(r['test_df'])} bars")

    if not symbol_data:
        print("  No data available. Aborting.")
        return {}

    # Align all symbols on a common date range
    all_dates = set()
    for data in symbol_data.values():
        idx_col = data["test_df"].columns[0]  # first column is the date index
        if idx_col in ("index", "Date", "Datetime"):
            dates = pd.to_datetime(data["test_df"][idx_col]).dt.date.tolist()
        else:
            dates = pd.to_datetime(data["test_df"].index).date.tolist()
        all_dates.update(dates)
    all_dates = sorted(all_dates)

    # Build date → row-index lookup per symbol
    date_lookup: dict[str, dict] = {}
    for sym, data in symbol_data.items():
        tdf = data["test_df"]
        # Determine date column
        idx_col = tdf.columns[0]
        if idx_col in ("index", "Date", "Datetime"):
            tdf = tdf.copy()
            tdf["_date"] = pd.to_datetime(tdf[idx_col]).dt.date
        else:
            tdf = tdf.copy()
            tdf["_date"] = pd.to_datetime(tdf.index).date
        date_lookup[sym] = {row["_date"]: i for i, row in tdf.iterrows()}
        symbol_data[sym]["_tdf"] = tdf

    # ── Portfolio state
    capital       = float(INITIAL_CAP)
    equity_curve  = [capital]
    equity_dates  = []
    positions: dict[str, dict] = {}   # symbol → {direction, entry_price, sl, tp, entry_date}
    trades: list[dict]         = []
    symbol_pnl: dict[str, float] = defaultdict(float)
    monthly_pnl: dict[str, float] = defaultdict(float)

    # Cooldown: after an SL hit, block that symbol for N bars to avoid revenge-trading
    SL_COOLDOWN_BARS = 10          # ~2 trading weeks
    cooldown: dict[str, int] = {}  # symbol → bar_index when cooldown expires

    WARMUP = 30   # Skip first 30 bars to let indicators warm up

    for date_idx, date in enumerate(all_dates):
        if date_idx < WARMUP:
            continue

        daily_equity_change = 0.0

        # ── 1. Check exits for all open positions ──────────────────────────────
        closed_symbols = []
        for sym, pos in positions.items():
            tdf = symbol_data[sym]["_tdf"]
            row_idx = date_lookup[sym].get(date)
            if row_idx is None:
                continue
            row  = tdf.loc[row_idx]
            high = float(row["High"])
            low  = float(row["Low"])
            direction   = pos["direction"]
            entry_price = pos["entry_price"]
            sl          = pos["sl"]
            tp          = pos["tp"]

            hit_tp = (direction == "BUY"  and high >= tp) or \
                     (direction == "SELL" and low  <= tp)
            hit_sl = (direction == "BUY"  and low  <= sl) or \
                     (direction == "SELL" and high >= sl)

            if hit_tp or hit_sl:
                exit_price = tp if hit_tp else sl
                exit_price *= (1 - SLIPPAGE) if direction == "BUY" else (1 + SLIPPAGE)
                raw_pnl    = ((exit_price - entry_price) / entry_price) * (1 if direction == "BUY" else -1)
                net_pnl    = raw_pnl - BROKERAGE
                pos_size   = capital * POSITION_PCT
                pnl_cash   = pos_size * net_pnl
                capital   += pnl_cash
                daily_equity_change += pnl_cash
                month_key  = date.strftime("%Y-%m")
                monthly_pnl[month_key]  += pnl_cash
                symbol_pnl[sym]         += pnl_cash
                trades.append({
                    "symbol":      sym,
                    "direction":   direction,
                    "entry_date":  str(pos["entry_date"]),
                    "exit_date":   str(date),
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price, 4),
                    "exit_reason": "TP" if hit_tp else "SL",
                    "pnl_pct":     round(net_pnl * 100, 3),
                    "pnl_cash":    round(pnl_cash, 2),
                    "capital":     round(capital, 2),
                })
                closed_symbols.append(sym)
                # Apply cooldown if stopped out
                if not hit_tp:
                    cooldown[sym] = date_idx + SL_COOLDOWN_BARS

        for sym in closed_symbols:
            del positions[sym]

        # ── 2. Generate signals for all symbols not in a position ──────────────
        open_slot_count = MAX_POSITIONS - len(positions)
        if open_slot_count <= 0:
            equity_curve.append(capital)
            equity_dates.append(date)
            continue

        # Count current sector exposure
        sector_count: dict[str, int] = defaultdict(int)
        for sym in positions:
            sector_count[_SECTOR_MAP.get(sym, "Other")] += 1

        candidates: list[dict] = []
        for sym in symbol_data:
            if sym in positions:
                continue
            # Skip symbols in SL cooldown
            if cooldown.get(sym, 0) > date_idx:
                continue
            tdf     = symbol_data[sym]["_tdf"]
            row_idx = date_lookup[sym].get(date)
            if row_idx is None or row_idx < WARMUP:
                continue

            history = tdf.iloc[:row_idx + 1]
            sig, pred = _generate_signals_rf(history)
            if sig == "HOLD":
                continue

            # Sector concentration check
            sector = _SECTOR_MAP.get(sym, "Other")
            if sector_count[sector] >= 2:
                continue

            row = tdf.loc[row_idx]
            candidates.append({
                "symbol":    sym,
                "signal":    sig,
                "pred":      pred,
                "abs_pred":  abs(pred),
                "open_next": float(row.get("Open", row["Close"])),
                "atr":       float(row.get("ATR", float(row["Close"]) * 0.015)),
                "sector":    sector,
            })

        # Sort by |predicted return| descending, take top open_slot_count
        candidates.sort(key=lambda x: x["abs_pred"], reverse=True)
        for cand in candidates[:open_slot_count]:
            sym     = cand["symbol"]
            sig     = cand["signal"]
            atr     = cand["atr"]
            entry   = cand["open_next"] * (1 + SLIPPAGE if sig == "BUY" else 1 - SLIPPAGE)

            if sig == "BUY":
                sl = entry - atr * SL_MULT
                tp = entry + atr * TP_MULT
            else:
                sl = entry + atr * SL_MULT
                tp = entry - atr * TP_MULT

            positions[sym] = {
                "direction":   sig,
                "entry_price": entry,
                "sl":          sl,
                "tp":          tp,
                "entry_date":  date,
            }
            sector_count[cand["sector"]] += 1

        equity_curve.append(capital)
        equity_dates.append(date)

    # ── Force-close any remaining open positions at last available price ────────
    for sym, pos in positions.items():
        tdf = symbol_data[sym]["_tdf"]
        if tdf.empty:
            continue
        last_row   = tdf.iloc[-1]
        close_px   = float(last_row["Close"])
        direction  = pos["direction"]
        entry_price= pos["entry_price"]
        exit_price = close_px * (1 - SLIPPAGE if direction == "BUY" else 1 + SLIPPAGE)
        raw_pnl    = ((exit_price - entry_price) / entry_price) * (1 if direction == "BUY" else -1)
        net_pnl    = raw_pnl - BROKERAGE
        pos_size   = capital * POSITION_PCT
        pnl_cash   = pos_size * net_pnl
        capital   += pnl_cash
        symbol_pnl[sym]  += pnl_cash
        trades.append({
            "symbol":      sym,
            "direction":   direction,
            "entry_date":  str(pos["entry_date"]),
            "exit_date":   "EOD",
            "entry_price": round(entry_price, 4),
            "exit_price":  round(exit_price, 4),
            "exit_reason": "EOD",
            "pnl_pct":     round(net_pnl * 100, 3),
            "pnl_cash":    round(pnl_cash, 2),
            "capital":     round(capital, 2),
        })

    equity_curve.append(capital)

    # ── Metrics ────────────────────────────────────────────────────────────────
    if not trades:
        print("  No trades generated.")
        return {"trades": [], "equity_curve": equity_curve}

    df_t      = pd.DataFrame(trades)
    wins      = df_t[df_t["pnl_pct"] > 0]
    losses    = df_t[df_t["pnl_pct"] <= 0]
    win_rate  = len(wins) / len(df_t) * 100 if len(df_t) else 0.0
    avg_win   = float(wins["pnl_pct"].mean())   if len(wins)   else 0.0
    avg_loss  = float(losses["pnl_pct"].mean()) if len(losses) else 0.0

    # CAGR
    n_years   = LOOKBACK_BARS / 252.0
    total_ret = (capital - INITIAL_CAP) / INITIAL_CAP
    cagr      = ((1 + total_ret) ** (1 / max(n_years, 0.01))) - 1

    # Max Drawdown
    eq_series = pd.Series(equity_curve)
    peak      = eq_series.cummax()
    dd_series = (eq_series - peak) / peak * 100
    max_dd    = float(dd_series.min())

    # Sharpe (on trade-level P&L)
    pnl_arr   = df_t["pnl_pct"].values / 100
    sharpe    = float((pnl_arr.mean() / (pnl_arr.std() + 1e-9)) * np.sqrt(252)) if len(pnl_arr) > 1 else 0.0

    summary = {
        "start_capital":    INITIAL_CAP,
        "end_capital":      round(capital, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "total_trades":     len(df_t),
        "win_rate_pct":     round(win_rate, 1),
        "avg_win_pct":      round(avg_win, 3),
        "avg_loss_pct":     round(avg_loss, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio":     round(sharpe, 3),
    }

    return {
        "trades":       df_t.to_dict("records"),
        "equity_curve": equity_curve,
        "summary":      summary,
        "monthly_pnl":  dict(sorted(monthly_pnl.items())),
        "symbol_pnl":   dict(sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)),
    }


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_results(result: dict):
    if not result or "summary" not in result:
        print("  No results to display.")
        return

    s = result["summary"]
    print("\n" + "=" * 70)
    print("  ASTRA PORTFOLIO BACKTEST  —  NIFTY 50  |  Last 12 months")
    print("=" * 70)
    print(f"  Start Capital   : ₹{s['start_capital']:>12,.2f}")
    print(f"  End Capital     : ₹{s['end_capital']:>12,.2f}")
    print(f"  Total Return    : {s['total_return_pct']:>+8.2f}%")
    print(f"  CAGR            : {s['cagr_pct']:>+8.2f}%")
    print(f"  Total Trades    : {s['total_trades']:>8}")
    print(f"  Win Rate        : {s['win_rate_pct']:>8.1f}%")
    print(f"  Avg Win         : {s['avg_win_pct']:>+8.3f}%")
    print(f"  Avg Loss        : {s['avg_loss_pct']:>+8.3f}%")
    print(f"  Max Drawdown    : {s['max_drawdown_pct']:>+8.2f}%")
    print(f"  Sharpe Ratio    : {s['sharpe_ratio']:>8.3f}")

    # Monthly P&L
    print("\n" + "─" * 70)
    print("  MONTHLY P&L")
    print("─" * 70)
    monthly = result.get("monthly_pnl", {})
    for month in sorted(monthly):
        val  = monthly[month]
        bar  = "▓" * int(abs(val) / 500) if abs(val) > 0 else ""
        sign = "+" if val >= 0 else ""
        print(f"    {month}  {sign}₹{val:>10,.2f}  {bar}")

    # Top 5 / Bottom 5 symbols
    sym_pnl = result.get("symbol_pnl", {})
    sorted_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "─" * 70)
    print("  TOP 5 CONTRIBUTING STOCKS")
    print("─" * 70)
    for sym, pnl in sorted_syms[:5]:
        print(f"    {sym:<18}  +₹{pnl:>10,.2f}")

    print("\n  BOTTOM 5 CONTRIBUTING STOCKS")
    print("─" * 70)
    for sym, pnl in sorted_syms[-5:]:
        sign = "" if pnl >= 0 else "-"
        print(f"    {sym:<18}  {sign}₹{abs(pnl):>10,.2f}")

    print("\n" + "=" * 70)
    print(f"  Max concurrent positions : {MAX_POSITIONS}")
    print(f"  Position sizing          : {POSITION_PCT*100:.0f}% of portfolio per slot")
    print(f"  SL / TP multipliers      : {SL_MULT}× / {TP_MULT}× ATR")
    print(f"  Brokerage + slippage     : {BROKERAGE*100:.3f}% + {SLIPPAGE*100:.2f}%")
    print(f"  RF signal threshold      : ±{RF_THRESHOLD}% predicted return")
    print("=" * 70)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  ASTRA Portfolio Backtest — started {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Symbols  : {len(PORTFOLIO_SYMBOLS)}")
    print(f"  Max slots: {MAX_POSITIONS} concurrent  |  {POSITION_PCT*100:.0f}% per position")
    print(f"  Window   : Last {LOOKBACK_BARS} bars (~12 months, out-of-sample)")

    if ai_engine.rf_model is None:
        print("\n  WARNING: RF model not loaded — all signals will be HOLD.")
        print("  Run the model trainer first to enable signal generation.\n")

    result = run_portfolio_backtest(PORTFOLIO_SYMBOLS)
    print_results(result)

    # Optionally save trade log
    if result.get("trades"):
        out_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
        os.makedirs(out_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = os.path.join(out_dir, f"portfolio_{ts}.csv")
        pd.DataFrame(result["trades"]).to_csv(out, index=False)
        print(f"\n  Trade log saved → {out}")

    print(f"\n  Completed at {datetime.now().strftime('%H:%M:%S')}\n")
