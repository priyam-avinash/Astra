"""
ASTRA Stage-2 Trend Following — Hypothesis Test
=================================================
2-year daily backtest of Weinstein/Minervini Stage-2 buying across the
86-stock intraday universe.

  Entry (all 4 required):
    1. Close > SMA150 (≈30-week MA — defines Stage 2)
    2. SMA150 slope > 0 over last 30 days (uptrend confirmed)
    3. Close within 8% of 252-day high (price strength filter)
    4. Volume > 1.5× 20-day avg (breakout volume confirmation)

  Exit (any 1):
    A. Close < SMA150 (stage breakdown)
    B. End of backtest window (mark-to-market exit)

  Cost model (realistic for NSE delivery / positional):
    - Slippage:           0.05% per side (large-caps, planned entries/exits)
    - STT:                0.1% on sell side (delivery equity)
    - Brokerage + exch + GST + stamp: ~0.05% round-trip
    - TOTAL ROUND-TRIP:   ~0.25%

This is a SINGLE hypothesis test — no Layer-1 weights, no parameter tuning.
We want to answer: "Does basic Stage-2 trend following on Indian equities
have positive expectancy after realistic costs?" If yes → build full module.
If no → don't.

Usage:
    cd backend && source .venv/bin/activate
    python run_stage2_backtest.py
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
TRADE_NOTIONAL       = 100_000   # ₹1L per trade
SLIPPAGE_PCT         = 0.0005    # 0.05% per side (delivery is gentler than intraday)
TRANSACTION_COST_PCT = 0.0025    # 0.25% round-trip total (delivery STT + fees)

SMA_PERIOD           = 150       # ≈30-week MA
SMA_SLOPE_LOOKBACK   = 30        # SMA rising over last 30 days
HIGH_PROXIMITY_PCT   = 0.99      # iter-3: Close must be ≥ 99% of 252-day high (real new high)
VOL_MULTIPLIER       = 1.5       # Today's volume > 1.5× 20-day avg

# Confirmation filters (iteration 2 + 3)
MIN_6M_RETURN        = 0.15      # iter-3: stock must be up > 15% over 6 months (was 5%)
USE_NIFTY_REGIME     = True      # Skip entries when NIFTY is below its own SMA150
NIFTY_PROXY_TICKER   = "^NSEI"   # via yahoo_service — single fetch, cached

# Smarter exits (iteration 2 + 3)
HARD_STOP_PCT        = -0.10     # iter-3: -10% hard stop (was -7%; Indian stocks too volatile for tight)
TRAIL_ACTIVATE_PCT   = 0.05      # start trailing only after +5% unrealized
TRAIL_DRAWDOWN_PCT   = 0.15      # exit if 15% below peak unrealized

# Loaded once at run-time, reused across all symbols
from typing import Optional
_NIFTY_REGIME: Optional[pd.DataFrame] = None


# ── NIFTY market regime (loaded once) ───────────────────────────────────────

def _load_nifty_regime() -> Optional[pd.DataFrame]:
    """Fetch NIFTY 50 daily 2y data and pre-compute SMA150. Returns None on failure."""
    global _NIFTY_REGIME
    if _NIFTY_REGIME is not None:
        return _NIFTY_REGIME
    try:
        from app.services.yahoo_finance import yahoo_service
        # Try multiple symbols — yfinance sometimes serves NIFTY under different keys
        for tick in (NIFTY_PROXY_TICKER, "NIFTY", "NSEI"):
            df = yahoo_service.get_ohlcv(tick, period="2y", interval="1d")
            if df is not None and not df.empty and len(df) >= SMA_PERIOD + 30:
                df = df.sort_index().copy()
                df["SMA"] = df["Close"].rolling(SMA_PERIOD).mean()
                _NIFTY_REGIME = df
                logger.info(f"NIFTY regime loaded via {tick}: {len(df)} bars")
                return _NIFTY_REGIME
    except Exception as e:
        logger.warning(f"NIFTY regime load failed: {e}")

    # Fallback — proxy via NIFTYBEES ETF on Upstox if Yahoo is dead
    try:
        from app.services.upstox_data import upstox_service
        if upstox_service.is_available():
            df = upstox_service.get_ohlcv("NIFTYBEES", period="2y", interval="1d")
            if df is not None and not df.empty and len(df) >= SMA_PERIOD + 30:
                df = df.sort_index().copy()
                df["SMA"] = df["Close"].rolling(SMA_PERIOD).mean()
                _NIFTY_REGIME = df
                logger.info(f"NIFTY regime loaded via NIFTYBEES (Upstox): {len(df)} bars")
                return _NIFTY_REGIME
    except Exception as e:
        logger.warning(f"NIFTYBEES fallback failed: {e}")

    logger.warning("NIFTY regime UNAVAILABLE — regime filter will be disabled")
    return None


def _nifty_in_stage2(date) -> bool:
    """True if NIFTY's Close > NIFTY's SMA150 on or just before `date`."""
    if _NIFTY_REGIME is None:
        return True   # fail-open: if we can't load NIFTY, don't block entries
    try:
        # asof lookup — find most recent NIFTY bar at or before `date`
        idx = _NIFTY_REGIME.index.get_indexer([date], method="pad")[0]
        if idx < 0:
            return True
        row = _NIFTY_REGIME.iloc[idx]
        if pd.isna(row["SMA"]):
            return True
        return float(row["Close"]) > float(row["SMA"])
    except Exception:
        return True


# ── Single-symbol backtest ─────────────────────────────────────────────────

def backtest_stage2(symbol: str, df: pd.DataFrame) -> list:
    """Run Stage-2 logic on one symbol's daily OHLCV. Returns list of trade dicts."""
    if df is None or len(df) < SMA_PERIOD + 130:   # need 126-bar return + SMA + buffer
        return []

    df = df.sort_index().copy()
    df["SMA"]        = df["Close"].rolling(SMA_PERIOD).mean()
    df["SMA_slope"]  = df["SMA"].diff(SMA_SLOPE_LOOKBACK)
    df["52w_high"]   = df["High"].rolling(252).max()
    df["vol_avg"]    = df["Volume"].rolling(20).mean()
    df["ret_6m"]     = df["Close"].pct_change(126)      # NEW: 6-month return

    trades = []
    in_pos = False
    entry_price = entry_idx = peak_price = None
    entry_date_label = ""

    for i in range(SMA_PERIOD + 130, len(df) - 1):
        row  = df.iloc[i]
        nxt  = df.iloc[i + 1]

        # Skip if any rolling indicator missing
        if pd.isna(row["SMA"]) or pd.isna(row["52w_high"]) or pd.isna(row["vol_avg"]) or pd.isna(row["ret_6m"]):
            continue

        if not in_pos:
            cond1 = row["Close"]   > row["SMA"]
            cond2 = row["SMA_slope"] > 0
            cond3 = row["Close"]   >= HIGH_PROXIMITY_PCT * row["52w_high"]
            cond4 = row["Volume"]  > VOL_MULTIPLIER * row["vol_avg"]
            cond5 = row["ret_6m"]  > MIN_6M_RETURN                          # NEW
            cond6 = (not USE_NIFTY_REGIME) or _nifty_in_stage2(df.index[i]) # NEW

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                entry_price = float(nxt["Open"]) * (1 + SLIPPAGE_PCT)
                entry_idx   = i + 1
                peak_price  = entry_price
                in_pos      = True

        else:
            # Update peak (uses high for trailing-stop tracking)
            high_today  = float(row["High"])
            if high_today > peak_price:
                peak_price = high_today

            # Decide whether to exit this bar
            close      = float(row["Close"])
            low_today  = float(row["Low"])
            unrealized = (close - entry_price) / entry_price
            peak_gain  = (peak_price - entry_price) / entry_price
            trail_lvl  = peak_price * (1 - TRAIL_DRAWDOWN_PCT)
            hard_lvl   = entry_price * (1 + HARD_STOP_PCT)

            outcome   = None
            exit_raw  = None

            # 1. Hard stop — intrabar low touches -7% line
            if low_today <= hard_lvl:
                outcome  = "HARD_STOP"
                exit_raw = hard_lvl

            # 2. Trailing stop — only active after +5% peak gain
            elif peak_gain >= TRAIL_ACTIVATE_PCT and low_today <= trail_lvl:
                outcome  = "TRAIL_STOP"
                exit_raw = trail_lvl

            # 3. SMA breakdown — original Stage-2 exit
            elif close < row["SMA"]:
                outcome  = "SMA_EXIT"
                exit_raw = float(nxt["Open"])

            if outcome is not None:
                # Hard/trail stops fill intrabar at stop level; SMA exits fill at next open
                if outcome == "SMA_EXIT":
                    exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                    exit_date_label = str(df.index[i + 1].date())
                    days_held = i + 1 - entry_idx
                else:
                    exit_price = exit_raw * (1 - SLIPPAGE_PCT)
                    exit_date_label = str(df.index[i].date())
                    days_held = i - entry_idx

                gross = (exit_price - entry_price) / entry_price
                net   = gross - TRANSACTION_COST_PCT
                trades.append({
                    "symbol":     symbol,
                    "entry_date": str(df.index[entry_idx].date()),
                    "exit_date":  exit_date_label,
                    "entry":      round(entry_price, 2),
                    "exit":       round(exit_price, 2),
                    "peak":       round(peak_price, 2),
                    "days_held":  days_held,
                    "gross_pct":  round(gross * 100, 2),
                    "net_pct":    round(net * 100, 2),
                    "pnl_rs":     round(net * TRADE_NOTIONAL, 2),
                    "outcome":    outcome,
                })
                in_pos = False
                entry_price = entry_idx = peak_price = None

    # Mark-to-market any open position at end
    if in_pos:
        last = df.iloc[-1]
        exit_price = float(last["Close"]) * (1 - SLIPPAGE_PCT)
        gross = (exit_price - entry_price) / entry_price
        net   = gross - TRANSACTION_COST_PCT
        trades.append({
            "symbol":     symbol,
            "entry_date": str(df.index[entry_idx].date()),
            "exit_date":  str(df.index[-1].date()),
            "entry":      round(entry_price, 2),
            "exit":       round(exit_price, 2),
            "peak":       round(peak_price, 2),
            "days_held":  len(df) - 1 - entry_idx,
            "gross_pct":  round(gross * 100, 2),
            "net_pct":    round(net * 100, 2),
            "pnl_rs":     round(net * TRADE_NOTIONAL, 2),
            "outcome":    "OPEN_EOD",
        })

    return trades


# ── Data fetch (Upstox first, Yahoo fallback) ──────────────────────────────

def fetch_2y_daily(symbol: str) -> pd.DataFrame:
    """Try Upstox first (no rate limit), then Yahoo."""
    try:
        from app.services.upstox_data import upstox_service
        if upstox_service.is_available():
            df = upstox_service.get_ohlcv(symbol, period="2y", interval="1d")
            if df is not None and not df.empty and len(df) >= SMA_PERIOD + 60:
                return df
    except Exception as e:
        logger.debug(f"[stage2] Upstox failed for {symbol}: {e}")

    try:
        from app.services.yahoo_finance import yahoo_service
        df = yahoo_service.get_ohlcv(f"{symbol}.NS", period="2y", interval="1d")
        if df is not None and not df.empty and len(df) >= SMA_PERIOD + 60:
            return df
    except Exception as e:
        logger.debug(f"[stage2] Yahoo failed for {symbol}: {e}")

    return None


# ── Main runner ────────────────────────────────────────────────────────────

def run(top_n_print: int = 10, max_workers: int = 4):
    from app.services.intraday_universe import INTRADAY_UNIVERSE

    print(f"\n🚀 ASTRA Stage-2 Hypothesis Test (iteration 3 — selective entry + wider stop)")
    print(f"   Universe: {len(INTRADAY_UNIVERSE)} symbols  |  Window: 2y  |  Costs: 0.25% round-trip")
    print(f"   Filters : 6m-return > {MIN_6M_RETURN*100:.0f}%  |  NIFTY regime: {'ON' if USE_NIFTY_REGIME else 'OFF'}")
    print(f"   Stops   : hard {HARD_STOP_PCT*100:+.0f}%  |  trail {TRAIL_DRAWDOWN_PCT*100:.0f}% after +{TRAIL_ACTIVATE_PCT*100:.0f}%  |  SMA-break\n")

    # Load NIFTY once (single fetch, reused across all 86 symbols)
    if USE_NIFTY_REGIME:
        nifty = _load_nifty_regime()
        if nifty is None:
            print("⚠  NIFTY regime data unavailable — proceeding with filter disabled (fail-open).")
        else:
            in_s2 = (nifty["Close"] > nifty["SMA"]).sum()
            total = nifty["SMA"].notna().sum()
            print(f"   NIFTY regime: {in_s2}/{total} bars in Stage 2 ({in_s2/total*100:.0f}%)\n")

    all_trades   = []
    fetched      = 0
    failed       = []
    no_signal    = []
    start        = time.time()

    def _job(sym):
        df = fetch_2y_daily(sym)
        if df is None:
            return sym, None, "fetch_failed"
        trades = backtest_stage2(sym, df)
        return sym, trades, f"len(df)={len(df)}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_job, s): s for s in INTRADAY_UNIVERSE}
        for fut in as_completed(futures):
            sym, trades, msg = fut.result()
            if trades is None:
                failed.append(sym)
                continue
            fetched += 1
            if not trades:
                no_signal.append(sym)
            else:
                all_trades.extend(trades)

    duration = round(time.time() - start, 1)
    print(f"⏱  Done in {duration}s.  Fetched {fetched}/{len(INTRADAY_UNIVERSE)}.  "
          f"Failed: {len(failed)}.  No signal: {len(no_signal)}.\n")

    if not all_trades:
        print("❌ No trades generated. Either Stage-2 criteria too strict or data is missing.")
        if failed:
            print(f"   First few fetch failures: {failed[:5]}")
        return

    df_t = pd.DataFrame(all_trades)
    wins   = df_t[df_t["net_pct"] > 0]
    losses = df_t[df_t["net_pct"] <= 0]

    wr           = len(wins) / len(df_t) * 100
    avg_win      = wins["net_pct"].mean()   if len(wins)   else 0.0
    avg_loss     = losses["net_pct"].mean() if len(losses) else 0.0
    rr           = abs(avg_win / avg_loss)  if avg_loss < 0 else float("inf")
    expectancy   = wr / 100 * avg_win + (1 - wr / 100) * avg_loss
    total_pnl    = df_t["pnl_rs"].sum()
    avg_held     = df_t["days_held"].mean()
    median_held  = df_t["days_held"].median()
    max_win      = df_t["net_pct"].max()
    max_loss     = df_t["net_pct"].min()

    # Profit factor
    gross_profit = wins["pnl_rs"].sum()
    gross_loss   = abs(losses["pnl_rs"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualised return rate per trade (rough)
    # Each trade locks ₹1L for `days_held`. CAGR per trade ≈ (1 + net_pct/100) ^ (365/days_held) - 1
    df_t["cagr_pct"] = (
        np.where(df_t["days_held"] > 0,
                 ((1 + df_t["net_pct"] / 100) ** (365 / df_t["days_held"]) - 1) * 100,
                 0.0)
    )
    median_cagr = df_t["cagr_pct"].median()

    print("=" * 78)
    print("  ASTRA STAGE-2 TREND-FOLLOWING  —  HYPOTHESIS TEST RESULTS")
    print("=" * 78)
    print(f"  Trades                : {len(df_t)}")
    print(f"  Symbols with trades   : {df_t['symbol'].nunique()} / {len(INTRADAY_UNIVERSE)}")
    print(f"  Win rate              : {wr:.1f}%")
    print(f"  Avg win               : +{avg_win:.2f}%")
    print(f"  Avg loss              : {avg_loss:.2f}%")
    print(f"  Max win               : +{max_win:.2f}%")
    print(f"  Max loss              : {max_loss:.2f}%")
    print(f"  R:R ratio             : {rr:.2f}x")
    print(f"  Expectancy / trade    : {expectancy:+.2f}%")
    print(f"  Profit factor         : {pf:.2f}")
    print(f"  Total P&L (₹1L size)  : ₹{total_pnl:+,.0f}")
    print(f"  Avg / median days held: {avg_held:.0f} / {median_held:.0f}")
    print(f"  Median trade CAGR     : {median_cagr:+.1f}%")
    # NEW: outcome-mix breakdown — shows which exit rule fired and what its P&L was
    print(f"  Exit-rule breakdown   :")
    for outcome, sub in df_t.groupby("outcome"):
        n     = len(sub)
        sub_wr = (sub["net_pct"] > 0).mean() * 100
        sub_avg = sub["net_pct"].mean()
        sub_pnl = sub["pnl_rs"].sum()
        print(f"    {outcome:<12} n={n:>3}  WR={sub_wr:>5.1f}%  avg {sub_avg:>+6.2f}%  net ₹{sub_pnl:>+8,.0f}")
    print()

    # Hypothesis verdict
    print("  📊 HYPOTHESIS VERDICT")
    print("  " + "─" * 70)
    if expectancy > 0.5 and pf > 1.3 and rr > 2:
        verdict = "✅ STRONG — build the full Stage-2 module"
    elif expectancy > 0 and pf > 1.0:
        verdict = "⚠  MARGINAL — has edge but tune entry criteria first"
    else:
        verdict = "❌ NEGATIVE — Stage-2 doesn't work as-is on this universe / period"
    print(f"  {verdict}")
    print(f"  Decision rule: strong = expectancy > 0.5% AND PF > 1.3 AND R:R > 2")
    print()

    print(f"  TOP {top_n_print} PROFITABLE TRADES")
    for _, t in df_t.nlargest(top_n_print, "pnl_rs").iterrows():
        print(f"    {t['symbol']:<12} {t['entry_date']} → {t['exit_date']}  "
              f"{int(t['days_held']):>4}d  net {t['net_pct']:>+7.2f}%  ₹{t['pnl_rs']:>+9,.0f}")
    print()
    print(f"  BOTTOM {top_n_print} LOSING TRADES")
    for _, t in df_t.nsmallest(top_n_print, "pnl_rs").iterrows():
        print(f"    {t['symbol']:<12} {t['entry_date']} → {t['exit_date']}  "
              f"{int(t['days_held']):>4}d  net {t['net_pct']:>+7.2f}%  ₹{t['pnl_rs']:>+9,.0f}")

    # Per-symbol breakdown
    by_sym = df_t.groupby("symbol").agg(
        n=("net_pct", "count"),
        wr=("net_pct", lambda x: (x > 0).mean() * 100),
        net_pnl=("pnl_rs", "sum"),
        avg_pct=("net_pct", "mean"),
        avg_days=("days_held", "mean"),
    ).sort_values("net_pnl", ascending=False)

    print()
    print(f"  PER-SYMBOL — top {top_n_print} most profitable")
    for sym, row in by_sym.head(top_n_print).iterrows():
        print(f"    {sym:<12} n={int(row['n']):>2}  WR={row['wr']:>5.1f}%  "
              f"net ₹{row['net_pnl']:>+9,.0f}  avg {row['avg_pct']:>+6.2f}%  "
              f"hold {row['avg_days']:>4.0f}d")

    print(f"\n  PER-SYMBOL — bottom {top_n_print} most losing")
    for sym, row in by_sym.tail(top_n_print).iterrows():
        print(f"    {sym:<12} n={int(row['n']):>2}  WR={row['wr']:>5.1f}%  "
              f"net ₹{row['net_pnl']:>+9,.0f}  avg {row['avg_pct']:>+6.2f}%  "
              f"hold {row['avg_days']:>4.0f}d")

    # Save full trade log
    out_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stage2_trades_{int(time.time())}.csv")
    df_t.to_csv(out_path, index=False)
    print(f"\n  💾 Full trade log saved → {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",     type=int, default=10, help="Show top/bottom N in summary")
    ap.add_argument("--workers", type=int, default=4,  help="Parallel data fetchers (keep ≤6)")
    args = ap.parse_args()
    run(top_n_print=args.top, max_workers=args.workers)
