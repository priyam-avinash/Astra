"""
ASTRA Backtest Engine v1.0
===========================
Vectorized walk-forward backtest for all 3 ASTRA engines + Ensemble.

Methodology:
- Out-of-sample only: last 12 months of data (not seen during training)
- Simulates bar-by-bar signal generation with ATR-based SL/TP
- One trade at a time (no pyramiding)
- Brokerage: 0.03% round-trip (Dhan intraday rate)
- Slippage: 0.05% per leg

Metrics reported per engine:
  Total Return%, CAGR%, Win Rate%, Avg Win%, Avg Loss%,
  Risk-Reward Ratio, Expectancy%, Max Drawdown%, Sharpe Ratio,
  Total Trades, Profitable Trades
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Constants ──────────────────────────────────────────────────────────────────
BROKERAGE    = 0.0003   # 0.03% round-trip
SLIPPAGE     = 0.0005   # 0.05% per leg
INITIAL_CAP  = 100_000  # ₹1,00,000 starting capital
POSITION_PCT = 0.10     # 10% of capital per trade (fixed for fair comparison)
SL_MULT_MAP  = {"astra": 1.5, "astra_ai": 1.6, "astra_ml": 1.5, "ensemble": 1.6}
TP_MULT_MAP  = {"astra": 3.0, "astra_ai": 4.0, "astra_ml": 3.5, "ensemble": 4.0}

SYMBOLS = [
    "^NSEI",       # NIFTY 50
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]

CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD"]


# ── Import engines ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from app.services.ai_predictor import ai_engine, FEATURE_COLS
from app.services.crypto_engine import crypto_engine, CRYPTO_FEATURE_COLS


# ── Backtest Core ──────────────────────────────────────────────────────────────

def run_backtest(symbol: str, engine: str, lookback_bars: int = 252,
                 is_crypto: bool = False) -> dict:
    """
    Walk-forward backtest for a single symbol + engine combination.
    Uses last `lookback_bars` bars as the out-of-sample test window.
    Returns trade list + summary metrics.
    """
    # ── Fetch data (2y to ensure enough history for SMA-200 + test window)
    try:
        if is_crypto:
            df_raw = crypto_engine.fetch_data(symbol, market="international",
                                               timeframe="1d", limit=800)
            df = crypto_engine._compute_features(df_raw)
            feat_cols = CRYPTO_FEATURE_COLS
            sl_mult = 2.5
            tp_mult = 5.0
        else:
            df_raw = ai_engine._fetch_data(symbol, period="2y", interval="1d")
            df = ai_engine._compute_features(df_raw)
            feat_cols = FEATURE_COLS
            sl_mult = SL_MULT_MAP.get(engine, 1.5)
            tp_mult = TP_MULT_MAP.get(engine, 3.0)

        if df.empty or len(df) < 60:
            return {"symbol": symbol, "engine": engine, "error": "Insufficient data"}
    except Exception as e:
        return {"symbol": symbol, "engine": engine, "error": str(e)}

    # ── Out-of-sample window: last 12 months
    test_df = df.tail(lookback_bars).copy().reset_index(drop=False)
    if len(test_df) < 30:
        return {"symbol": symbol, "engine": engine, "error": "Test window too small"}

    # ── Simulate trades ────────────────────────────────────────────────────────
    trades   = []
    capital  = INITIAL_CAP
    in_trade = False
    entry_price = sl = tp = direction = None
    entry_bar   = 0

    for i in range(30, len(test_df)):           # 30-bar warmup for indicators
        row  = test_df.iloc[i]
        prev = test_df.iloc[i - 1]

        high   = float(row["High"])
        low    = float(row["Low"])
        close  = float(row["Close"])
        open_  = float(row["Open"])
        atr    = float(prev.get("ATR", close * 0.015))

        # ── Check exit first (uses current bar's H/L)
        if in_trade:
            hit_tp = (direction == "BUY" and high >= tp) or \
                     (direction == "SELL" and low <= tp)
            hit_sl = (direction == "BUY" and low <= sl) or \
                     (direction == "SELL" and high >= sl)

            if hit_tp or hit_sl:
                exit_price = tp if hit_tp else sl
                # Slippage on exit
                exit_price *= (1 - SLIPPAGE) if direction == "BUY" else (1 + SLIPPAGE)
                raw_pnl_pct = ((exit_price - entry_price) / entry_price) * (1 if direction == "BUY" else -1)
                net_pnl_pct = raw_pnl_pct - BROKERAGE
                position_size = capital * POSITION_PCT
                pnl_cash = position_size * net_pnl_pct
                capital += pnl_cash
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar":  i,
                    "bars_held": i - entry_bar,
                    "direction": direction,
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price, 4),
                    "exit_reason": "TP" if hit_tp else "SL",
                    "pnl_pct":  round(net_pnl_pct * 100, 3),
                    "pnl_cash": round(pnl_cash, 2),
                    "capital":  round(capital, 2),
                })
                in_trade = False
            continue           # Skip signal generation while in trade

        # ── Generate signal using past rows (no look-ahead)
        history = test_df.iloc[:i]

        try:
            if engine == "astra":
                last = history.iloc[-1]
                patterns = ai_engine._detect_candlestick_patterns(history)
                buy_ok, buy_conf, _ = ai_engine._check_buy_confirmations(last, patterns, True)
                sell_ok, sell_conf, _ = ai_engine._check_sell_confirmations(last, patterns)
                if buy_ok and buy_conf > sell_conf:
                    sig = "BUY"
                elif sell_ok and sell_conf > buy_conf:
                    sig = "SELL"
                else:
                    sig = "HOLD"

            elif engine == "astra_ai":
                if ai_engine.rf_model is None:
                    sig = "HOLD"
                else:
                    feats = history[feat_cols].tail(1)
                    if feats.isnull().any().any():
                        sig = "HOLD"
                    else:
                        pred = float(ai_engine.rf_model.predict(feats)[0])
                        vol  = float(history.iloc[-1].get("Volat_Ratio", 1.0))
                        thr  = max(1.2, vol * 0.5)
                        sig  = "BUY" if pred > thr else ("SELL" if pred < -thr else "HOLD")

            elif engine == "astra_ml":
                if ai_engine.lstm_model is None:
                    sig = "HOLD"
                else:
                    feat_data = history[feat_cols].tail(30).values
                    if len(feat_data) < 30 or np.isnan(feat_data).any():
                        sig = "HOLD"
                    else:
                        scaled = ai_engine.lstm_scaler.transform(feat_data)
                        X = scaled.reshape(1, 30, len(feat_cols))
                        pred = float(ai_engine.lstm_model.predict(X, verbose=0)[0][0])
                        vol  = float(history.iloc[-1].get("Volat_Ratio", 1.0))
                        thr  = max(0.8, vol * 0.4)
                        sig  = "BUY" if pred > thr else ("SELL" if pred < -thr else "HOLD")

            elif engine == "ensemble":
                # Quick ensemble: RF + rules (skip LSTM for speed)
                sigs = []
                last = history.iloc[-1]
                patterns = ai_engine._detect_candlestick_patterns(history)
                buy_ok, buy_conf, _ = ai_engine._check_buy_confirmations(last, patterns, True)
                sell_ok, sell_conf, _ = ai_engine._check_sell_confirmations(last, patterns)
                sigs.append("BUY" if (buy_ok and buy_conf > sell_conf) else
                            ("SELL" if (sell_ok and sell_conf > buy_conf) else "HOLD"))
                if ai_engine.rf_model is not None:
                    feats = history[feat_cols].tail(1)
                    if not feats.isnull().any().any():
                        pred = float(ai_engine.rf_model.predict(feats)[0])
                        vol  = float(last.get("Volat_Ratio", 1.0))
                        thr  = max(1.2, vol * 0.5)
                        sigs.append("BUY" if pred > thr else ("SELL" if pred < -thr else "HOLD"))
                from collections import Counter
                top_sig, top_votes = Counter(sigs).most_common(1)[0]
                sig = top_sig if top_votes >= 2 else "HOLD"

            elif engine == "crypto_rules":
                last = history.iloc[-1]
                adx = float(last.get("ADX", 15))
                rsi = float(last.get("RSI", 50))
                bb  = float(last.get("BB_PctB", 0.5))
                macd = float(last.get("MACD", 0))
                macd_sig = float(last.get("MACD_Signal", 0))
                if adx >= 25:   # Trending: momentum
                    sig = "BUY" if (macd > macd_sig and rsi < 65) else \
                          ("SELL" if (macd < macd_sig and rsi > 35) else "HOLD")
                else:           # Ranging: mean reversion
                    sig = "BUY" if bb < 0.15 else ("SELL" if bb > 0.85 else "HOLD")
            else:
                sig = "HOLD"

        except Exception:
            sig = "HOLD"

        # ── Enter trade on next open (realistic execution)
        if sig in ("BUY", "SELL"):
            entry_price = open_ * (1 + SLIPPAGE if sig == "BUY" else 1 - SLIPPAGE)
            sl = entry_price - atr * sl_mult if sig == "BUY" else entry_price + atr * sl_mult
            tp = entry_price + atr * tp_mult if sig == "BUY" else entry_price - atr * tp_mult
            direction  = sig
            entry_bar  = i
            in_trade   = True

    # ── Close any open trade at last bar
    if in_trade and len(test_df) > 0:
        last_close = float(test_df.iloc[-1]["Close"])
        exit_price = last_close * (1 - SLIPPAGE if direction == "BUY" else 1 + SLIPPAGE)
        raw_pnl    = ((exit_price - entry_price) / entry_price) * (1 if direction == "BUY" else -1)
        net_pnl    = raw_pnl - BROKERAGE
        pnl_cash   = capital * POSITION_PCT * net_pnl
        capital   += pnl_cash
        trades.append({
            "direction": direction, "exit_reason": "EOD",
            "pnl_pct": round(net_pnl * 100, 3),
            "pnl_cash": round(pnl_cash, 2), "capital": round(capital, 2),
        })

    # ── Compute metrics ────────────────────────────────────────────────────────
    if not trades:
        return {"symbol": symbol, "engine": engine,
                "total_trades": 0, "win_rate": 0.0, "total_return_pct": 0.0}

    df_t = pd.DataFrame(trades)
    wins   = df_t[df_t["pnl_pct"] > 0]
    losses = df_t[df_t["pnl_pct"] <= 0]

    total_return  = (capital - INITIAL_CAP) / INITIAL_CAP * 100
    win_rate      = len(wins) / len(df_t) * 100
    avg_win       = wins["pnl_pct"].mean()   if len(wins)   else 0.0
    avg_loss      = losses["pnl_pct"].mean() if len(losses) else 0.0
    rr_ratio      = abs(avg_win / avg_loss)  if avg_loss else float("inf")
    expectancy    = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Drawdown
    cap_series = [INITIAL_CAP] + list(df_t["capital"])
    peak = pd.Series(cap_series).cummax()
    dd   = ((pd.Series(cap_series) - peak) / peak * 100)
    max_dd = dd.min()

    # Sharpe (daily returns approximation)
    pnl_series = df_t["pnl_pct"].values / 100
    sharpe = (pnl_series.mean() / (pnl_series.std() + 1e-9)) * np.sqrt(252) if len(pnl_series) > 1 else 0.0

    return {
        "symbol":          symbol,
        "engine":          engine,
        "total_trades":    len(df_t),
        "win_trades":      len(wins),
        "loss_trades":     len(losses),
        "win_rate":        round(win_rate, 1),
        "avg_win_pct":     round(avg_win, 3),
        "avg_loss_pct":    round(avg_loss, 3),
        "rr_ratio":        round(rr_ratio, 2),
        "expectancy_pct":  round(expectancy, 3),
        "total_return_pct":round(total_return, 2),
        "max_drawdown_pct":round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 3),
        "final_capital":   round(capital, 2),
        "trades":          df_t.to_dict("records"),
    }


def print_summary(results: list):
    """Pretty-print aggregate results per engine."""
    by_engine = defaultdict(list)
    for r in results:
        if "error" not in r and r.get("total_trades", 0) > 0:
            by_engine[r["engine"]].append(r)

    print("\n" + "=" * 80)
    print("  ASTRA BACKTEST RESULTS  —  Out-of-sample: last 12 months")
    print("=" * 80)

    engine_order = ["astra", "astra_ai", "astra_ml", "ensemble", "crypto_rules"]
    for eng in engine_order:
        group = by_engine.get(eng)
        if not group:
            continue
        trades_total  = sum(g["total_trades"]    for g in group)
        wins_total    = sum(g["win_trades"]       for g in group)
        returns       = [g["total_return_pct"]   for g in group]
        win_rates     = [g["win_rate"]            for g in group]
        avg_wins      = [g["avg_win_pct"]         for g in group if g["avg_win_pct"]]
        avg_losses    = [g["avg_loss_pct"]        for g in group if g["avg_loss_pct"]]
        rr_ratios     = [g["rr_ratio"]            for g in group if g["rr_ratio"] != float("inf")]
        expect        = [g["expectancy_pct"]      for g in group]
        drawdowns     = [g["max_drawdown_pct"]    for g in group]
        sharpes       = [g["sharpe_ratio"]        for g in group]

        label = {
            "astra":        "ASTRA 1.0  (Rule-Based)",
            "astra_ai":     "ASTRA.AI   (Random Forest)",
            "astra_ml":     "ASTRA.ML   (BiLSTM Equity)",
            "ensemble":     "ENSEMBLE   (Rules + RF, 2/3 vote)",
            "crypto_rules": "CRYPTO     (Regime Rules)",
        }.get(eng, eng)

        overall_wr = wins_total / trades_total * 100 if trades_total else 0

        print(f"\n  ┌─ {label}")
        print(f"  │  Symbols tested   : {len(group)}")
        print(f"  │  Total trades     : {trades_total}")
        print(f"  │  Win Rate         : {overall_wr:.1f}%  (target >50%)")
        print(f"  │  Avg Win          : {np.mean(avg_wins) if avg_wins else 0:.2f}%")
        print(f"  │  Avg Loss         : {np.mean(avg_losses) if avg_losses else 0:.2f}%")
        print(f"  │  Risk/Reward      : {np.mean(rr_ratios) if rr_ratios else 0:.2f}x")
        print(f"  │  Expectancy/trade : {np.mean(expect):.3f}%")
        print(f"  │  Avg Return       : {np.mean(returns):.2f}%  | Best: {max(returns):.2f}%  | Worst: {min(returns):.2f}%")
        print(f"  │  Max Drawdown     : {np.mean(drawdowns):.2f}%")
        print(f"  └─ Sharpe Ratio     : {np.mean(sharpes):.3f}")

    # Per-symbol breakdown
    print("\n" + "─" * 80)
    print("  PER-SYMBOL BREAKDOWN")
    print("─" * 80)
    header = f"  {'Symbol':<16} {'Engine':<12} {'Trades':>6} {'WinRate':>8} {'AvgWin':>7} {'AvgLoss':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}"
    print(header)
    print("  " + "-" * 78)
    for r in sorted(results, key=lambda x: (x.get("engine",""), x.get("symbol",""))):
        if "error" in r or r.get("total_trades", 0) == 0:
            print(f"  {r['symbol']:<16} {r['engine']:<12}   —  no trades / {r.get('error','')[:30]}")
            continue
        print(f"  {r['symbol']:<16} {r['engine']:<12} {r['total_trades']:>6} "
              f"{r['win_rate']:>7.1f}% {r['avg_win_pct']:>7.2f}% {r['avg_loss_pct']:>7.2f}% "
              f"{r['total_return_pct']:>7.2f}% {r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>6.2f}%")

    print("=" * 80)
    print(f"\n  Note: ₹{INITIAL_CAP:,} initial capital | {POSITION_PCT*100:.0f}% per trade | "
          f"{BROKERAGE*100:.3f}% brokerage + {SLIPPAGE*100:.2f}% slippage included")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 ASTRA Backtest Engine — Starting at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Symbols  : {len(SYMBOLS)} equity + {len(CRYPTO_SYMBOLS)} crypto")
    print(f"   Engines  : ASTRA 1.0, ASTRA.AI (RF), ASTRA.ML (LSTM), Ensemble, Crypto Rules")
    print(f"   Window   : Last 12 months (out-of-sample)")
    print(f"   Capital  : ₹{INITIAL_CAP:,} | {POSITION_PCT*100:.0f}% per trade\n")

    all_results = []
    EQUITY_ENGINES = ["astra", "astra_ai", "astra_ml", "ensemble"]

    # ── Equity backtests
    for symbol in SYMBOLS:
        for eng in EQUITY_ENGINES:
            print(f"  ▶ {symbol:<16} [{eng}] ...", end=" ", flush=True)
            r = run_backtest(symbol, eng, lookback_bars=252, is_crypto=False)
            all_results.append(r)
            if "error" in r:
                print(f"❌ {r['error'][:40]}")
            else:
                print(f"✅  {r['total_trades']} trades | WR={r['win_rate']:.1f}% | R={r['total_return_pct']:+.2f}%")

    # ── Crypto backtests
    for symbol in CRYPTO_SYMBOLS:
        print(f"  ▶ {symbol:<16} [crypto_rules] ...", end=" ", flush=True)
        r = run_backtest(symbol, "crypto_rules", lookback_bars=252, is_crypto=True)
        all_results.append(r)
        if "error" in r:
            print(f"❌ {r['error'][:40]}")
        else:
            print(f"✅  {r['total_trades']} trades | WR={r['win_rate']:.1f}% | R={r['total_return_pct']:+.2f}%")

    # ── Save detailed results
    results_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Save trade logs per engine
    for r in all_results:
        if "trades" in r and r["trades"]:
            fname = f"{results_dir}/{r['symbol'].replace('^','').replace('.','_')}_{r['engine']}_{ts}.csv"
            pd.DataFrame(r["trades"]).to_csv(fname, index=False)

    # Save summary
    summary_rows = [{k: v for k, v in r.items() if k != "trades"} for r in all_results]
    summary_path = f"{results_dir}/summary_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  💾 Detailed trade logs saved → {results_dir}/")

    print_summary(all_results)
    print(f"\n  ⏱  Completed at {datetime.now().strftime('%H:%M:%S')}\n")
