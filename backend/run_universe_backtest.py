"""
CLI: Universe-driven intraday backtest.

Usage:
    python run_universe_backtest.py                       # 30d, top-15, weights ON
    python run_universe_backtest.py --no-weights          # baseline (no Layer-1)
    python run_universe_backtest.py --top 10 --days 60
    python run_universe_backtest.py --reset-weights       # clear Layer-1 state first
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.services.intraday_backtest import run_universe_backtest
from app.services.signal_weights import signal_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",          type=int, default=30)
    ap.add_argument("--top",           type=int, default=15)
    ap.add_argument("--no-weights",    action="store_true")
    ap.add_argument("--reset-weights", action="store_true")
    args = ap.parse_args()

    if args.reset_weights:
        n = signal_weights.reset()
        print(f"Reset {n} weight entries.\n")

    result = run_universe_backtest(
        days=args.days,
        top_n_per_strategy=args.top,
        use_weights=not args.no_weights,
    )

    print("\n" + "=" * 78)
    print(f"  ASTRA UNIVERSE BACKTEST  —  {args.days} days, top-{args.top} per strategy")
    print(f"  Layer-1 weights: {'ON ' if not args.no_weights else 'OFF'}     "
          f"Skipped by weight: {result['skipped_by_weight']}")
    print("=" * 78)

    print("\n  AGGREGATE PER STRATEGY")
    print("  " + "-" * 70)
    print(f"  {'Strategy':<12} {'Symbols':>8} {'Trades':>8} {'WR%':>7} {'TotalP&L':>12} {'AvgP&L':>10}")
    for strat, s in result["aggregate"].items():
        st  = s.get("symbols_traded", 0)
        tr  = s.get("trades", 0)
        wr  = s.get("win_rate_pct", 0)
        pnl = s.get("total_pnl_rs", 0)
        ap_ = s.get("avg_pnl_rs", 0)
        print(f"  {strat:<12} {st:>8} {tr:>8} {wr:>7.1f} {pnl:>11.0f} {ap_:>9.0f}")

    print("\n  UNIVERSE USED (top per strategy)")
    for strat, syms in result["universe_used"].items():
        print(f"  {strat:<12}: {', '.join(syms[:12])}{' ...' if len(syms) > 12 else ''}")

    # Best/worst per (strategy, symbol)
    flat = []
    for sym, by_strat in result["per_symbol"].items():
        for strat, s in by_strat.items():
            flat.append((s["pnl_rs"], sym, strat, s["trades"], s["win_rate"]))
    flat.sort(reverse=True)
    print("\n  TOP 5 PROFITABLE (strategy, symbol)")
    for pnl, sym, strat, tr, wr in flat[:5]:
        print(f"   +Rs{pnl:>7.0f}  {strat:<10} {sym:<12}  ({tr} trades, WR {wr}%)")
    print("\n  BOTTOM 5 LOSING")
    for pnl, sym, strat, tr, wr in flat[-5:]:
        print(f"    Rs{pnl:>7.0f}  {strat:<10} {sym:<12}  ({tr} trades, WR {wr}%)")

    if not args.no_weights:
        stats = signal_weights.stats()
        if stats:
            print(f"\n  LAYER-1 WEIGHTS  —  {len(stats)} (engine, symbol) keys")
            top_w = sorted(stats.values(), key=lambda x: x["weight"], reverse=True)[:5]
            bot_w = sorted(stats.values(), key=lambda x: x["weight"])[:5]
            print("    Top-weighted (most favoured):")
            for s in top_w:
                print(f"     {s['engine']:<10} {s['symbol']:<12}  n={s['n']:>2}  WR={s['wr']}  w={s['weight']}")
            print("    Bottom-weighted (suppressed):")
            for s in bot_w:
                print(f"     {s['engine']:<10} {s['symbol']:<12}  n={s['n']:>2}  WR={s['wr']}  w={s['weight']}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
