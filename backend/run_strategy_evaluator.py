"""
CLI: Evaluate ASTRA engines via the unified evaluator.
Uses the strategy registry — every Strategy subclass in app/strategies/ is auto-listed.

Usage:
    cd backend && source .venv/bin/activate
    python run_strategy_evaluator.py              # all registered engines
    python run_strategy_evaluator.py --only ASTRA.PULLBACK
    python run_strategy_evaluator.py --list       # just list available engines
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from app.services.strategy_evaluator import evaluate, print_report
from app.strategies.registry import list_strategies, get_strategy, get_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",    type=str, default=None, help="run only this engine by name")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list",    action="store_true", help="just list registered engines")
    args = ap.parse_args()

    if args.list:
        print("\n📋 Registered strategies:\n")
        for s in list_strategies():
            print(f"  {s['name']:<22} v{s['version']:<6} ({s['timeframe']:<5}) — {s['description'][:60]}")
        print()
        return

    engines = []
    if args.only is not None:
        eng = get_strategy(args.only)
        if eng is None:
            print(f"❌ Strategy '{args.only}' not registered. Use --list to see options.")
            return
        engines = [eng]
    else:
        engines = get_all()

    print(f"\n🔬 ASTRA Strategy Evaluator — {len(engines)} engine(s) via registry")

    reports = []
    for eng in engines:
        report = evaluate(eng, max_workers=args.workers)
        print_report(report)
        reports.append(report)

    if len(reports) > 1:
        print("=" * 78)
        print("  HEAD-TO-HEAD SUMMARY")
        print("=" * 78)
        print(f"  {'Engine':<25} {'Verdict':<10} {'Trades':>7} {'WR':>6} {'R:R':>6} {'Exp':>7} {'PF':>5}  Regime")
        for r in reports:
            s = r.base_stats
            print(f"  {r.strategy_name:<25} {r.verdict:<10} {s.n_trades:>7} "
                  f"{s.win_rate:>5.1f}% {s.rr_ratio:>5.2f} {s.expectancy:>+6.2f}% {s.profit_factor:>5.2f}  "
                  f"{r.regime_pass_rate*100:.0f}%")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
