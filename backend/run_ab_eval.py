"""
A/B Evaluation: baseline strategies vs. enhancement-filter strategies.

Usage:
    python run_ab_eval.py          # runs both runs, prints side-by-side comparison
    python run_ab_eval.py baseline # baseline only
    python run_ab_eval.py enhanced # enhanced only
"""
import os
import sys
import logging

logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("UPSTOX_ACCESS_TOKEN", "")
sys.path.insert(0, os.path.dirname(__file__))

import app.strategies.filters as _f
from app.strategies.pullback  import PullbackStrategy
from app.strategies.momentum  import MomentumStrategy
from app.strategies.stage2    import Stage2Strategy
from app.strategies.quality   import QualityStrategy
from app.services.strategy_evaluator import evaluate, print_report, EvaluationReport, clear_universe_cache

_STRATEGIES = [PullbackStrategy, MomentumStrategy, Stage2Strategy, QualityStrategy]
_WORKERS    = 8


def _run(label: str, use_hv: bool, use_vix: bool, use_vwap: bool) -> list[EvaluationReport]:
    # Toggle filters globally
    _f.USE_HV_RANK_FILTER = use_hv
    _f.USE_VIX_FILTER     = use_vix
    _f.USE_VWAP_FILTER    = use_vwap
    _f.clear_vix_cache()   # fresh VIX fetch per run

    print(f"\n{'='*78}")
    print(f"  RUN: {label}  (HV_rank={use_hv}, VIX_gate={use_vix}, VWAP={use_vwap})")
    print(f"{'='*78}")

    reports = []
    for cls in _STRATEGIES:
        r = evaluate(cls(), max_workers=_WORKERS)
        print_report(r)
        reports.append(r)
    return reports


def _compare(baseline: list[EvaluationReport], enhanced: list[EvaluationReport]) -> None:
    print("\n" + "=" * 78)
    print("  A/B COMPARISON — Baseline vs Enhanced Filters")
    print("=" * 78)
    print(f"  {'Strategy':<22} {'Trades':>6}  {'WR%':>6}  {'Exp%':>6}  {'PF':>5}  "
          f"{'P&L ₹':>10}  {'Δ Trades':>8}  {'Δ WR':>7}  {'Δ Exp':>7}  {'Decision':>10}")
    print("  " + "-" * 96)

    gate_passed = 0
    for b, e in zip(baseline, enhanced):
        bs, es = b.base_stats, e.base_stats
        d_trades = es.n_trades - bs.n_trades
        d_wr     = es.win_rate - bs.win_rate
        d_exp    = es.expectancy - bs.expectancy
        d_pnl    = es.total_pnl_rs - bs.total_pnl_rs
        pnl_pct  = (d_pnl / abs(bs.total_pnl_rs) * 100) if bs.total_pnl_rs else 0

        # Gate: WR +5pp, expectancy +0.10%, trades >= 30, P&L +15%
        criteria = [
            d_wr   >= 5.0,
            d_exp  >= 0.10,
            es.n_trades >= 30,
            pnl_pct >= 15.0,
        ]
        passed = sum(criteria)
        decision = "INCORPORATE" if passed >= 3 else "REJECT"
        if passed >= 3:
            gate_passed += 1

        print(f"  {b.strategy_name:<22} {bs.n_trades:>6}  {bs.win_rate:>5.1f}%  "
              f"{bs.expectancy:>+5.2f}%  {bs.profit_factor:>5.2f}  {bs.total_pnl_rs:>10,.0f}  "
              f"{d_trades:>+8}  {d_wr:>+6.1f}pp  {d_exp:>+6.2f}%  {decision:>10}")

    print("  " + "-" * 96)
    print(f"\n  Engines passing gate: {gate_passed}/4")
    if gate_passed >= 3:
        print("  ✓ VERDICT: INCORPORATE — filters improve ≥ 3/4 engines. Add to production + build UI charts.")
    else:
        print("  ✗ VERDICT: REJECT — filters do not move the needle broadly. Proceed with current v2.x.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "baseline":
        _run("BASELINE (no filters)", False, False, False)
    elif mode == "enhanced":
        _run("ENHANCED (HV+VIX+VWAP)", True, True, True)
    else:
        # Fetch data once — shared across both runs via _UNIVERSE_DATA_CACHE
        print("\n[A/B eval] Fetching universe data (shared across both runs)...")
        baseline = _run("BASELINE (no filters)", False, False, False)
        # Data already cached — enhanced run only re-runs simulations, no re-fetch
        enhanced = _run("ENHANCED (HV+VIX+VWAP)", True, True, True)
        _compare(baseline, enhanced)
