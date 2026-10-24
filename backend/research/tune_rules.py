"""
Tune the rule strategies' parameters honestly.

For each parameter combination (module constants patched at runtime):
  1. per-symbol trades over the full history (strategy_evaluator._simulate_symbol)
  2. capital-constrained portfolio (scripts/portfolio_backtest.run_portfolio)
     scored separately on SELECT (2017-06 → 2024-04) and TEST (2024-07 → latest)
The configuration with the best SELECT Sharpe (≥ 150 trades, positive CAGR)
is chosen; its TEST result is reported as the honest estimate.

  ASTRA_OFFLINE_DATA_DIR=... python research/tune_rules.py STAGE2 PULLBACK
"""
from __future__ import annotations

import importlib
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

import pandas as pd   # noqa: E402

SELECT = ("2017-06-01", "2024-04-30")
TEST = ("2024-07-01", None)
OUT = BACKEND / "scripts" / "out"

GRIDS = {
    "STAGE2": ("app.strategies.stage2", "ASTRA.STAGE2", {
        "HIGH_PROXIMITY_PCT": [0.90, 0.95],
        "VOL_MULTIPLIER": [1.2, 1.5],
        "ATR_STOP_MULT": [2.0, 2.5, 3.0],
        "TIME_STOP_DAYS": [40, 80],
    }),
    "PULLBACK": ("app.strategies.pullback", "ASTRA.PULLBACK", {
        "TARGET_ATR_MULT": [3.0, 4.5, 6.0],
        "ATR_STOP_MULT": [1.5, 2.0, 2.5],
        "USE_SECTOR_FILTER": [True, False],
    }),
    "MOMENTUM": ("app.strategies.momentum", "ASTRA.MOMENTUM", {
        "TOP_PCT_ENTRY": [0.10, 0.20],
        "TARGET_ATR_MULT": [5.0, 8.0],
        "ATR_STOP_MULT": [2.0, 3.0],
        "ADX_MIN_THRESHOLD": [0, 20],
    }),
    "QUALITY": ("app.strategies.quality", "ASTRA.QUALITY", {
        "TOP_PCT_ENTRY": [0.10, 0.20],
        "TARGET_ATR_MULT": [3.0, 6.0],
        "ATR_STOP_MULT": [2.0, 3.0],
        "ADX_MAX_THRESHOLD": [35, 100],
    }),
}
MAX_POS = [8, 12]


def window(res_trades, data, start, end, max_pos):
    from portfolio_backtest import run_portfolio
    tr = [t for t in res_trades if pd.Timestamp(t.entry_ts).tz_localize(None) >= pd.Timestamp(start)
          and (end is None or pd.Timestamp(t.entry_ts).tz_localize(None) <= pd.Timestamp(end))]
    r = run_portfolio(tr, data, start, end, max_pos=max_pos)
    r.pop("_curve", None)
    return r


def tune(key: str, data: dict) -> dict:
    from portfolio_backtest import generate_trades
    from app.strategies.registry import get_strategy
    mod_name, strat_name, grid = GRIDS[key]
    mod = importlib.import_module(mod_name)
    defaults = {k: getattr(mod, k) for k in grid}
    rows = []
    t0 = time.time()
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        for k, v in params.items():
            setattr(mod, k, v)
        strat = get_strategy(strat_name)
        trades = generate_trades(strat, data, SELECT[0], None)
        for mp in MAX_POS:
            sel = window(trades, data, SELECT[0], SELECT[1], mp)
            tst = window(trades, data, TEST[0], TEST[1], mp)
            rows.append({"params": params, "max_pos": mp, "select": sel, "test": tst})
        print(f"[{key}] {params} trades={len(trades)} ({time.time()-t0:.0f}s)", flush=True)
    for k, v in defaults.items():
        setattr(mod, k, v)
    elig = [r for r in rows if r["select"].get("trades_taken", 0) >= 150 and (r["select"].get("cagr_pct") or -1) > 0]
    best = max(elig or rows, key=lambda r: r["select"].get("sharpe") or -9)
    base = next(r for r in rows if r["params"] == defaults and r["max_pos"] == 8) if any(
        r["params"] == defaults for r in rows) else None
    out = {"strategy": strat_name, "defaults": defaults, "baseline": base, "best": best,
           "top": sorted(rows, key=lambda r: -(r["select"].get("sharpe") or -9))[:8]}
    (OUT / f"tune_{key}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[{key}] BEST {json.dumps(best, default=str)}", flush=True)
    return out


def main():
    from portfolio_backtest import load_universe
    from app.services.intraday_universe import INTRADAY_UNIVERSE
    from app.services import strategy_evaluator as ev
    ev.EVAL_WINDOW["history_period"] = "10y"
    data = load_universe(INTRADAY_UNIVERSE)
    for key in sys.argv[1:] or list(GRIDS):
        tune(key, data)


if __name__ == "__main__":
    main()
