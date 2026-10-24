"""
Portfolio-level backtest for ASTRA engines/strategies → year-by-year P&L.

Per-symbol trades come from the same walk-forward simulator the evaluator
uses (strategy_evaluator._simulate_symbol: next-bar-open entries, gap-aware
stops, costs). They are then run through a capital-constrained portfolio:

  - starting capital ₹10,00,000, at most MAX_POS open positions
  - each new position gets equity / MAX_POS at entry (compounding)
  - candidate entries on the same day are taken in a fixed random order
    (seeded) until slots run out; skipped signals are counted
  - equity marked to market daily on closes; costs already in trade returns

Usage (offline data):
  ASTRA_OFFLINE_DATA_DIR=/path/export python scripts/portfolio_backtest.py \
      --strategies ASTRA.1.0,ASTRA.PULLBACK --start 2017-01-01
  # held-out ML models:
  ASTRA_MODELS_DIR=models_oos ... --strategies ASTRA.AI,ASTRA.ML --start 2024-07-01
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

logging.basicConfig(level=logging.WARNING)
CAPITAL = 1_000_000.0


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def load_universe(symbols, history="10y") -> dict:
    from app.services.market_data import market_data
    out = {}
    for s in symbols:
        df = market_data.get_ohlcv(f"{s}.NS", period=history, interval="1d")
        if len(df) > 260:
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.attrs["symbol"] = s
            out[s] = df
    return out


def generate_trades(strategy, data: dict, start, end, workers: int = 2) -> list:
    from concurrent.futures import ThreadPoolExecutor
    from app.services import strategy_evaluator as ev
    from app.strategies.base import CostModel
    ev.EVAL_WINDOW.update(trade_start=pd.Timestamp(start), trade_end=pd.Timestamp(end) if end else None)
    try:
        strategy.prepare_universe(data)
    except Exception as e:
        print(f"  prepare_universe failed: {e}")
    cost = CostModel()
    trades = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda s: ev._simulate_symbol(strategy, s, data[s], cost), list(data)):
            trades.extend(res)
    return trades


def run_portfolio(trades: list, data: dict, start, end, max_pos: int = 10, seed: int = 7) -> dict:
    start, end = _naive(start), _naive(end) if end else None
    closes = {s: pd.Series(df["Close"].values, index=[_naive(t) for t in df.index]) for s, df in data.items()}
    days = sorted({d for c in closes.values() for d in c.index if d >= start and (end is None or d <= end)})
    if not days:
        return {}
    by_entry: dict = {}
    for t in trades:
        by_entry.setdefault(_naive(t.entry_ts).normalize(), []).append(t)
    rng = random.Random(seed)

    cash, open_pos, curve = CAPITAL, [], []
    taken = skipped = 0
    realised = []
    for d in days:
        dn = d.normalize()
        # 1) exits scheduled for today (exit fill already includes costs via net_pct)
        still = []
        for p in open_pos:
            if _naive(p["t"].exit_ts).normalize() <= dn:
                proceeds = p["alloc"] * (1 + p["t"].net_pct / 100.0)
                cash += proceeds
                realised.append(p["t"].net_pct)
            else:
                still.append(p)
        open_pos = still
        # 2) entries today
        cands = by_entry.get(dn, [])
        rng.shuffle(cands)
        held = {p["t"].symbol for p in open_pos}
        for t in cands:
            if t.symbol in held:
                continue
            if len(open_pos) >= max_pos:
                skipped += 1
                continue
            equity = cash + sum(_mtm(p, closes, d) for p in open_pos)
            alloc = min(cash, equity / max_pos)
            if alloc <= 1000:
                skipped += 1
                continue
            cash -= alloc
            open_pos.append({"t": t, "alloc": alloc})
            held.add(t.symbol)
            taken += 1
        # 3) mark to market
        equity = cash + sum(_mtm(p, closes, d) for p in open_pos)
        curve.append((d, equity, len(open_pos)))

    eq = pd.Series([e for _, e, _ in curve], index=[d for d, _, _ in curve])
    n_open = pd.Series([n for _, _, n in curve], index=eq.index)
    out = summarize(eq, n_open, taken, skipped, realised, max_pos)
    out["_curve"] = eq
    return out


def _mtm(p, closes, d) -> float:
    t = p["t"]
    c = closes.get(t.symbol)
    if c is None:
        return p["alloc"]
    px = c.asof(d)
    if pd.isna(px) or t.entry_price <= 0:
        return p["alloc"]
    return p["alloc"] * float(px) / float(t.entry_price)


def summarize(eq: pd.Series, n_open: pd.Series, taken, skipped, realised, max_pos) -> dict:
    years = {}
    prev = CAPITAL
    for y, g in eq.groupby(eq.index.year):
        last = float(g.iloc[-1])
        years[int(y)] = round((last / prev - 1) * 100, 1)
        prev = last
    total_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / CAPITAL) ** (1 / total_years) - 1 if total_years > 0.2 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    r = np.array(realised) if realised else np.array([0.0])
    return {
        "final_equity": round(float(eq.iloc[-1])),
        "cagr_pct": round(float(cagr) * 100, 1) if pd.notna(cagr) else None,
        "max_drawdown_pct": round(float(dd) * 100, 1),
        "sharpe": round(float(sharpe), 2) if pd.notna(sharpe) else None,
        "avg_exposure_pct": round(float((n_open / max_pos).mean()) * 100, 1),
        "trades_taken": taken, "signals_skipped_full": skipped,
        "win_rate_pct": round(float((r > 0).mean()) * 100, 1),
        "avg_trade_net_pct": round(float(r.mean()), 2),
        "yearly_return_pct": years,
        "years_ge_20pct": sum(1 for v in years.values() if v >= 20),
        "years_negative": sum(1 for v in years.values() if v < 0),
    }


def benchmark_nifty(start, end) -> dict:
    from app.services.market_data import market_data
    n = market_data.get_ohlcv("^NSEI", period="10y", interval="1d")["Close"]
    n.index = [_naive(t) for t in n.index]
    n = n[(n.index >= _naive(start)) & ((n.index <= _naive(end)) if end else True)]
    eq = n / n.iloc[0] * CAPITAL
    out = summarize(eq, pd.Series(10, index=eq.index), 0, 0, [], 10)
    return {k: out[k] for k in ("cagr_pct", "max_drawdown_pct", "sharpe", "yearly_return_pct")}


def make_strategy(name: str):
    from app.strategies.registry import get_strategy
    s = get_strategy(name)
    if s is None:
        raise SystemExit(f"unknown strategy {name}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="ASTRA.1.0,ASTRA.PULLBACK,ASTRA.MOMENTUM,ASTRA.STAGE2,ASTRA.QUALITY")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--combine", action="store_true", help="also report an equal-capital multi-strategy book")
    args = ap.parse_args()
    if not os.getenv("ASTRA_OFFLINE_DATA_DIR"):
        sys.exit("Set ASTRA_OFFLINE_DATA_DIR")

    from app.services.intraday_universe import INTRADAY_UNIVERSE
    t0 = time.time()
    data = load_universe(INTRADAY_UNIVERSE)
    print(f"universe: {len(data)} symbols loaded in {time.time()-t0:.0f}s")
    out_path = HERE / "out" / f"portfolio_{args.tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    report["_window"] = {"start": args.start, "end": args.end, "max_positions": args.max_pos,
                         "capital": CAPITAL, "models_dir": os.getenv("ASTRA_MODELS_DIR", "saved_models")}
    report["NIFTY50"] = benchmark_nifty(args.start, args.end)
    curves = {}
    for name in args.strategies.split(","):
        t1 = time.time()
        strat = make_strategy(name)
        trades = generate_trades(strat, data, args.start, args.end)
        res = run_portfolio(trades, data, args.start, args.end, max_pos=args.max_pos)
        res["raw_trades"] = len(trades)
        res["seconds"] = round(time.time() - t1)
        curves[name] = res.pop("_curve")
        report[name] = res
        print(name, json.dumps(res))
        out_path.write_text(json.dumps(report, indent=2, default=str))
    if args.combine and len(curves) > 1:
        # equal capital per sleeve, each sleeve runs its own book (rebalanced never)
        df = pd.concat(curves, axis=1).ffill().dropna()
        combo = (df / CAPITAL).mean(axis=1) * CAPITAL
        res = summarize(combo, pd.Series(args.max_pos, index=combo.index), 0, 0, [], args.max_pos)
        res = {k: res[k] for k in ("final_equity", "cagr_pct", "max_drawdown_pct", "sharpe", "yearly_return_pct",
                                   "years_negative")}
        res["sleeves"] = list(curves)
        report["COMBINED"] = res
        print("COMBINED", json.dumps(res))
        out_path.write_text(json.dumps(report, indent=2, default=str))
        pd.concat(curves, axis=1).to_csv(out_path.with_suffix(".curves.csv"))
    print(json.dumps(report["NIFTY50"]))


if __name__ == "__main__":
    main()
