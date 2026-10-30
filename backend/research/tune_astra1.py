"""
Tune ASTRA 1.0 (rule engine) — vectorised replica of the live 7-confirmation
logic for fast search; the chosen config is then verified once with the
live-parity evaluator (ai_engine.decide via ai_wrappers.Astra1Strategy).

Live logic replicated per stock-day:
  BUY checks (7): RSI<rsi_max & RSI rising · above SMA200 · ADX>adx_min ·
  MACD>signal · volume>vol_min×avg · bullish candle (close>open & lower shadow
  >30% of range) · BB %B<bb_max.  Needs min_bull (NIFTY>SMA200) or min_bear.
  BUY only if buy score > SELL score (6 sell checks) and weekly trend ≠ BEAR.
Select 2017-06 → 2024-04, test 2024-07 →.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from research.run_research import load_ds, OUT   # noqa: E402
import research.simulate as sim                   # noqa: E402

SELECT = ("2017-06-01", "2024-04-30")
TEST = ("2024-07-01", "2026-12-31")


def prep(ds: pd.DataFrame) -> pd.DataFrame:
    from app.strategies.ai_wrappers import _weekly_trend_series
    ds = ds.copy()
    ds["weekly"] = "NEUTRAL"
    for sym, g in ds.groupby("symbol"):
        s = g.set_index("date")["Close"]
        ds.loc[g.index, "weekly"] = _weekly_trend_series(s).values
    rng = (ds["High"] - ds["Low"]).replace(0, np.nan)
    ds["bull_candle"] = (ds["Close"] > ds["Open"]) & ((ds["Open"] - ds["Low"]) / rng > 0.3)
    sell = ((ds["RSI"] > 65) & (ds["RSI_Slope"] < 0)).astype(int) + (ds["Dist_SMA200"] < 0).astype(int) \
        + (ds["MACD"] < ds["MACD_Signal"]).astype(int) + ((ds["Volume_Ratio"] > 1.5) & (ds["Daily_Ret"] < 0)).astype(int) \
        + (ds["BB_PctB"] > 0.8).astype(int) + (ds["Stoch_K"] > 80).astype(int)
    ds["sell_conf"] = sell / 6 * 100
    return ds


def buy_mask(ds, c):
    n = ((ds["RSI"] < c["rsi_max"]) & (ds["RSI_Slope"] > 0)).astype(int) + (ds["Dist_SMA200"] > 0).astype(int) \
        + (ds["ADX"] > c["adx_min"]).astype(int) + (ds["MACD"] > ds["MACD_Signal"]).astype(int) \
        + (ds["Volume_Ratio"] > c["vol_min"]).astype(int) + ds["bull_candle"].astype(int) \
        + (ds["BB_PctB"] < c["bb_max"]).astype(int)
    conf = n / 7 * 100
    bull = ds["mkt_above_200"] > 0.5
    need = np.where(bull, c["min_bull"], c["min_bear"])
    ok = (n >= need) & (conf > ds["sell_conf"]) & (ds["weekly"] != "BEAR")
    if c["block_bear"]:
        ok &= bull
    return ok, conf


def main():
    sim.COST_RT, sim.SLIP = 0.0025, 0.0005
    ds = prep(load_ds())
    base = dict(rsi_max=50, adx_min=25, vol_min=1.5, bb_max=0.35, min_bull=4, min_bear=5, block_bear=False,
                sl_atr=1.5, tp_atr=3.0, max_bars=60)
    grid = {"min_bull": [4, 5, 6], "block_bear": [False, True], "rsi_max": [50, 60],
            "sl_atr": [1.5, 2.0, 2.5], "tp_atr": [3.0, 4.5, 6.0]}
    rows = []
    for combo in itertools.product(*grid.values()):
        c = {**base, **dict(zip(grid, combo))}
        c["min_bear"] = max(c["min_bull"] + 1, 5)
        ok, conf = buy_mask(ds, c)
        d = ds.assign(score=conf)
        for mp in (8, 12):
            kw = dict(max_pos=mp, stop_k=c["sl_atr"], tgt_k=c["tp_atr"], max_bars=c["max_bars"])
            rows.append({"cfg": c, "max_pos": mp,
                         "select": sim.simulate(d, "score", ok, *SELECT, **kw),
                         "test": sim.simulate(d, "score", ok, *TEST, **kw)})
        print(c, rows[-1]["select"].get("cagr_pct"), rows[-1]["test"].get("cagr_pct"), flush=True)
    live = next(r for r in rows if all(r["cfg"][k] == base[k] for k in grid) and r["max_pos"] == 8)
    elig = [r for r in rows if r["select"].get("trades", 0) >= 150]
    best = max(elig or rows, key=lambda r: r["select"].get("sharpe") or -9)
    out = {"best": best, "live_defaults": live,
           "top": sorted(rows, key=lambda r: -(r["select"].get("sharpe") or -9))[:10]}
    (OUT / "tune_astra1.json").write_text(json.dumps(out, indent=2, default=str))
    print("BEST", json.dumps(best, default=str))
    print("LIVE", json.dumps(live, default=str))


if __name__ == "__main__":
    main()
