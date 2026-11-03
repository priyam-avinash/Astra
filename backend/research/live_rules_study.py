"""Live-implementable version of the rank-rotation book (out-of-sample scores).
Daily: buy the best-ranked stocks (must be in today's top `top` by score) into free
slots (max `slots`); a holding is sold once it has been held `min_hold` trading days
and its score rank drops below `keep_pct`; optional catastrophic stop at entry − k×ATR.
Signals at day t's close, trades at day t+1's open; costs 0.30% per side."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from research.run_research import OUT
from research.compare_models import add_scores, FOLDS, TEST

SIDE = 0.003

def run(ds, score, start, end, slots=10, top=10, min_hold=20, keep_pct=0.6, stop_k=None, curve_out=None):
    d = ds[(ds.date >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (ds.date <= pd.Timestamp(end))]
    P = {c: d.pivot(index="date", columns="symbol", values=c) for c in ("Open", "Close", "Low", "ATR")}
    sc = d.pivot(index="date", columns="symbol", values=score)
    pct = sc.rank(axis=1, pct=True)
    pos_n = sc.rank(axis=1, ascending=False)
    dates = list(P["Close"].index[P["Close"].index >= pd.Timestamp(start)])
    cash, hold, curve, trades = 1.0, {}, [], []   # hold: sym -> dict(units, entry, stop, age)
    orders_buy, orders_sell = [], []
    for t in dates:
        o = P["Open"].loc[t]
        for s in orders_sell:                       # exits at today's open
            h = hold.pop(s); px = o.get(s)
            px = h["last"] if px is None or np.isnan(px) else px
            cash += h["units"] * px * (1 - SIDE); trades.append(px / h["entry"] - 1)
        orders_sell = []
        if orders_buy:
            free = slots - len(hold)
            eq = cash + sum(h["units"] * h["last"] for h in hold.values())
            for s in orders_buy[:free]:
                px = o.get(s)
                if px is None or np.isnan(px) or s in hold: continue
                alloc = min(eq / slots, cash)
                if alloc <= 0: break
                u = alloc * (1 - SIDE) / px; cash -= alloc
                atr = P["ATR"].loc[t].get(s, np.nan)
                hold[s] = {"units": u, "entry": px, "last": px, "age": 0,
                           "stop": px - stop_k * atr if stop_k and not np.isnan(atr) else -1}
        orders_buy = []
        c, lo = P["Close"].loc[t], P["Low"].loc[t]
        for s, h in hold.items():
            if not np.isnan(c.get(s, np.nan)): h["last"] = c[s]
            h["age"] += 1
            if h["stop"] > 0 and lo.get(s, np.inf) <= h["stop"]:
                orders_sell.append(s)               # stop breached: sell next open (conservative)
            elif h["age"] >= min_hold and pct.loc[t].get(s, 0) < keep_pct:
                orders_sell.append(s)
        eq = cash + sum(h["units"] * h["last"] for h in hold.values())
        curve.append((t, eq))
        free = slots - len(hold) + len(orders_sell)
        if free > 0:
            cand = pos_n.loc[t].dropna().sort_values()
            orders_buy = [s for s in cand.index[:top] if s not in hold][:free]
    c = pd.Series(dict(curve))
    if curve_out is not None:
        curve_out.append(c)
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    dr = c.pct_change().dropna()
    y = c.resample("YE").last(); y = pd.concat([c.iloc[:1], y]).pct_change().dropna()
    return {"cagr_pct": round(((c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, .1)) - 1) * 100, 1),
            "total_return_pct": round((c.iloc[-1] / c.iloc[0] - 1) * 100, 1),
            "max_dd_pct": round(((c / c.cummax()) - 1).min() * 100, 1),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(252)), 2) if dr.std() > 0 else None,
            "trades": len(trades), "win_rate_pct": round(100 * np.mean([x > 0 for x in trades]), 1) if trades else None,
            "yearly_pct": {str(k.year): round(v * 100, 1) for k, v in y.items()}}

def main():
    ds = add_scores(pd.read_pickle(OUT / "oos.pkl").reset_index(drop=True))
    ds["ens_tb"] = (ds["r_et_tb15"] + ds["r_et_tb20"]) / 2
    rows = []
    for slots in (10, 15):
        for min_hold in (10, 20, 40):
            for keep in (0.5, 0.6, 0.7):
                for stop_k in (None, 4.0):
                    kw = dict(slots=slots, top=slots, min_hold=min_hold, keep_pct=keep, stop_k=stop_k)
                    folds = [run(ds, "ens_tb", a, e, **kw) for a, e in FOLDS]
                    rows.append({"kw": kw, "mean_sharpe": round(float(np.mean([f["sharpe"] or -1 for f in folds])), 3),
                                 "fold_ret": [f["total_return_pct"] for f in folds]})
                    print(json.dumps(rows[-1]), flush=True)
    best = max([r for r in rows if sum(x > 0 for x in r["fold_ret"]) >= 4] or rows, key=lambda r: r["mean_sharpe"])
    best["test"] = run(ds, "ens_tb", *TEST, **best["kw"])
    best["full_oos"] = run(ds, "ens_tb", "2019-01-01", TEST[1], **best["kw"])
    best["test_et_tb15_same_rules"] = run(ds, "et_tb15", *TEST, **best["kw"])
    print("BEST", json.dumps(best), flush=True)
    (OUT / "live_rules_study.json").write_text(json.dumps({"best": best, "rows": rows}, indent=2))

if __name__ == "__main__":
    main()
