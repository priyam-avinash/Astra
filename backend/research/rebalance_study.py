"""Periodic-rebalance portfolios on the out-of-sample model scores (no stops/targets).
Every `every` trading days hold the top `n` stocks by score (signal on day t, traded at
t+1 open approximated by holding from t+1 close-to-close; costs 0.30%/side on turnover).
Hysteresis: a held stock stays while its score rank is in the top `keep` fraction."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from research.run_research import OUT
from research.compare_models import add_scores, FOLDS, TEST

COST = 0.003

def book(ds, score, n, every, keep, trend, start, end):
    d = ds[(ds.date >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (ds.date <= pd.Timestamp(end))]
    px = d.pivot(index="date", columns="symbol", values="Close")
    ret = px.pct_change(fill_method=None).fillna(0.0)
    sc = d.pivot(index="date", columns="symbol", values=score)
    rk = sc.rank(axis=1, pct=True)
    up = d.pivot(index="date", columns="symbol", values="Dist_SMA200") > 0
    dates = px.index[px.index >= pd.Timestamp(start)]
    held, pending, eq, curve = [], None, 1.0, []
    for k, t in enumerate(dates):
        if k > 0:
            r = ret.loc[t, held].mean() if held else 0.0
            eq *= 1 + r
        if pending is not None:          # trade one day after the signal (at that day's close)
            nh, pending = pending, None
            turn = len(set(nh) ^ set(held)) / max(n, 1)
            eq *= 1 - COST * turn
            held = nh
        if k % every == 0:
            row = rk.loc[t].dropna()
            ok = row[up.loc[t, row.index]] if trend else row
            keepers = [s for s in held if s in row.index and row[s] >= 1 - keep]
            new = [s for s in ok.sort_values(ascending=False).index if s not in keepers][: max(0, n - len(keepers))]
            pending = keepers + new
        curve.append((t, eq))
    c = pd.Series(dict(curve))
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    dr = c.pct_change().dropna()
    y = c.resample("YE").last(); y = pd.concat([c.iloc[:1], y]).pct_change().dropna()
    return {"cagr_pct": round(((c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, 0.1)) - 1) * 100, 1),
            "total_return_pct": round((c.iloc[-1] / c.iloc[0] - 1) * 100, 1),
            "max_dd_pct": round(((c / c.cummax()) - 1).min() * 100, 1),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(252)), 2) if dr.std() > 0 else None,
            "yearly_pct": {str(k.year): round(v * 100, 1) for k, v in y.items()}}

def main():
    ds = add_scores(pd.read_pickle(OUT / "oos.pkl").reset_index(drop=True))
    ds["ens_tb"] = (ds["r_et_tb15"] + ds["r_et_tb20"]) / 2
    res = {}
    for score in sys.argv[1:] or ["et_tb15", "ens_tb", "ens_all", "et_xs20"]:
        rows = []
        for n in (10, 20):
            for every in (10, 20):
                for keep in (0.2, 0.4):
                    for trend in (False, True):
                        folds = [book(ds, score, n, every, keep, trend, a, e) for a, e in FOLDS]
                        rows.append({"n": n, "every": every, "keep": keep, "trend": trend,
                                     "mean_sharpe": round(float(np.mean([f["sharpe"] or -1 for f in folds])), 3),
                                     "fold_ret": [f["total_return_pct"] for f in folds]})
        best = max([r for r in rows if sum(x > 0 for x in r["fold_ret"]) >= 4] or rows, key=lambda r: r["mean_sharpe"])
        best["test"] = book(ds, score, best["n"], best["every"], best["keep"], best["trend"], *TEST)
        res[score] = best
        print(score, json.dumps(best), flush=True)
    ew = ds.assign(_s=1.0)
    print("EW_universe_test", json.dumps(book(ew, "_s", 200, 10000, 1.0, False, *TEST)))
    (OUT / "rebalance_study.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
