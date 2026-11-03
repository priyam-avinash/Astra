"""Regime-filter study for the model rankers (out-of-sample scores from oos_scores.py).
Regimes: none | NIFTY>200DMA | breadth (share of universe above its 200DMA) > 0.5 / 0.6 | breadth > 0.5 AND NIFTY 20d return > 0.
Chosen on the 2019–2024-04 folds, reported on the untouched 2024-07 → test."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from research.run_research import OUT
import research.simulate as sim
from research.compare_models import add_scores, FOLDS, TEST

def main():
    sim.COST_RT, sim.SLIP = 0.0025, 0.0005
    ds = add_scores(pd.read_pickle(OUT / "oos.pkl").reset_index(drop=True))
    br = ds.assign(a=(ds["Dist_SMA200"] > 0).astype(float)).groupby("date")["a"].mean()
    ds["breadth"] = ds["date"].map(br)
    ds["rg_200"] = ds["mkt_above_200"]
    ds["rg_b50"] = (ds["breadth"] > 0.5).astype(float)
    ds["rg_b60"] = (ds["breadth"] > 0.6).astype(float)
    ds["rg_b50m"] = ((ds["breadth"] > 0.5) & (ds["mkt_ret_20"] > 0)).astype(float)
    out = {}
    for score in sys.argv[1:] or ["et_tb15", "ens_all"]:
        q = ds.groupby("date")[score].rank(pct=True)
        rows = []
        for (s, t, b) in ((2.0, 6.0, 60), (3.0, 9.0, 120)):
            for rg in (None, "rg_200", "rg_b50", "rg_b60", "rg_b50m"):
                mask = (q >= 0.8) & (ds["Dist_SMA200"] > 0)
                kw = dict(max_pos=15, stop_k=s, tgt_k=t, max_bars=b, regime_col=rg)
                d = ds.assign(_s=ds[score])
                folds = [sim.simulate(d, "_s", mask, a, e, **kw) for a, e in FOLDS]
                sh = [f.get("sharpe") or -1 for f in folds]
                tst = sim.simulate(d, "_s", mask, *TEST, **kw)
                rows.append({"stop": s, "tgt": t, "bars": b, "regime": rg, "mean_sharpe": round(float(np.mean(sh)), 3),
                             "fold_ret": [f.get("total_return_pct") for f in folds],
                             "test": {k: tst.get(k) for k in ("cagr_pct", "max_dd_pct", "sharpe", "trades", "yearly_pct")}})
                print(score, json.dumps(rows[-1]), flush=True)
        out[score] = rows
    (OUT / "regime_study.json").write_text(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
