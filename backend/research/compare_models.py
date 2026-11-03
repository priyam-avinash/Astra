"""
Compare model scores (from oos_scores.py) as stock-ranking engines.

For each score and each trading-rule config:
  select folds  2019, 2020, 2021, 2022, 2023-01→2024-04  (choose here only)
  test          2024-07 → latest                         (reported, never used to choose)
A config is chosen per score by mean fold Sharpe among configs profitable in ≥ 4 of 5 folds.
All scores are out-of-sample (walk-forward refits), so the fold results are honest too.
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

from research.run_research import OUT   # noqa: E402
import research.simulate as sim         # noqa: E402

FOLDS = [("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"),
         ("2022-01-01", "2022-12-31"), ("2023-01-01", "2024-04-30")]
TEST = ("2024-07-01", "2026-12-31")
GRID = [dict(stop_k=s, tgt_k=t, max_bars=b, max_pos=p, q=q, regime=r, trend=tr)
        for (s, t, b) in ((2.0, 4.0, 40), (2.0, 6.0, 60), (3.0, 9.0, 120))
        for p in (10, 15) for q in (0.9, 0.8) for r in (False, True) for tr in (False, True)]


def add_scores(ds):
    rk = lambda c: ds.groupby("date")[c].rank(pct=True)   # noqa: E731
    for c in ("et_tb15", "et_xs20", "et_tb20", "mom"):
        ds["r_" + c] = rk(c)
    ds["ens_ml"] = (ds["r_et_tb15"] + ds["r_et_xs20"] + ds["r_et_tb20"]) / 3
    ds["ens_all"] = (ds["r_et_tb15"] + ds["r_et_xs20"] + ds["r_et_tb20"] + ds["r_mom"]) / 4
    return ds


def run(ds, score, g, start, end):
    q = ds.groupby("date")[score].rank(pct=True)
    mask = q >= g["q"]
    if g["trend"]:
        mask &= ds["Dist_SMA200"] > 0
    return sim.simulate(ds.assign(_s=ds[score]), "_s", mask, start, end, max_pos=g["max_pos"],
                        stop_k=g["stop_k"], tgt_k=g["tgt_k"], max_bars=g["max_bars"],
                        regime_col="mkt_above_200" if g["regime"] else None)


def main(scores):
    sim.COST_RT, sim.SLIP = 0.0025, 0.0005
    ds = add_scores(pd.read_pickle(OUT / "oos.pkl").reset_index(drop=True))
    report = {}
    for score in scores:
        rows = []
        for g in GRID:
            folds = [run(ds, score, g, s, e) for s, e in FOLDS]
            sh = [f.get("sharpe") if f.get("sharpe") is not None else -1 for f in folds]
            rows.append({"cfg": g, "mean_sharpe": round(float(np.mean(sh)), 3),
                         "profitable": sum((f.get("total_return_pct") or 0) > 0 for f in folds),
                         "fold_ret": [f.get("total_return_pct") for f in folds]})
        elig = [r for r in rows if r["profitable"] >= 4] or rows
        best = max(elig, key=lambda r: r["mean_sharpe"])
        best["test"] = run(ds, score, best["cfg"], *TEST)
        best["full_oos"] = run(ds, score, best["cfg"], "2019-01-01", TEST[1])
        report[score] = best
        print(score, json.dumps({k: best[k] for k in ("cfg", "mean_sharpe", "profitable", "fold_ret")}),
              "TEST", json.dumps({k: best["test"].get(k) for k in ("cagr_pct", "max_dd_pct", "sharpe", "trades", "yearly_pct")}),
              "FULL", json.dumps({k: best["full_oos"].get(k) for k in ("cagr_pct", "max_dd_pct", "sharpe", "yearly_pct")}),
              flush=True)
    (OUT / "compare_models.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["et_tb15", "et_xs20", "et_tb20", "mom", "ens_ml", "ens_all"])
