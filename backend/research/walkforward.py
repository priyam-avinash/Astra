"""
Walk-forward selection of ASTRA.AI v2 trading rules (more robust than one
validation window).

Five expanding-window folds, each validated on a different market year:
  fold  train ends     validate
  2019  2018-10-31     2019-01-01 → 2019-12-31
  2020  2019-10-31     2020
  2021  2020-10-31     2021
  2022  2021-10-31     2022
  2023  2022-10-31     2023-01-01 → 2024-04-30
(2-month embargo between train end and validation: labels look ≤ 40 bars ahead.)

A configuration is chosen by its mean Sharpe across folds, requiring a profit
in at least 4 of 5 folds. The chosen rules are then run on the untouched
TEST window (2024-07 →) twice:
  static   one model fit on data to 2024-04-30
  rolling  model refit every 6 months on all data available 2 months before
           each segment (how production retraining works)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from research.run_research import OUT, LABELS, make_model, load_ds   # noqa: E402
from research.simulate import simulate                                # noqa: E402
from app.services.features_v2 import feature_columns                  # noqa: E402

FOLDS = [
    ("2018-10-31", "2019-01-01", "2019-12-31"),
    ("2019-10-31", "2020-01-01", "2020-12-31"),
    ("2020-10-31", "2021-01-01", "2021-12-31"),
    ("2021-10-31", "2022-01-01", "2022-12-31"),
    ("2022-10-31", "2023-01-01", "2024-04-30"),
]
TEST_START = pd.Timestamp("2024-07-01")
GRID = [dict(stop_k=s, tgt_k=t, max_bars=b, max_pos=p, q=q, regime=r)
        for (s, t, b) in ((1.5, 3.0, 20), (2.0, 4.0, 40), (2.0, 6.0, 60))
        for p in (5, 10, 15) for q in (0.95, 0.9, 0.8) for r in (False, True)]


def scored(ds, feats, col, algo, train_end, start, end):
    tr = ds[ds["date"] <= pd.Timestamp(train_end)].dropna(subset=[col])
    m = make_model("reg", algo)
    m.fit(tr[feats].values, tr[col].values.astype(float))
    mask = (ds["date"] >= pd.Timestamp(start)) & (ds["date"] <= pd.Timestamp(end))
    out = ds.copy()
    out["score"] = np.nan
    out.loc[mask, "score"] = m.predict(ds.loc[mask, feats].values)
    out["q"] = out.groupby("date")["score"].rank(pct=True)
    return out


def run(ds_scored, g, start, end):
    return simulate(ds_scored, "score", ds_scored["q"] >= g["q"], start, end,
                    max_pos=g["max_pos"], stop_k=g["stop_k"], tgt_k=g["tgt_k"],
                    max_bars=g["max_bars"], regime_col="mkt_above_200" if g["regime"] else None)


def main(label="tb15", algo="et"):
    col, _ = LABELS[label]
    ds = load_ds()
    feats = feature_columns()
    ds = ds.dropna(subset=feats).reset_index(drop=True)
    last = ds["date"].max()
    table = {json.dumps(g, sort_keys=True): [] for g in GRID}
    t0 = time.time()
    for train_end, vs, ve in FOLDS:
        s = scored(ds, feats, col, algo, train_end, vs, ve)
        for g in GRID:
            res = run(s, g, vs, ve)
            table[json.dumps(g, sort_keys=True)].append(
                {"fold": vs[:4], "ret": res.get("total_return_pct"), "sharpe": res.get("sharpe"),
                 "dd": res.get("max_dd_pct"), "trades": res.get("trades")})
        print(f"fold {vs[:4]} done ({time.time()-t0:.0f}s)", flush=True)
    summary = []
    for k, rows in table.items():
        sh = [r["sharpe"] if r["sharpe"] is not None else -1 for r in rows]
        summary.append({"config": json.loads(k), "mean_sharpe": round(float(np.mean(sh)), 3),
                        "min_ret": min(r["ret"] for r in rows), "profitable_folds": sum(r["ret"] > 0 for r in rows),
                        "mean_ret": round(float(np.mean([r["ret"] for r in rows])), 1), "folds": rows})
    eligible = [x for x in summary if x["profitable_folds"] >= 4] or summary
    best = max(eligible, key=lambda x: x["mean_sharpe"])
    g = best["config"]
    print("SELECTED", json.dumps(best), flush=True)

    # TEST — static model
    s = scored(ds, feats, col, algo, "2024-04-30", TEST_START, last)
    static = run(s, g, TEST_START, last)
    # TEST — rolling 6-monthly refits
    seg_starts = pd.date_range(TEST_START, last, freq="6MS")
    rolled = ds.copy()
    rolled["score"] = np.nan
    for i, ss in enumerate(seg_starts):
        se = seg_starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(seg_starts) else last
        te = ss - pd.Timedelta(days=61)
        part = scored(ds, feats, col, algo, te, ss, se)
        m = (ds["date"] >= ss) & (ds["date"] <= se)
        rolled.loc[m, "score"] = part.loc[m, "score"]
    rolled["q"] = rolled.groupby("date")["score"].rank(pct=True)
    rolling = run(rolled, g, TEST_START, last)
    rep = {"label": label, "algo": algo, "selected": best, "test_static": static, "test_rolling": rolling,
           "all_configs": sorted(summary, key=lambda x: -x["mean_sharpe"])[:15]}
    (OUT / f"walkforward_{label}_{algo}.json").write_text(json.dumps(rep, indent=2, default=str))
    print("TEST_STATIC", json.dumps(static))
    print("TEST_ROLLING", json.dumps(rolling))


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else []))
