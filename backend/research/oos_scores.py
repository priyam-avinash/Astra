"""
Out-of-sample score history for model comparison.

Every row from 2019 on gets scores from models trained ONLY on data that
ended ≥ 61 days before its segment (labels look ≤ 40 bars ahead). Segments are
calendar years 2019–2023 and half-years from 2024 on (production refits every
6 months). Any trading rule evaluated on these scores is out-of-sample.

Scores written to scripts/out/oos.pkl:
  et_tb15   ExtraTrees on the 1.5/3.0-ATR triple-barrier win label (current v2)
  et_xs20   ExtraTrees on 20-day return in excess of the universe
  et_tb20   ExtraTrees on the 2.0/4.0-ATR barrier label
  mom       rule factor: rank of 6- and 12-month return + RS vs NIFTY (no fitting)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from research.run_research import OUT, load_ds   # noqa: E402
from app.services.features_v2 import feature_columns   # noqa: E402

MODELS = {"et_tb15": "tb15_30_20_win", "et_xs20": "xs_ret_20", "et_tb20": "tb20_40_40_win"}


def segments(last: pd.Timestamp):
    segs = [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")) for y in range(2019, 2024)]
    s = pd.Timestamp("2024-01-01")
    while s <= last:
        e = (s + pd.DateOffset(months=6)) - pd.Timedelta(days=1)
        segs.append((s, min(e, last)))
        s = e + pd.Timedelta(days=1)
    return segs


def main():
    from sklearn.ensemble import ExtraTreesRegressor
    ds = load_ds()
    feats = feature_columns()
    ds = ds.dropna(subset=feats).reset_index(drop=True)
    last = ds["date"].max()
    for name in MODELS:
        ds[name] = np.nan
    t0 = time.time()
    for ss, se in segments(last):
        te = ss - pd.Timedelta(days=61)
        m = (ds["date"] >= ss) & (ds["date"] <= se)
        for name, col in MODELS.items():
            tr = ds[ds["date"] <= te].dropna(subset=[col])
            mdl = ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=100,
                                      max_features=0.5, n_jobs=-1, random_state=42)
            mdl.fit(tr[feats].values, tr[col].values.astype(float))
            ds.loc[m, name] = mdl.predict(ds.loc[m, feats].values)
        print(f"segment {ss.date()}→{se.date()} done ({time.time()-t0:.0f}s)", flush=True)
    ds["mom"] = ds[["rk_ret_120", "rk_ret_250", "rk_rs_nifty_60"]].mean(axis=1)
    ds[ds["date"] >= "2019-01-01"].to_pickle(OUT / "oos.pkl")
    print("saved", OUT / "oos.pkl")


if __name__ == "__main__":
    main()
