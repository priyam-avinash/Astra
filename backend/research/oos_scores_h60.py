"""Out-of-sample scores for a 60-day-horizon ranker (candidate for ASTRA.ML):
ExtraTrees on the 60-trading-day return in excess of the universe average, walk-forward
segments as in oos_scores.py (embargo 100 days since the label looks 60 bars ahead)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from research.run_research import OUT, load_ds
from research.oos_scores import segments
from app.services.features_v2 import feature_columns

def main():
    from sklearn.ensemble import ExtraTreesRegressor
    ds = load_ds(); feats = feature_columns()
    ds = ds.dropna(subset=feats).sort_values(["symbol", "date"]).reset_index(drop=True)
    nxt = ds.groupby("symbol")["Open"].shift(-1)
    fut = ds.groupby("symbol")["Close"].shift(-60)
    ds["fwd60"] = fut / nxt - 1
    ds["xs60"] = ds["fwd60"] - ds.groupby("date")["fwd60"].transform("mean")
    oos = pd.read_pickle(OUT / "oos.pkl")[["date", "symbol"]]
    ds["et_xs60"] = np.nan
    t0 = time.time()
    for ss, se in segments(ds["date"].max()):
        te = ss - pd.Timedelta(days=100)
        tr = ds[ds["date"] <= te].dropna(subset=["xs60"])
        m = ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=7)
        m.fit(tr[feats].values, tr["xs60"].values)
        msk = (ds["date"] >= ss) & (ds["date"] <= se)
        ds.loc[msk, "et_xs60"] = m.predict(ds.loc[msk, feats].values)
        print(f"{ss.date()} done ({time.time()-t0:.0f}s)", flush=True)
    full = pd.read_pickle(OUT / "oos.pkl")
    full = full.merge(ds[["date", "symbol", "et_xs60"]], on=["date", "symbol"], how="left")
    full.to_pickle(OUT / "oos.pkl")
    print("merged")

if __name__ == "__main__":
    main()
