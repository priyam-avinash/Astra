"""
ASTRA.AI research driver — walk-forward model and strategy selection.

  TRAIN  : 2017 → 2022-10-31        (embargo before validation: labels look ≤40 bars ahead)
  VALID  : 2023-01-01 → 2024-04-30  (model / label / trading-rule selection happens here only)
  TEST   : 2024-07-01 → latest      (touched once, for the final report)

Stages:
  build   features + labels for the universe → out/ds.pkl
  models  label × model grid, scored on VALID (AUC / rank-IC / top-decile returns)
  trade   portfolio rules for the best models, tuned on VALID, then run once on TEST

  ASTRA_OFFLINE_DATA_DIR=... python research/run_research.py build models trade
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
OUT = BACKEND / "scripts" / "out"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

TRAIN_END = pd.Timestamp("2022-10-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-04-30")
TEST_START = pd.Timestamp("2024-07-01")
FINAL_TRAIN_END = pd.Timestamp("2024-04-30")     # train+valid, embargoed before TEST

LABELS = {   # name: (column, kind)
    "tb15": ("tb15_30_20_win", "clf"),
    "tb20": ("tb20_40_40_win", "clf"),
    "xs20": ("xs_ret_20", "reg"),
    "fwd20": ("fwd_ret_20", "reg"),
}


def load_ds() -> pd.DataFrame:
    return pd.read_pickle(OUT / "ds.pkl")


def stage_build():
    from app.services.intraday_universe import INTRADAY_UNIVERSE
    from app.services.market_data import market_data
    from app.strategies.pullback import SECTOR_MAP
    from research.dataset import build
    t0 = time.time()
    data = {}
    for s in INTRADAY_UNIVERSE:
        df = market_data.get_ohlcv(f"{s}.NS", period="10y", interval="1d")
        if len(df) > 300:
            data[s] = df
    nifty = market_data.get_ohlcv("^NSEI", period="10y", interval="1d")
    ds = build(data, nifty, SECTOR_MAP)
    ds.to_pickle(OUT / "ds.pkl")
    print(f"dataset: {len(ds)} rows, {ds['symbol'].nunique()} symbols, "
          f"{ds['date'].min().date()} → {ds['date'].max().date()} ({time.time()-t0:.0f}s)")


def make_model(kind: str, algo: str, seed=42):
    if algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=100,
                                     max_features=0.4, n_jobs=-1, random_state=seed)
    if algo == "et":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=300, max_depth=10, min_samples_leaf=100,
                                   max_features=0.5, n_jobs=-1, random_state=seed)
    if algo == "lgbm":
        import lightgbm as lgb
        common = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                      random_state=seed, verbose=-1, n_jobs=2)
        return lgb.LGBMClassifier(**common) if kind == "clf" else lgb.LGBMRegressor(**common)
    raise ValueError(algo)


def predict(model, X, kind):
    if kind == "clf" and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def score_block(d: pd.DataFrame, pred: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    d = d.assign(pred=pred)
    out = {}
    if d["tb15_30_20_win"].notna().any():
        m = d["tb15_30_20_win"].notna()
        out["auc_tb15"] = round(float(roc_auc_score(d.loc[m, "tb15_30_20_win"], d.loc[m, "pred"])), 4)
    ric = d.groupby("date").apply(lambda g: g["pred"].rank().corr(g["fwd_ret_20"].rank())
                                  if g["fwd_ret_20"].notna().sum() >= 20 else np.nan).dropna()
    out["rank_ic_20"] = round(float(ric.mean()), 4)
    out["rank_ic_t"] = round(float(ric.mean() / ric.std() * np.sqrt(len(ric) / 20)), 2)  # overlap-adjusted
    q = d.groupby("date")["pred"].rank(pct=True)
    top, rest = d[q >= 0.9], d
    out["top10_fwd20"] = round(float(top["fwd_ret_20"].mean()), 3)
    out["all_fwd20"] = round(float(rest["fwd_ret_20"].mean()), 3)
    out["top10_tb15_ret"] = round(float(top["tb15_30_20_ret"].mean()), 3)
    out["all_tb15_ret"] = round(float(rest["tb15_30_20_ret"].mean()), 3)
    return out


def stage_models():
    from research.dataset import feature_columns
    ds = load_ds()
    feats = feature_columns()
    ds = ds.dropna(subset=feats)
    tr = ds[ds["date"] <= TRAIN_END]
    va = ds[(ds["date"] >= VAL_START) & (ds["date"] <= VAL_END)]
    results = {}
    for lname, (col, kind) in LABELS.items():
        trl = tr.dropna(subset=[col])
        for algo in ("rf", "et", "lgbm"):
            t0 = time.time()
            if algo != "lgbm" and kind == "clf":
                y = trl[col].values.astype(float)      # regressor on 0/1 → probability
            else:
                y = trl[col].values
            m = make_model(kind if algo == "lgbm" else "reg", algo)
            m.fit(trl[feats].values, y)
            p = predict(m, va[feats].values, kind if algo == "lgbm" else "reg")
            res = score_block(va, p)
            res["seconds"] = round(time.time() - t0)
            results[f"{lname}/{algo}"] = res
            print(lname, algo, res, flush=True)
            (OUT / "research_models.json").write_text(json.dumps(results, indent=2))
    # baseline: the 20 live features only, original RF target (5-day return)
    return results


def fit_predict(ds, feats, lname, algo, train_end, pred_start, pred_end):
    col, kind = LABELS[lname]
    trl = ds[(ds["date"] <= train_end)].dropna(subset=[col])
    k = kind if algo == "lgbm" else "reg"
    m = make_model(k, algo)
    m.fit(trl[feats].values, trl[col].values.astype(float) if k == "reg" else trl[col].values)
    mask = (ds["date"] >= pred_start) & (ds["date"] <= pred_end)
    return m, pd.Series(predict(m, ds.loc[mask, feats].values, k), index=ds.index[mask])


def stage_trade(candidates: list[str]):
    from research.dataset import feature_columns
    from research.simulate import simulate
    ds = load_ds()
    feats = feature_columns()
    ds = ds.dropna(subset=feats).reset_index(drop=True)
    last = ds["date"].max()
    grid = []
    for stop_k, tgt_k, mb in ((1.5, 3.0, 20), (2.0, 4.0, 40), (2.0, 6.0, 60)):
        for max_pos in (5, 10):
            for q in (0.95, 0.9, 0.8):
                for regime in (False, True):
                    grid.append(dict(stop_k=stop_k, tgt_k=tgt_k, max_bars=mb, max_pos=max_pos, q=q, regime=regime))
    report = {"valid": {}, "test": {}}
    best = None
    for cand in candidates:
        lname, algo = cand.split("/")
        _, pv = fit_predict(ds, feats, lname, algo, TRAIN_END, VAL_START, VAL_END)
        d = ds.loc[pv.index].assign(score=pv.values)
        d["q"] = d.groupby("date")["score"].rank(pct=True)
        dsv = ds.copy()
        dsv["score"] = np.nan
        dsv.loc[d.index, "score"] = d["score"]
        dsv["q"] = np.nan
        dsv.loc[d.index, "q"] = d["q"]
        for g in grid:
            res = simulate(dsv, "score", dsv["q"] >= g["q"], VAL_START, VAL_END,
                           max_pos=g["max_pos"], stop_k=g["stop_k"], tgt_k=g["tgt_k"],
                           max_bars=g["max_bars"], regime_col="mkt_above_200" if g["regime"] else None)
            key = f"{cand}|{json.dumps(g, sort_keys=True)}"
            report["valid"][key] = res
            if res.get("trades", 0) >= 40 and res.get("sharpe") is not None:
                if best is None or res["sharpe"] > best[1]["sharpe"]:
                    best = (key, res, cand, g)
        print(cand, "best so far:", best[0] if best else None, best[1] if best else None, flush=True)
        (OUT / "research_trade.json").write_text(json.dumps(report, indent=2, default=str))

    # ── one-shot TEST with the validation-selected configuration ──
    key, vres, cand, g = best
    lname, algo = cand.split("/")
    model, pt = fit_predict(ds, feats, lname, algo, FINAL_TRAIN_END, TEST_START, last)
    dst = ds.copy()
    dst["score"] = np.nan
    dst.loc[pt.index, "score"] = pt.values
    dst["q"] = dst.groupby("date")["score"].rank(pct=True)
    tres = simulate(dst, "score", dst["q"] >= g["q"], TEST_START, last,
                    max_pos=g["max_pos"], stop_k=g["stop_k"], tgt_k=g["tgt_k"],
                    max_bars=g["max_bars"], regime_col="mkt_above_200" if g["regime"] else None)
    report["selected"] = {"key": key, "valid": vres, "test": tres}
    # benchmark: NIFTY over the same windows
    from app.services.market_data import market_data
    from research.simulate import buy_hold
    n = market_data.get_ohlcv("^NSEI", period="10y", interval="1d")["Close"]
    n.index = pd.DatetimeIndex(n.index).tz_localize(None).normalize() if pd.DatetimeIndex(n.index).tz is not None else pd.DatetimeIndex(n.index).normalize()
    report["nifty"] = {"valid": buy_hold(ds, VAL_START, VAL_END, n), "test": buy_hold(ds, TEST_START, last, n)}
    (OUT / "research_trade.json").write_text(json.dumps(report, indent=2, default=str))
    print("SELECTED", json.dumps(report["selected"], default=str))
    print("NIFTY", json.dumps(report["nifty"], default=str))


if __name__ == "__main__":
    if not os.getenv("ASTRA_OFFLINE_DATA_DIR"):
        sys.exit("Set ASTRA_OFFLINE_DATA_DIR")
    stages = sys.argv[1:] or ["build", "models"]
    if "build" in stages:
        stage_build()
    if "models" in stages:
        stage_models()
    if "trade" in stages:
        cands = os.getenv("CANDIDATES", "tb15/lgbm,xs20/lgbm,tb15/rf").split(",")
        stage_trade(cands)
