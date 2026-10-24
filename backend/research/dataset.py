"""
ASTRA research dataset: features + labels for every (date, symbol).

Features (all known at the close of day t — no look-ahead):
  - the 20 live engine features (ai_engine._compute_features)
  - momentum: 5/20/60/120/250-day returns, 52-week high/low distance
  - risk: 20/60-day volatility, ATR %
  - trend: EMA20/EMA50, EMA50/EMA200 spreads
  - relative strength vs NIFTY (20/60d) and vs own sector (60d)
  - market regime: NIFTY vs SMA200, NIFTY 20d return and volatility
  - cross-sectional percentile ranks of the momentum/RS features per date

Labels (entry at the NEXT day's open, as in live trading):
  - fwd_ret_h:  open[t+1] → close[t+h] return %, h = 10, 20
  - xs_ret_20:  fwd_ret_20 minus the universe median that day (relative)
  - tb_*:       triple barrier with the trade rules (stop k_s·ATR, target
                k_t·ATR, time limit T bars, gap-aware fills): win flag and
                realised % return
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TB_CONFIGS = {            # name: (stop_atr, target_atr, max_bars)
    "tb15_30_20": (1.5, 3.0, 20),
    "tb20_40_40": (2.0, 4.0, 40),
}

EXTRA_FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_250",
    "dist_52w_high", "dist_52w_low", "vol_20", "vol_60", "atr_pct",
    "ema20_50", "ema50_200", "vol_trend",
    "rs_nifty_20", "rs_nifty_60", "rs_sector_60",
    "mkt_above_200", "mkt_ret_20", "mkt_vol_20",
]
RANK_FEATURES = ["ret_20", "ret_60", "ret_120", "ret_250", "rs_nifty_60", "rs_sector_60",
                 "dist_52w_high", "vol_20"]


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index().copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


def triple_barrier(o, h, l, c, atr, stop_k, tgt_k, max_bars, cost_pct=0.35):
    """Vectorised-per-row loop. Entry at o[t+1]. Returns (win, ret%) arrays."""
    n = len(c)
    win = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    for t in range(n - 2):
        e = o[t + 1]
        a = atr[t]
        if not (np.isfinite(e) and np.isfinite(a)) or a <= 0 or e <= 0:
            continue
        stop, tgt = e - stop_k * a, e + tgt_k * a
        end = min(n - 1, t + max_bars)
        if t + max_bars > n - 1:
            continue                       # label not knowable yet
        exit_px, w = None, 0.0
        for j in range(t + 1, end + 1):
            if l[j] <= stop:               # stop first (conservative)
                exit_px = min(stop, o[j]) if j > t + 1 else stop
                break
            if h[j] >= tgt:
                exit_px, w = tgt, 1.0
                break
        if exit_px is None:
            exit_px = c[end]
            w = 1.0 if exit_px > e * (1 + cost_pct / 100) else 0.0
        win[t] = w
        ret[t] = (exit_px / e - 1) * 100 - cost_pct
    return win, ret


def build(data: dict, nifty: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    from app.services.ai_predictor import ai_engine, FEATURE_COLS
    nifty = _naive_index(nifty)
    nc = nifty["Close"]
    mkt = pd.DataFrame({
        "mkt_above_200": (nc > nc.rolling(200).mean()).astype(float),
        "mkt_ret_20": nc.pct_change(20) * 100,
        "mkt_vol_20": nc.pct_change().rolling(20).std() * np.sqrt(252) * 100,
        "n_ret_20": nc.pct_change(20) * 100,
        "n_ret_60": nc.pct_change(60) * 100,
    })
    parts = []
    for sym, raw in data.items():
        raw = _naive_index(raw)
        if len(raw) < 300:
            continue
        f = ai_engine._compute_features(raw)
        f = _naive_index(f)[FEATURE_COLS + ["ATR"]]
        df = raw[["Open", "High", "Low", "Close", "Volume"]].join(f, how="left")
        c = df["Close"]
        r = c.pct_change()
        for k in (5, 20, 60, 120, 250):
            df[f"ret_{k}"] = c.pct_change(k) * 100
        df["dist_52w_high"] = (c / c.rolling(250, min_periods=120).max() - 1) * 100
        df["dist_52w_low"] = (c / c.rolling(250, min_periods=120).min() - 1) * 100
        df["vol_20"] = r.rolling(20).std() * np.sqrt(252) * 100
        df["vol_60"] = r.rolling(60).std() * np.sqrt(252) * 100
        df["atr_pct"] = df["ATR"] / c * 100
        e20, e50, e200 = (c.ewm(span=s, adjust=False).mean() for s in (20, 50, 200))
        df["ema20_50"] = (e20 / e50 - 1) * 100
        df["ema50_200"] = (e50 / e200 - 1) * 100
        df["vol_trend"] = df["Volume"].rolling(20).mean() / df["Volume"].rolling(60).mean()
        m = mkt.reindex(df.index, method="pad")
        df[["mkt_above_200", "mkt_ret_20", "mkt_vol_20"]] = m[["mkt_above_200", "mkt_ret_20", "mkt_vol_20"]]
        df["rs_nifty_20"] = df["ret_20"] - m["n_ret_20"]
        df["rs_nifty_60"] = df["ret_60"] - m["n_ret_60"]
        # labels
        o, h, l, cl = (df[x].values for x in ("Open", "High", "Low", "Close"))
        nxt_open = df["Open"].shift(-1)
        for hzn in (10, 20):
            df[f"fwd_ret_{hzn}"] = (df["Close"].shift(-hzn) / nxt_open - 1) * 100
        for name, (ks, kt, mb) in TB_CONFIGS.items():
            w, rt = triple_barrier(o, h, l, cl, df["ATR"].values, ks, kt, mb)
            df[f"{name}_win"], df[f"{name}_ret"] = w, rt
        df["symbol"] = sym
        df["sector"] = sector_map.get(sym, "OTHER")
        parts.append(df)
    ds = pd.concat(parts)
    ds.index.name = "date"
    ds = ds.reset_index()
    # cross-sectional features
    ds["rs_sector_60"] = ds["ret_60"] - ds.groupby(["date", "sector"])["ret_60"].transform("median")
    ds["xs_ret_20"] = ds["fwd_ret_20"] - ds.groupby("date")["fwd_ret_20"].transform("median")
    for col in RANK_FEATURES:
        ds[f"rk_{col}"] = ds.groupby("date")[col].rank(pct=True)
    return ds


def feature_columns() -> list:
    from app.services.ai_predictor import FEATURE_COLS
    return list(FEATURE_COLS) + EXTRA_FEATURES + [f"rk_{c}" for c in RANK_FEATURES]
