"""
ASTRA.AI v2 feature set — shared by research (training/backtests) and live scoring.

All features are known at the close of day t (no look-ahead):
  - the 20 v1 engine features (ai_engine._compute_features)
  - momentum: 5/20/60/120/250-day returns, distance to 52-week high/low
  - risk: 20/60-day volatility, ATR %
  - trend: EMA20/EMA50 and EMA50/EMA200 spreads, volume trend
  - relative strength vs NIFTY (20/60d) and vs own sector (60d, cross-sectional)
  - market regime: NIFTY vs its 200-day SMA, NIFTY 20d return and volatility
  - cross-sectional percentile ranks of momentum/RS/vol features per date
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRA_FEATURES = [
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_250",
    "dist_52w_high", "dist_52w_low", "vol_20", "vol_60", "atr_pct",
    "ema20_50", "ema50_200", "vol_trend",
    "rs_nifty_20", "rs_nifty_60", "rs_sector_60",
    "mkt_above_200", "mkt_ret_20", "mkt_vol_20",
]
RANK_FEATURES = ["ret_20", "ret_60", "ret_120", "ret_250", "rs_nifty_60", "rs_sector_60",
                 "dist_52w_high", "vol_20"]


def feature_columns() -> list:
    from app.services.ai_predictor import FEATURE_COLS
    return list(FEATURE_COLS) + EXTRA_FEATURES + [f"rk_{c}" for c in RANK_FEATURES]


def naive_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index().copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


def market_frame(nifty: pd.DataFrame) -> pd.DataFrame:
    nc = naive_daily(nifty)["Close"]
    return pd.DataFrame({
        "mkt_above_200": (nc > nc.rolling(200).mean()).astype(float),
        "mkt_ret_20": nc.pct_change(20) * 100,
        "mkt_vol_20": nc.pct_change().rolling(20).std() * np.sqrt(252) * 100,
        "n_ret_20": nc.pct_change(20) * 100,
        "n_ret_60": nc.pct_change(60) * 100,
    })


def symbol_features(raw: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol (non cross-sectional) features on the symbol's own daily bars."""
    from app.services.ai_predictor import ai_engine, FEATURE_COLS
    raw = naive_daily(raw)
    f = naive_daily(ai_engine._compute_features(raw))[FEATURE_COLS + ["ATR"]]
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
    return df


def add_cross_sectional(ds: pd.DataFrame) -> pd.DataFrame:
    """ds has columns date, symbol, sector + per-symbol features (long format)."""
    ds = ds.copy()
    ds["rs_sector_60"] = ds["ret_60"] - ds.groupby(["date", "sector"])["ret_60"].transform("median")
    for col in RANK_FEATURES:
        ds[f"rk_{col}"] = ds.groupby("date")[col].rank(pct=True)
    return ds


def build_features(data: dict, nifty: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    mkt = market_frame(nifty)
    parts = []
    for sym, raw in data.items():
        if raw is None or len(raw) < 300:
            continue
        df = symbol_features(raw, mkt)
        df["symbol"] = sym
        df["sector"] = sector_map.get(sym, "OTHER")
        parts.append(df)
    ds = pd.concat(parts)
    ds.index.name = "date"
    return add_cross_sectional(ds.reset_index())
