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

from app.services.features_v2 import (  # noqa: E402  (shared with live scoring)
    EXTRA_FEATURES, RANK_FEATURES, feature_columns, naive_daily as _naive_index,
    market_frame, symbol_features, add_cross_sectional,
)


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
    mkt = market_frame(nifty)
    parts = []
    for sym, raw in data.items():
        if raw is None or len(raw) < 300:
            continue
        df = symbol_features(raw, mkt)
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
    ds = add_cross_sectional(ds.reset_index())
    ds["xs_ret_20"] = ds["fwd_ret_20"] - ds.groupby("date")["fwd_ret_20"].transform("median")
    return ds


