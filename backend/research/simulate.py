"""
Score-ranked portfolio simulator (daily bars, long-only, NSE delivery costs).

Each day after the close: rank symbols by `score`; take the best ones that
pass `entry_mask` into free slots (max_pos). Entry at next day's open + slippage.
Exits: stop (stop_k·ATR, gap-aware), target (tgt_k·ATR), time limit
(max_bars). Position size = equity / max_pos at entry (optionally scaled
by inverse volatility). Equity marked to market at each close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_RT = 0.0025        # brokerage + STT + exchange + GST + stamp (delivery)
SLIP = 0.0005           # per side
CAPITAL = 1_000_000.0


def simulate(ds: pd.DataFrame, score_col: str, entry_mask: pd.Series, start, end,
             max_pos=10, stop_k=1.5, tgt_k=3.0, max_bars=20, vol_scale=False,
             regime_col: str | None = None) -> dict:
    d = ds[(ds["date"] >= pd.Timestamp(start)) & (ds["date"] <= pd.Timestamp(end))].copy()
    d["_ok"] = entry_mask.reindex(d.index).fillna(False).values
    if regime_col:
        d["_ok"] &= d[regime_col].fillna(0).values > 0.5
    px = {s: g.set_index("date")[["Open", "High", "Low", "Close", "ATR", "vol_20"]]
          for s, g in ds.groupby("symbol")}
    dates = sorted(d["date"].unique())
    cands_by_day = {dt: g[g["_ok"]].sort_values(score_col, ascending=False)
                    for dt, g in d.groupby("date")}

    cash, positions, curve, trades = CAPITAL, {}, [], []
    pending = []                                  # entries to fill at next open
    for i, dt in enumerate(dates):
        # 1) fill yesterday's entries at today's open
        for sym, atr, vol in pending:
            p = px[sym]
            if dt not in p.index or sym in positions:
                continue
            o = float(p.at[dt, "Open"])
            equity = cash + sum(_mark(px, s, pos, dt, "Open") for s, pos in positions.items())
            alloc = equity / max_pos
            if vol_scale and vol and np.isfinite(vol) and vol > 0:
                alloc *= float(np.clip(25.0 / vol, 0.5, 1.5))
            alloc = min(alloc, cash)
            if alloc < 5000:
                continue
            fill = o * (1 + SLIP)
            positions[sym] = {"entry": fill, "qty": alloc / fill, "stop": fill - stop_k * atr,
                              "tgt": fill + tgt_k * atr, "bars": 0, "entry_date": dt, "alloc": alloc}
            cash -= alloc
        pending = []
        # 2) exits on today's bar
        for sym in list(positions):
            pos, p = positions[sym], px[sym]
            if dt not in p.index:
                continue
            o, h, l, c = (float(p.at[dt, k]) for k in ("Open", "High", "Low", "Close"))
            pos["bars"] += 1
            exit_px = None
            if l <= pos["stop"]:
                exit_px = min(pos["stop"], o)
            elif h >= pos["tgt"]:
                exit_px = max(pos["tgt"], o) if pos["bars"] > 1 else pos["tgt"]
            elif pos["bars"] >= max_bars:
                exit_px = c
            if exit_px is not None:
                proceeds = pos["qty"] * exit_px * (1 - SLIP) * (1 - COST_RT)
                cash += proceeds
                trades.append((proceeds / pos["alloc"] - 1) * 100)
                del positions[sym]
        # 3) queue entries for tomorrow's open
        free = max_pos - len(positions)
        if free > 0 and i < len(dates) - 1:
            for _, row in cands_by_day.get(dt, pd.DataFrame()).iterrows():
                if free <= 0:
                    break
                if row["symbol"] in positions or not np.isfinite(row["ATR"]):
                    continue
                pending.append((row["symbol"], float(row["ATR"]), float(row.get("vol_20", np.nan))))
                free -= 1
        equity = cash + sum(_mark(px, s, pos, dt, "Close") for s, pos in positions.items())
        curve.append((dt, equity, len(positions)))

    eq = pd.Series([e for _, e, _ in curve], index=pd.DatetimeIndex([t for t, _, _ in curve]))
    expo = pd.Series([n for _, _, n in curve], index=eq.index) / max_pos
    return stats(eq, expo, trades)


def _mark(px, sym, pos, dt, field):
    p = px[sym]
    if dt in p.index:
        return pos["qty"] * float(p.at[dt, field])
    prev = p.loc[:dt]
    return pos["qty"] * float(prev["Close"].iloc[-1]) if len(prev) else pos["alloc"]


def stats(eq: pd.Series, expo: pd.Series, trades: list) -> dict:
    if eq.empty:
        return {}
    years, prev = {}, CAPITAL
    for y, g in eq.groupby(eq.index.year):
        years[int(y)] = round((float(g.iloc[-1]) / prev - 1) * 100, 1)
        prev = float(g.iloc[-1])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / CAPITAL) ** (1 / yrs) - 1 if yrs > 0.25 else np.nan
    r = eq.pct_change().dropna()
    t = np.array(trades) if trades else np.array([0.0])
    return {
        "cagr_pct": round(float(cagr) * 100, 1) if np.isfinite(cagr) else None,
        "total_return_pct": round((float(eq.iloc[-1]) / CAPITAL - 1) * 100, 1),
        "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
        "exposure_pct": round(float(expo.mean()) * 100, 1),
        "trades": len(trades),
        "win_rate_pct": round(float((t > 0).mean()) * 100, 1),
        "avg_trade_pct": round(float(t.mean()), 2),
        "yearly_pct": years,
    }


def buy_hold(ds: pd.DataFrame, start, end, symbol_close: pd.Series) -> dict:
    s = symbol_close[(symbol_close.index >= pd.Timestamp(start)) & (symbol_close.index <= pd.Timestamp(end))]
    eq = s / s.iloc[0] * CAPITAL
    return stats(eq, pd.Series(1.0, index=eq.index), [])
