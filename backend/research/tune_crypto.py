"""
Tune the ASTRA.CRYPTO rule engine honestly (select 2018 → 2024-04, test 2024-07 →).

Variants searched (per risk class: majors BTC/ETH/BNB, altcoins SOL/XRP/ADA/AVAX/DOGE):
  mode     regime_switch (live: ADX≥25 momentum breakout, else mean reversion)
           momentum_only (breakouts only)   meanrev_only
  regime   none | btc200 (no new longs while BTC is below its 200-day average)
  exits    stop ATR × {1.5, 2, 3}, target ATR × {3, 4.5, 6}, time limit {20, 40} days

Writes scripts/out/tune_crypto.json with the selected config per class and its
untouched test result.
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

import research.simulate as sim   # noqa: E402

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD"]
SELECT = ("2018-06-01", "2024-04-30")
TEST = ("2024-07-01", "2026-12-31")


def build():
    from app.services.crypto_engine import crypto_engine, INTERNATIONAL_CRYPTO
    from app.services.market_data import market_data
    btc = market_data.get_ohlcv("BTC-USD", period="10y", interval="1d")["Close"]
    bi = pd.DatetimeIndex(btc.index)
    btc.index = (bi.tz_localize(None) if bi.tz is not None else bi).normalize()
    btc200 = (btc > btc.rolling(200).mean()).astype(float)
    parts = []
    for c in COINS:
        raw = market_data.get_ohlcv(c, period="10y", interval="1d")
        f = crypto_engine._compute_features(raw).copy()
        idx = pd.DatetimeIndex(f.index)
        f.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        mom, mr, conf_m, conf_r = [], [], [], []
        for _, last in f.iterrows():
            s1, c1, _ = crypto_engine._strategy_momentum_breakout(last)
            s2, c2, _ = crypto_engine._strategy_mean_reversion(last)
            mom.append(s1 == "BUY"); conf_m.append(c1)
            mr.append(s2 == "BUY"); conf_r.append(c2)
        f["buy_mom"], f["buy_mr"] = mom, mr
        f["conf_mom"], f["conf_mr"] = conf_m, conf_r
        f["vol_20"] = f["Close"].pct_change().rolling(20).std() * np.sqrt(365) * 100
        f["btc_above_200"] = btc200.reindex(f.index, method="pad").fillna(0).values
        f["symbol"] = c
        rc = INTERNATIONAL_CRYPTO.get(c, {}).get("risk_class", "altcoin")
        f["risk_class"] = "major" if rc == "major" else "altcoin"
        f.index.name = "date"
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)


def entries(d: pd.DataFrame, mode: str):
    adx = d["ADX"]
    if mode == "regime_switch":
        mask = np.where(adx >= 25, d["buy_mom"], d["buy_mr"])
        score = np.where(adx >= 25, d["conf_mom"], d["conf_mr"])
    elif mode == "momentum_only":
        mask, score = d["buy_mom"].values, d["conf_mom"].values
    else:
        mask, score = d["buy_mr"].values, d["conf_mr"].values
    return pd.Series(mask.astype(bool), index=d.index), pd.Series(score, index=d.index)


def main():
    sim.COST_RT, sim.SLIP = 0.002, 0.0005
    ds = build()
    out = {}
    for cls in ("major", "altcoin"):
        d = ds[ds.risk_class == cls].reset_index(drop=True)
        rows = []
        for mode, regime, sl, tp, mb in itertools.product(
                ("regime_switch", "momentum_only", "meanrev_only"), (None, "btc_above_200"),
                (1.5, 2.0, 3.0), (3.0, 4.5, 6.0), (20, 40)):
            mask, score = entries(d, mode)
            dd = d.assign(score=score.values)
            kw = dict(max_pos=d.symbol.nunique(), stop_k=sl, tgt_k=tp, max_bars=mb, regime_col=regime)
            sel = sim.simulate(dd, "score", mask, *SELECT, **kw)
            tst = sim.simulate(dd, "score", mask, *TEST, **kw)
            rows.append({"mode": mode, "regime": regime, "stop": sl, "target": tp, "max_bars": mb,
                         "select": sel, "test": tst})
        elig = [r for r in rows if r["select"].get("trades", 0) >= 60]
        best = max(elig or rows, key=lambda r: r["select"].get("sharpe") or -9)
        live = next(r for r in rows if r["mode"] == "regime_switch" and r["regime"] is None
                    and r["stop"] == (2.0 if cls == "major" else 3.0) and r["max_bars"] == 40
                    and r["target"] == (4.5 if cls == "major" else 6.0))
        out[cls] = {"best": best, "current_live_like": live,
                    "top": sorted(rows, key=lambda r: -(r["select"].get("sharpe") or -9))[:8]}
        print(cls, "BEST", json.dumps({k: best[k] for k in ("mode", "regime", "stop", "target", "max_bars")}),
              "select", json.dumps(best["select"]), "test", json.dumps(best["test"]), flush=True)
        print(cls, "LIVE-LIKE select", json.dumps(live["select"]), "test", json.dumps(live["test"]), flush=True)
    btc = ds[ds.symbol == "BTC-USD"].set_index("date")["Close"]
    out["btc_buy_hold"] = {"select": sim.buy_hold(ds, *SELECT, btc), "test": sim.buy_hold(ds, *TEST, btc)}
    (HERE.parent / "scripts" / "out" / "tune_crypto.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
