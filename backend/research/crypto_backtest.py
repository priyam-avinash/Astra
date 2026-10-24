"""
ASTRA.CRYPTO rule engine backtest (daily, long-only, live decision functions).

Per coin and day: the engine's regime switch (ADX ≥ 25 momentum breakout,
ADX < 20 mean reversion, otherwise transitional mean reversion) decides BUY.
Entries at next day's open; exits with the coin class's ATR stop/target
(crypto_engine.RISK_CONFIG) or a 30-day time limit; costs 0.2% round trip +
0.05% slippage per side. Fear & Greed / BTC-dominance filters are not
applied (no point-in-time history available offline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

import research.simulate as sim   # noqa: E402

COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD"]


def build():
    from app.services.crypto_engine import crypto_engine, INTERNATIONAL_CRYPTO, RISK_CONFIG
    from app.services.market_data import market_data
    parts = []
    for c in COINS:
        raw = market_data.get_ohlcv(c, period="10y", interval="1d")
        f = crypto_engine._compute_features(raw).copy()
        idx = pd.DatetimeIndex(f.index)
        f.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        sig, conf = [], []
        for _, last in f.iterrows():
            adx = float(last["ADX"])
            if adx >= 25:
                s, sc, _ = crypto_engine._strategy_momentum_breakout(last)
            else:
                s, sc, _ = crypto_engine._strategy_mean_reversion(last)
                if adx >= 20:
                    sc *= 0.8
            sig.append(s)
            conf.append(sc)
        f["signal"], f["score"] = sig, conf
        f["vol_20"] = f["Close"].pct_change().rolling(20).std() * np.sqrt(365) * 100
        f["symbol"] = c
        f["risk_class"] = INTERNATIONAL_CRYPTO.get(c, {}).get("risk_class", "altcoin")
        f.index.name = "date"
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True), RISK_CONFIG


def main():
    sim.COST_RT, sim.SLIP = 0.002, 0.0005
    ds, risk = build()
    out = {}
    btc = ds[ds.symbol == "BTC-USD"].set_index("date")["Close"]
    for start, end, tag in (("2018-01-01", "2026-12-31", "full"), ("2024-07-01", "2026-12-31", "test_window")):
        res = {}
        for cls in ("major", "altcoin"):
            sub = ds[ds.risk_class == cls].reset_index(drop=True)
            if sub.empty:
                continue
            cfg = risk[cls]
            res[cls] = sim.simulate(sub, "score", sub["signal"] == "BUY", start, end,
                                    max_pos=max(2, sub.symbol.nunique()), stop_k=cfg["sl_mult"],
                                    tgt_k=cfg["tp_mult"], max_bars=30)
        res["BTC_buy_hold"] = sim.buy_hold(ds, start, end, btc)
        out[tag] = res
        print(tag, json.dumps(res, default=str))
    (sim.__file__ and Path(HERE.parent / "scripts" / "out" / "crypto_backtest.json")).write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
