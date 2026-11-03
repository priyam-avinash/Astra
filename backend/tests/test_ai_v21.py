"""ASTRA.AI v2.1 rotation rules: top-N BUY, no fixed target, rank exit after the minimum hold."""
import numpy as np
import pandas as pd
import pytest

from app.services.ai_v2 import AIv2


@pytest.fixture()
def eng(monkeypatch):
    e = AIv2()
    assert e.available and e.rotation and len(e.models) == 2
    n = 40
    snap = pd.DataFrame({"score": np.linspace(1, 0, n), "mkt_above_200": 1.0, "ATR": 10.0, "Close": 1000.0,
                         "sector": "IT"}, index=[f"S{i}" for i in range(n)])
    snap["pct"] = snap["score"].rank(pct=True)
    snap["rank_pos"] = snap["score"].rank(ascending=False, method="first")
    monkeypatch.setattr(e, "snapshot", lambda force=False: snap)
    return e


def test_top_n_buy_and_no_target(eng):
    top = eng.cfg["top_n"]
    d = eng.decide(f"S{top - 1}")
    assert d["signal"] == "BUY" and d["tp_atr"] is None and d["sl_atr"] == eng.cfg["stop_k"]
    assert eng.decide(f"S{top}")["signal"] == "HOLD"
    assert "trading days" in d["exit_rule"]
    scan = eng.scan(top + 5)
    assert sum(r["signal"] == "BUY" for r in scan) == top and scan[0]["target"] is None


def test_rank_exit_only_after_min_hold_and_below_keep(eng):
    now = pd.Timestamp("2026-10-30")
    young = now - pd.Timedelta(days=20)
    old = now - pd.Timedelta(days=90)
    assert eng.rank_exit_due("S39", young, now) is None           # bottom-ranked but too young
    assert eng.rank_exit_due("S0", old, now) is None              # still top-ranked
    assert eng.rank_exit_due("S39", old, now).startswith("Rank Exit")
