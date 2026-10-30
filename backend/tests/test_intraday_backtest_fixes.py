"""Intraday backtest look-ahead fixes: trailing stop, entry timing, trend filter, ORB volume, VWAP, Combined."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import app.services.intraday_backtest as bt
from app.services.intraday_engine import _compute_daily_vwap


def bars(rows, day="2026-10-01", start="09:15"):
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="15min", tz="Asia/Kolkata")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1000.0)


def test_same_bar_trail_does_not_apply_until_next_bar():
    # BUY at 100, SL 99, TP 103. Bar 1 runs to 102.5 (83% of the way) then falls to 99.
    # Old code raised the trail inside the bar and booked a profit; correct is the original stop.
    fut = bars([[100, 102.5, 98.9, 99.2]], start="10:00")
    r = bt._simulate_trade("BUY", 100.0, 99.0, 103.0, fut, entry_slipped=True)
    assert r["outcome"] == "SL" and r["gross_pnl_pct"] < 0


def test_trail_applies_from_next_bar():
    fut = bars([[100, 102.5, 100.2, 102.0], [102, 102.1, 101.0, 101.2]], start="10:00")
    r = bt._simulate_trade("BUY", 100.0, 99.0, 103.0, fut, entry_slipped=True)
    assert r["outcome"] == "SL" and r["gross_pnl_pct"] > 1.0     # trailed to 101.5


def test_stop_beats_target_in_same_bar_and_sell_side():
    fut = bars([[100, 103.5, 98.5, 101]], start="10:00")
    assert bt._simulate_trade("BUY", 100.0, 99.0, 103.0, fut, entry_slipped=True)["outcome"] == "SL"
    assert bt._simulate_trade("SELL", 100.0, 101.0, 97.0, bars([[100, 101.5, 96.5, 99]], start="10:00"),
                              entry_slipped=True)["outcome"] == "SL"


def test_gap_through_stop_fills_at_open():
    fut = bars([[97, 97.5, 96.5, 97]], start="10:00")
    r = bt._simulate_trade("BUY", 100.0, 99.0, 103.0, fut, entry_slipped=True)
    assert r["outcome"] == "SL" and r["exit_price"] < 97.0


def test_square_off_at_1515_open():
    fut = bars([[100, 100.5, 99.6, 100.2], [101, 102, 100.9, 101.9]], start="15:00")
    r = bt._simulate_trade("BUY", 100.0, 99.0, 110.0, fut, entry_slipped=True)
    assert r["outcome"] == "SQ_OFF" and abs(r["exit_price"] - 101 * (1 - bt.SLIPPAGE_PCT)) < 0.02


def test_entry_is_next_bar_open_and_levels_reanchored():
    day = bars([[100, 100, 100, 100], [100, 101, 100, 101], [102, 102.2, 101.8, 102], [102, 102.1, 101.9, 102]])
    sig = {"signal": "BUY", "entry_price": 101.0, "sl": 100.0, "tp": 104.0, "ts": day.index[1]}
    r = bt._execute_signal(sig, day)
    assert r["entry_time"] == str(day.index[2])
    assert abs(r["entry_price"] - 102 * (1 + bt.SLIPPAGE_PCT)) < 0.01
    assert abs((r["entry_price"] - r["sl"]) - 1.0) < 0.01


def test_no_entry_when_next_bar_is_square_off():
    day = bars([[100, 100, 100, 100], [100, 100, 100, 100]], start="15:00")
    sig = {"signal": "BUY", "entry_price": 100.0, "sl": 99.0, "tp": 103.0, "ts": day.index[0]}
    assert bt._execute_signal(sig, day) is None


def test_trend_filter_sees_only_past_bars(monkeypatch):
    seen = []
    monkeypatch.setattr(bt, "_trend_aligned", lambda df, s: seen.append(df.index.max()) or True)
    monkeypatch.setattr(bt, "_compute_rsi_series", lambda s, *a: pd.Series(60.0, index=s.index))
    monkeypatch.setattr(bt, "_compute_daily_vwap", lambda df: pd.Series(0.0, index=df.index))
    close = np.r_[np.linspace(100, 95, 13), np.linspace(95, 103, 13)]
    d1 = bars([[c, c + 0.2, c - 0.2, c] for c in close])
    d2 = bars([[c, c + 0.2, c - 0.2, c] for c in close], day="2026-10-02")
    full = pd.concat([d1, d2])
    bt._get_ema_cross_signal_for_day(d1, full)
    assert seen and max(seen) <= d1.index.max()


def test_orb_volume_uses_bars_so_far():
    rows = [[100, 100.5, 99.5, 100]] * 4 + [[100, 101.5, 100, 101.2]] + [[101, 101.2, 100.8, 101]] * 10
    day = bars(rows)
    day["Volume"] = 1000.0
    day.iloc[4, day.columns.get_loc("Volume")] = 1200.0
    day.iloc[5:, day.columns.get_loc("Volume")] = 50_000.0          # big later volume must not matter
    s_full = bt._get_orb_signal_for_day(day, date(2026, 10, 1))
    s_trunc = bt._get_orb_signal_for_day(day.iloc[:5], date(2026, 10, 1))
    assert s_full is not None and s_full["signal"] == "BUY"
    assert s_trunc is not None and s_full["ts"] == s_trunc["ts"]


def test_daily_vwap_resets():
    d1 = bars([[100, 100, 100, 100]] * 3)
    d2 = bars([[200, 200, 200, 200]] * 3, day="2026-10-02")
    v = _compute_daily_vwap(pd.concat([d1, d2]))
    assert v.iloc[3] == pytest.approx(200.0)


def test_combined_takes_earliest_bar_and_skips_ties():
    t = pd.Timestamp("2026-10-01 10:00", tz="Asia/Kolkata")
    later = t + pd.Timedelta("1h")
    a = {"signal": "BUY", "entry_price": 100, "sl": 99, "tp": 103, "ts": t}
    b = {"signal": "SELL", "entry_price": 100, "sl": 101, "tp": 90, "ts": later}
    assert bt._pick_combined_signal([a, b, None]) is a
    c = {"signal": "SELL", "entry_price": 100, "sl": 101, "tp": 97, "ts": t}
    assert bt._pick_combined_signal([a, c]) is None


def test_universe_backtest_frozen_by_default(monkeypatch):
    from app.services import signal_weights as sw
    monkeypatch.setattr(sw.signal_weights, "update", lambda *a, **k: pytest.fail("learned in backtest"))
    monkeypatch.setattr(sw.signal_weights, "weight", lambda *a, **k: pytest.fail("weight filter used"))
    monkeypatch.setattr(bt, "run_intraday_backtest", lambda sym, days=30: {
        "strategies": {"ORB": {"trades": 1}}, "trade_log": [
            {"strategy": "ORB", "pnl_rs": 10.0, "pnl_pct": 0.01, "gross_pnl_pct": 0.1}],
        "combined_log": []})
    r = bt.run_universe_backtest(days=5, symbols=["AAA", "BBB"])
    assert r["in_sample"] is False and r["aggregate"]["ORB"]["trades"] == 2
