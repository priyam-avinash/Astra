# Dynamic Universe + Layer-1 Signal Weighting Implementation Plan

> **For executors:** Inline execution via superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Stop hardcoding 5 stocks. Each morning, dynamically rank the NIFTY-200 universe per-strategy by tradability score. Track rolling per-(engine, symbol) win rates and auto-suppress losing combinations.

**Architecture:** Two new services + one CLI script. (1) `intraday_universe.py` scores stocks for ORB / EMA_Cross / Momentum suitability using ADX, ATR%, volume, and SMA structure. (2) `signal_weights.py` maintains a Bayesian-prior rolling WR per (engine, symbol) persisted to JSON. (3) Backtest is rewired to scan universe daily and apply weights before opening trades.

**Tech Stack:** Python 3.9, pandas, numpy, yfinance (via existing `_fetch_intraday_extended`), JSON for state persistence (no DB needed).

**Token-budget note for executor:** This is research/trading code, not a production service. We use *verification-driven development* (write → run → verify output), not formal TDD. Skip unit tests. One commit per task.

---

## File Structure

| Path | Purpose | Action |
|---|---|---|
| `backend/app/services/intraday_universe.py` | Score + rank NIFTY-200 stocks daily per strategy | Create |
| `backend/app/services/signal_weights.py` | Bayesian rolling-WR weights, persisted | Create |
| `backend/app/services/intraday_backtest.py` | Rewire to use dynamic universe + weights | Modify |
| `backend/run_universe_backtest.py` | CLI to run full backtest, print summary | Create |
| `backend/data/signal_weights.json` | Persisted weights file | Auto-created at runtime |

The `universe_scanner.py` is daily-equity, untouched. The two layers are independent — universe selects *which* stocks; weights filter *which combinations* fire.

---

## Task 1: Intraday Universe Scorer

**Files:**
- Create: `backend/app/services/intraday_universe.py`

- [ ] **Step 1: Create the file**

```python
"""
ASTRA Intraday Universe Scorer
================================
Each morning, rank NSE stocks by their suitability for each intraday strategy.

Scoring rationale (research-driven, not arbitrary):
  - ORB        → wants trending + volatile stocks (ADX > 22, ATR% in 1-3% band)
  - EMA_Cross  → wants directional trend (SMA50 slope > 0, price above SMA50)
  - Momentum   → wants high-beta volatile stocks (ATR% > 1.5%, recent breakouts)

Daily output: ranked list per strategy, top-N (default 20) used for the day.
Cached for the trading day (TTL 6 hours).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Universe — NIFTY-200 subset that has liquid intraday markets
# (Drawn from the larger NIFTY 500 list but trimmed to >₹50cr avg daily turnover)
INTRADAY_UNIVERSE = [
    # NIFTY 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "KOTAKBANK", "BHARTIARTL", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ADANIENT", "POWERGRID", "TECHM", "INDUSINDBK", "DIVISLAB", "JSWSTEEL",
    "BAJAJFINSV", "COALINDIA", "HCLTECH", "ONGC", "NTPC", "TATAMOTORS",
    "TATASTEEL", "CIPLA", "BRITANNIA", "DRREDDY", "BPCL", "HEROMOTOCO",
    "GRASIM", "EICHERMOT", "TATACONSUM", "SBILIFE", "HDFCLIFE",
    "APOLLOHOSP", "HINDALCO", "ADANIPORTS", "BAJAJ-AUTO", "ITC",
    # Top intraday movers from NIFTY NEXT 50
    "SIEMENS", "HAVELLS", "VOLTAS", "PIDILITIND", "TATAPOWER", "ADANIGREEN",
    "DMART", "INDIGO", "DLF", "GODREJCP", "CHOLAFIN", "TRENT",
    "AUROPHARMA", "LUPIN", "BIOCON", "BANDHANBNK", "IDFCFIRSTB", "PNB",
    "GAIL", "IOC", "RECLTD", "PFC", "IRFC", "RVNL",
    # High-beta midcaps
    "DIXON", "POLYCAB", "TIINDIA", "KPITTECH", "PERSISTENT", "LTIM",
    "MOTHERSON", "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR",
    "INDIANB", "CANBK", "BANKBARODA", "UNIONBANK",
]

# In-memory cache: date -> {strategy: ranked_list}
_CACHE: dict = {}
_CACHE_DATE: Optional[str] = None


def _score_single(symbol: str) -> Optional[dict]:
    """
    Fetch 30 days of daily bars, compute scoring metrics.
    Returns a dict with scores per strategy, or None on failure.
    """
    try:
        from app.services.yahoo_finance import yahoo_service
        df = yahoo_service.get_ohlcv(f"{symbol}.NS", period="3mo", interval="1d")
        if df is None or df.empty or len(df) < 30:
            return None

        df = df.tail(30).copy()
        df["TR"] = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_14 = df["TR"].rolling(14).mean().iloc[-1]
        close  = df["Close"].iloc[-1]
        atr_pct = (atr_14 / close * 100) if close > 0 else 0

        # SMA50 proxy on 30-day data using 20-period MA (we only have 30 daily bars)
        sma20 = df["Close"].rolling(20).mean()
        sma_slope = (sma20.iloc[-1] - sma20.iloc[-10]) / sma20.iloc[-10] * 100 if sma20.iloc[-10] > 0 else 0
        above_sma = close > sma20.iloc[-1]

        # ADX (14)
        up   = df["High"].diff()
        down = -df["Low"].diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr14 = df["TR"].rolling(14).sum()
        plus_di14  = 100 * pd.Series(plus_dm,  index=df.index).rolling(14).sum() / tr14
        minus_di14 = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
        dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14 + 1e-9)
        adx = dx.rolling(14).mean().iloc[-1] if len(dx) >= 14 else 15.0

        # Liquidity: 20-day avg turnover in ₹ crore
        avg_turnover_cr = (df["Close"] * df["Volume"]).tail(20).mean() / 1e7

        # 20-day return (momentum)
        ret_20d = (close / df["Close"].iloc[-20] - 1) * 100 if df["Close"].iloc[-20] > 0 else 0

        # ── Per-strategy scores (0–100) ──────────────────────────────────────
        # ORB: needs ADX > 22 AND ATR% in 1-3% AND liquid
        orb_score = 0.0
        if adx >= 22 and 1.0 <= atr_pct <= 3.5 and avg_turnover_cr >= 50:
            orb_score = min(100, adx * 1.5 + atr_pct * 8)

        # EMA_Cross: directional trend, price above SMA, slope positive
        ema_score = 0.0
        if above_sma and sma_slope > 1.0 and avg_turnover_cr >= 50:
            ema_score = min(100, sma_slope * 4 + adx)

        # Momentum: high recent return + volatility
        mom_score = 0.0
        if atr_pct >= 1.5 and abs(ret_20d) >= 3 and avg_turnover_cr >= 50:
            mom_score = min(100, abs(ret_20d) * 3 + atr_pct * 5)

        return {
            "symbol":    symbol,
            "adx":       round(float(adx), 1),
            "atr_pct":   round(float(atr_pct), 2),
            "sma_slope": round(float(sma_slope), 2),
            "ret_20d":   round(float(ret_20d), 2),
            "turnover":  round(float(avg_turnover_cr), 1),
            "scores":    {
                "ORB":       round(orb_score, 1),
                "EMA_Cross": round(ema_score, 1),
                "Momentum":  round(mom_score, 1),
            },
        }
    except Exception as e:
        logger.debug(f"[universe] {symbol} scoring failed: {e}")
        return None


def rank_universe(top_n: int = 20, max_workers: int = 6) -> dict:
    """
    Score the full intraday universe and return ranked lists per strategy.
    Cached for the current trading day.

    Returns:
        {
            "ORB":       ["RELIANCE", "INFY", ...],     # top_n by ORB score
            "EMA_Cross": [...],
            "Momentum":  [...],
            "scores":    {symbol: full_metrics_dict},   # for inspection
            "scored_at": ISO timestamp,
        }
    """
    global _CACHE, _CACHE_DATE
    today = datetime.now().strftime("%Y-%m-%d")
    if _CACHE_DATE == today and _CACHE:
        return _CACHE

    logger.info(f"[universe] Scoring {len(INTRADAY_UNIVERSE)} symbols...")
    start = time.time()

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_single, s): s for s in INTRADAY_UNIVERSE}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results[r["symbol"]] = r

    ranked: dict = {"scores": results, "scored_at": datetime.now().isoformat()}
    for strat in ("ORB", "EMA_Cross", "Momentum"):
        sorted_syms = sorted(
            results.values(),
            key=lambda x: x["scores"][strat],
            reverse=True,
        )
        # Only include stocks with non-zero score for that strategy
        ranked[strat] = [r["symbol"] for r in sorted_syms if r["scores"][strat] > 0][:top_n]

    _CACHE      = ranked
    _CACHE_DATE = today
    duration = round(time.time() - start, 1)
    logger.info(
        f"[universe] Scored {len(results)}/{len(INTRADAY_UNIVERSE)} in {duration}s — "
        f"ORB:{len(ranked['ORB'])} EMA:{len(ranked['EMA_Cross'])} MOM:{len(ranked['Momentum'])}"
    )
    return ranked
```

- [ ] **Step 2: Verify it imports and scores at least one stock**

Run from `backend/`:
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
from app.services.intraday_universe import _score_single
r = _score_single('RELIANCE')
print(r)
"
```
Expected: prints dict with `adx`, `atr_pct`, `scores: {ORB: ..., EMA_Cross: ..., Momentum: ...}`. Numbers should be reasonable (ADX 15-40, ATR% 0.8-3, scores 0-100).

- [ ] **Step 3: Run full universe ranking**

Run:
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
from app.services.intraday_universe import rank_universe
r = rank_universe(top_n=10)
print('ORB top 10:      ', r['ORB'])
print('EMA_Cross top 10:', r['EMA_Cross'])
print('Momentum top 10: ', r['Momentum'])
"
```
Expected: 3 lists with up to 10 symbols each. May be empty list for ORB/Momentum if market is in a quiet phase — that is correct behaviour.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/intraday_universe.py
git commit -m "feat: dynamic intraday universe scorer per strategy"
```

---

## Task 2: Signal Weights (Layer 1 — online Bayesian WR)

**Files:**
- Create: `backend/app/services/signal_weights.py`

- [ ] **Step 1: Create the file**

```python
"""
Layer 1 Self-Learning — Online Signal Weighting
================================================
Tracks rolling P&L per (engine, symbol) combination and produces a weight in [0, 1.5]
that scales position size / suppresses the trade entirely.

Bayesian update with weak priors so the system is usable from trade 1, not trade 100.
Persisted to a JSON file — no DB needed.

Usage:
    from app.services.signal_weights import signal_weights
    w = signal_weights.weight("ORB", "RELIANCE")   # → 1.2 (good combo)
    if w < 0.3: skip_trade()
    ...
    signal_weights.update("ORB", "RELIANCE", pnl_rs=420)   # call after exit
"""

import json
import logging
import math
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).parent.parent / "data" / "signal_weights.json"
_WINDOW     = 50          # rolling window size per (engine, symbol)
_PRIOR_WINS = 5.0         # Beta prior — equivalent to 5 fake wins
_PRIOR_LOSS = 5.0         # equivalent to 5 fake losses → start at WR 0.5
_MIN_WEIGHT = 0.0         # 0 means "do not take this trade"
_MAX_WEIGHT = 1.5         # cap upside leverage at 1.5×


class SignalWeights:
    """Bayesian rolling WR tracker per (engine, symbol)."""

    def __init__(self):
        self._lock   = Lock()
        self._state: dict = {}    # key "ENGINE|SYMBOL" → {pnls: [...], updated: iso}
        self._load()

    # ── Public ─────────────────────────────────────────────────────────────

    def update(self, engine: str, symbol: str, pnl_rs: float) -> None:
        """Record a closed trade. Called at trade exit."""
        key = f"{engine}|{symbol}"
        with self._lock:
            entry = self._state.setdefault(key, {"pnls": [], "updated": ""})
            pnls: list = entry["pnls"]
            pnls.append(round(float(pnl_rs), 2))
            if len(pnls) > _WINDOW:
                entry["pnls"] = pnls[-_WINDOW:]
            entry["updated"] = datetime.now().isoformat()
            self._save()

    def weight(self, engine: str, symbol: str) -> float:
        """
        Return a multiplier in [_MIN_WEIGHT, _MAX_WEIGHT].
        > 1.0 → expand size; 1.0 → neutral; < 1.0 → shrink; 0 → skip.
        """
        key  = f"{engine}|{symbol}"
        pnls = self._state.get(key, {}).get("pnls", [])
        wins = sum(1 for p in pnls if p > 0)
        loss = sum(1 for p in pnls if p <= 0)

        # Bayesian posterior WR with Beta prior
        post_wr = (wins + _PRIOR_WINS) / (wins + loss + _PRIOR_WINS + _PRIOR_LOSS)

        # For 1:3 R:R, break-even WR is 0.25. Scale weight around 0.35.
        # weight(0.25) = 0.0   weight(0.35) = 1.0   weight(0.50) = 1.5
        if post_wr <= 0.25:
            return _MIN_WEIGHT
        if post_wr >= 0.55:
            return _MAX_WEIGHT
        return round(1.0 + (post_wr - 0.35) * 5.0, 2)

    def stats(self, engine: Optional[str] = None) -> dict:
        """Inspection: return per-key {n, wins, wr, weight}."""
        out = {}
        for key, entry in self._state.items():
            eng, sym = key.split("|", 1)
            if engine and eng != engine:
                continue
            pnls = entry["pnls"]
            wins = sum(1 for p in pnls if p > 0)
            out[key] = {
                "engine":  eng,
                "symbol":  sym,
                "n":       len(pnls),
                "wins":    wins,
                "wr":      round(wins / len(pnls), 3) if pnls else None,
                "weight":  self.weight(eng, sym),
                "pnl_sum": round(sum(pnls), 2),
            }
        return out

    def reset(self, engine: Optional[str] = None, symbol: Optional[str] = None) -> int:
        """Clear state. Returns count of keys removed."""
        with self._lock:
            if engine is None and symbol is None:
                n = len(self._state)
                self._state = {}
            else:
                keys = [k for k in self._state
                        if (engine is None or k.split("|")[0] == engine)
                        and (symbol is None or k.split("|")[1] == symbol)]
                for k in keys:
                    del self._state[k]
                n = len(keys)
            self._save()
        return n

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if _STATE_PATH.exists():
            try:
                with open(_STATE_PATH) as f:
                    self._state = json.load(f)
                logger.info(f"[signal_weights] Loaded {len(self._state)} keys from {_STATE_PATH.name}")
            except Exception as e:
                logger.warning(f"[signal_weights] Load failed, starting fresh: {e}")
                self._state = {}

    def _save(self) -> None:
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_PATH, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"[signal_weights] Save failed: {e}")


signal_weights = SignalWeights()
```

- [ ] **Step 2: Verify the math**

Run:
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
from app.services.signal_weights import signal_weights

# Fresh slate
signal_weights.reset()

# Untouched combo → should return neutral-ish (prior is 0.5 WR → weight 1.0+)
print('neutral:', signal_weights.weight('ORB', 'TESTSTOCK'))   # expect ~1.5 (no data, prior is 0.5)

# Simulate 10 wins: should boost weight
for _ in range(10):
    signal_weights.update('ORB', 'GOODSTOCK', pnl_rs=300)
print('10 wins:', signal_weights.weight('ORB', 'GOODSTOCK'))   # expect 1.5

# Simulate 10 losses: should suppress
for _ in range(10):
    signal_weights.update('ORB', 'BADSTOCK', pnl_rs=-300)
print('10 loss:', signal_weights.weight('ORB', 'BADSTOCK'))    # expect 0.0

# Mixed: 3W 7L → WR=0.3 → weight should be < 1.0
for _ in range(3):
    signal_weights.update('ORB', 'MIXED', pnl_rs=300)
for _ in range(7):
    signal_weights.update('ORB', 'MIXED', pnl_rs=-200)
print('3W7L:  ', signal_weights.weight('ORB', 'MIXED'))        # expect 0.5-0.9 range

signal_weights.reset()
"
```
Expected: `neutral` ≈ 1.5 (or whatever the 0.5 prior maps to), `10 wins` = 1.5, `10 loss` = 0.0, `3W7L` in (0, 1).

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/signal_weights.py
git commit -m "feat: Layer-1 online signal weighting (Bayesian rolling WR per engine,symbol)"
```

---

## Task 3: Wire universe + weights into intraday backtest

**Files:**
- Modify: `backend/app/services/intraday_backtest.py`

- [ ] **Step 1: Add the new universe-driven function**

Open `backend/app/services/intraday_backtest.py`. Add this function at the end of the file (after `run_intraday_backtest`):

```python
def run_universe_backtest(days: int = 30, top_n_per_strategy: int = 15,
                          use_weights: bool = True) -> dict:
    """
    Universe-driven backtest:
      1. Rank universe → top-N stocks per strategy
      2. For each strategy, backtest only its top-N stocks
      3. Apply signal_weights filter (skip combos with weight < 0.3)
      4. Update signal_weights after every closed trade
      5. Return aggregate stats per strategy + per-symbol breakdown

    `use_weights=False` disables the Layer-1 filter (useful for clean baseline).
    """
    from app.services.intraday_universe import rank_universe
    from app.services.signal_weights import signal_weights

    logger.info(f"[universe-backtest] days={days} top_n={top_n_per_strategy} weights={use_weights}")
    ranked = rank_universe(top_n=top_n_per_strategy)

    strategy_universes = {
        "ORB":       ranked.get("ORB",       []),
        "EMA_Cross": ranked.get("EMA_Cross", []),
        "Momentum":  ranked.get("Momentum",  []),
    }

    per_strategy_trades: dict = {s: [] for s in strategy_universes}
    per_symbol_results: dict  = {}
    skipped_by_weight = 0

    for strategy, symbols in strategy_universes.items():
        logger.info(f"[universe-backtest] {strategy}: {len(symbols)} symbols")
        for sym in symbols:
            # Layer-1 pre-filter
            if use_weights and signal_weights.weight(strategy, sym) < 0.3:
                skipped_by_weight += 1
                continue

            # Run the existing single-symbol backtest
            res = run_intraday_backtest(sym, days=days)
            if "error" in res:
                continue
            stats = res["strategies"].get(strategy, {})
            trade_log = [t for t in res["trade_log"] if t["strategy"] == strategy]

            per_symbol_results.setdefault(sym, {})[strategy] = {
                "trades":     stats.get("trades", 0),
                "win_rate":   stats.get("win_rate_pct", 0),
                "pnl_rs":     stats.get("total_pnl_rs", 0),
            }

            # Aggregate trades + update signal weights
            for t in trade_log:
                per_strategy_trades[strategy].append({**t, "symbol": sym})
                if use_weights:
                    signal_weights.update(strategy, sym, t["pnl_rs"])

    # Build aggregate stats per strategy
    aggregate: dict = {}
    for strat, trades in per_strategy_trades.items():
        if not trades:
            aggregate[strat] = {"trades": 0, "win_rate_pct": 0, "total_pnl_rs": 0}
            continue
        wins = [t for t in trades if t["pnl_rs"] > 0]
        aggregate[strat] = {
            "trades":         len(trades),
            "wins":           len(wins),
            "losses":         len(trades) - len(wins),
            "win_rate_pct":   round(len(wins) / len(trades) * 100, 1),
            "total_pnl_rs":   round(sum(t["pnl_rs"] for t in trades), 2),
            "avg_pnl_rs":     round(sum(t["pnl_rs"] for t in trades) / len(trades), 2),
            "symbols_traded": len({t["symbol"] for t in trades}),
        }

    return {
        "period_days":         days,
        "top_n_per_strategy":  top_n_per_strategy,
        "use_weights":         use_weights,
        "skipped_by_weight":   skipped_by_weight,
        "universe_used":       strategy_universes,
        "aggregate":           aggregate,
        "per_symbol":          per_symbol_results,
        "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
```

- [ ] **Step 2: Sanity-check the import path**

Run:
```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, '.')
from app.services.intraday_backtest import run_universe_backtest
print(run_universe_backtest.__doc__[:80])
"
```
Expected: prints the docstring's first line.

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/intraday_backtest.py
git commit -m "feat: universe-driven intraday backtest with Layer-1 weight filter"
```

---

## Task 4: CLI runner + verification

**Files:**
- Create: `backend/run_universe_backtest.py`

- [ ] **Step 1: Create the CLI**

```python
"""
CLI: Universe-driven intraday backtest.

Usage:
    python run_universe_backtest.py              # 30 days, top 15 per strategy, weights ON
    python run_universe_backtest.py --no-weights # baseline (no Layer-1 filter)
    python run_universe_backtest.py --top 10 --days 60
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.services.intraday_backtest import run_universe_backtest
from app.services.signal_weights import signal_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",        type=int, default=30)
    ap.add_argument("--top",         type=int, default=15)
    ap.add_argument("--no-weights",  action="store_true")
    ap.add_argument("--reset-weights", action="store_true")
    args = ap.parse_args()

    if args.reset_weights:
        n = signal_weights.reset()
        print(f"Reset {n} weight entries.\n")

    result = run_universe_backtest(
        days=args.days,
        top_n_per_strategy=args.top,
        use_weights=not args.no_weights,
    )

    print("\n" + "=" * 78)
    print(f"  ASTRA UNIVERSE BACKTEST  —  {args.days} days, top-{args.top} per strategy")
    print(f"  Layer-1 weights: {'ON ' if not args.no_weights else 'OFF'}     "
          f"Skipped by weight: {result['skipped_by_weight']}")
    print("=" * 78)

    print("\n  AGGREGATE PER STRATEGY")
    print("  " + "-" * 70)
    print(f"  {'Strategy':<12} {'Symbols':>8} {'Trades':>8} {'WR%':>7} {'TotalP&L':>12} {'AvgP&L':>10}")
    for strat, s in result["aggregate"].items():
        st  = s.get("symbols_traded", 0)
        tr  = s.get("trades", 0)
        wr  = s.get("win_rate_pct", 0)
        pnl = s.get("total_pnl_rs", 0)
        ap  = s.get("avg_pnl_rs", 0)
        print(f"  {strat:<12} {st:>8} {tr:>8} {wr:>7.1f} {pnl:>11.0f} {ap:>9.0f}")

    print("\n  UNIVERSE USED (top per strategy)")
    for strat, syms in result["universe_used"].items():
        print(f"  {strat:<12}: {', '.join(syms[:12])}{' ...' if len(syms) > 12 else ''}")

    # Best/worst per strategy
    print("\n  TOP 5 PROFITABLE (strategy, symbol)")
    flat = []
    for sym, by_strat in result["per_symbol"].items():
        for strat, s in by_strat.items():
            flat.append((s["pnl_rs"], sym, strat, s["trades"], s["win_rate"]))
    flat.sort(reverse=True)
    for pnl, sym, strat, tr, wr in flat[:5]:
        print(f"   +Rs{pnl:>7.0f}  {strat:<10} {sym:<12}  ({tr} trades, WR {wr}%)")
    print("\n  BOTTOM 5 LOSING")
    for pnl, sym, strat, tr, wr in flat[-5:]:
        print(f"   Rs{pnl:>8.0f}  {strat:<10} {sym:<12}  ({tr} trades, WR {wr}%)")

    # Layer-1 stats
    if not args.no_weights:
        stats = signal_weights.stats()
        if stats:
            print(f"\n  LAYER-1 WEIGHTS — {len(stats)} (engine, symbol) keys tracked")
            top_w = sorted(stats.values(), key=lambda x: x["weight"], reverse=True)[:5]
            bot_w = sorted(stats.values(), key=lambda x: x["weight"])[:5]
            print("    Top-weighted (most-favoured):")
            for s in top_w:
                print(f"     {s['engine']:<10} {s['symbol']:<12}  n={s['n']:>2}  WR={s['wr']}  w={s['weight']}")
            print("    Bottom-weighted (suppressed):")
            for s in bot_w:
                print(f"     {s['engine']:<10} {s['symbol']:<12}  n={s['n']:>2}  WR={s['wr']}  w={s['weight']}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: First run — baseline (weights off, no prior data)**

Run from `backend/`:
```bash
source .venv/bin/activate && python run_universe_backtest.py --days 30 --top 10 --no-weights --reset-weights
```

Expected: prints aggregate table, universe lists, top/bottom symbols. Total trades should be in the hundreds (10 stocks × 3 strategies × 23 days), much larger than the previous 7-stock hardcoded backtest.

- [ ] **Step 3: Second run — with weights on**

```bash
source .venv/bin/activate && python run_universe_backtest.py --days 30 --top 10
```

Expected: this run *re-uses* the weights built in step 2's trades. `skipped_by_weight` should now be > 0 if any (engine, symbol) combos accumulated >10 losing trades in step 2. WR/P&L should be neutral or slightly better.

- [ ] **Step 4: Commit**

```bash
git add backend/run_universe_backtest.py
git commit -m "feat: CLI for universe-driven backtest with Layer-1 stats"
```

---

## Verification Checklist (run at end)

- [ ] `python run_universe_backtest.py --no-weights --reset-weights` produces aggregate table with non-zero trade counts for at least one strategy.
- [ ] `python run_universe_backtest.py` (second run) shows `skipped_by_weight > 0` and "Bottom-weighted" lists losing combos from run 1.
- [ ] `backend/data/signal_weights.json` exists and contains keys after run.
- [ ] No `.NS` suffix appears anywhere in `intraday_universe.INTRADAY_UNIVERSE`.

---

## What this plan deliberately does NOT include

- **Daily auto-run scheduler** — out of scope. The CLI can be wired to a cron later.
- **Layer 2 (contextual bandit)** — separate plan, only worth building after Layer 1 has 100+ trades of data.
- **Production unit tests** — verification-driven dev. Output inspection is the test for research code.
- **Frontend integration** — backend-only. Frontend wiring is a follow-up.
- **Live execution path** — paper trading is locked. Universe selection is for backtest first; we add it to live `intraday_engine.py` only after the backtest results justify it.
