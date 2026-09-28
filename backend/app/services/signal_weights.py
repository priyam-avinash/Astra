"""
Layer 1 Self-Learning — Online Signal Weighting
================================================
Tracks rolling P&L per (engine, symbol) combination and produces a weight in [0, 1.5]
that scales position size / suppresses the trade entirely.

Bayesian update with weak priors so the system is usable from trade 1, not trade 100.
Persisted to a JSON file — no DB needed.

Usage:
    from app.services.signal_weights import signal_weights
    w = signal_weights.weight("ORB", "RELIANCE")
    if w < 0.3: skip_trade()
    ...
    signal_weights.update("ORB", "RELIANCE", pnl_rs=420)   # call after exit
"""

import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).parent.parent / "data" / "signal_weights.json"
_WINDOW     = 50          # rolling window size per (engine, symbol)
_PRIOR_WINS = 3.0         # Beta prior — 3 fake wins (lighter so system learns faster)
_PRIOR_LOSS = 3.0         # 3 fake losses → start at WR 0.5 with fast updating
_MIN_WEIGHT = 0.0         # 0 means "do not take this trade"
_MAX_WEIGHT = 1.5         # cap upside leverage at 1.5×


class SignalWeights:
    """Bayesian rolling WR tracker per (engine, symbol)."""

    def __init__(self):
        self._lock   = Lock()
        self._state: dict = {}
        self._load()

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

        # For 1:3 R:R, break-even WR is 0.25. Map:
        #   WR ≤ 0.25 → 0.0    (suppress, below break-even)
        #   WR  = 0.35 → 1.0   (neutral)
        #   WR ≥ 0.55 → 1.5    (max boost)
        if post_wr <= 0.25:
            return _MIN_WEIGHT
        if post_wr >= 0.55:
            return _MAX_WEIGHT
        return round(max(_MIN_WEIGHT, min(_MAX_WEIGHT, 1.0 + (post_wr - 0.35) * 5.0)), 2)

    def stats(self, engine: Optional[str] = None) -> dict:
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

    def _load(self) -> None:
        if _STATE_PATH.exists():
            try:
                with open(_STATE_PATH) as f:
                    self._state = json.load(f)
                logger.info(f"[signal_weights] Loaded {len(self._state)} keys")
            except Exception as e:
                logger.warning(f"[signal_weights] Load failed: {e}")
                self._state = {}

    def _save(self) -> None:
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_PATH, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"[signal_weights] Save failed: {e}")


signal_weights = SignalWeights()
