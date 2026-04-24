"""
ASTRA Experience Replay Buffer
================================
Captures closed trade outcomes and converts them into training samples
for incremental model fine-tuning.

Every time a position closes (in Celery monitor_active_positions), call:
    replay_buffer.push(symbol, features_dict, outcome_dict)

The buffer persists to disk (JSONL) and can be sampled by train_models.py
for incremental fine-tuning.

Usage (in tasks.py, after position closes):
    from app.services.replay_buffer import replay_buffer
    replay_buffer.push(
        symbol=position.symbol,
        features=position.entry_features_json,  # dict of 20 feature values
        outcome={
            "pnl_pct": realized_pnl,
            "hit_target": hit_tp,
            "hit_sl": hit_sl,
            "bars_held": n_bars,
            "direction": direction,
        }
    )

Usage (in train_models.py for incremental fine-tune):
    from app.services.replay_buffer import replay_buffer
    X, y = replay_buffer.to_training_data(feature_cols, lookback=30)
    if X is not None and len(X) >= 64:
        # Fine-tune LSTM head on recent experience
        ...
"""

import json
import logging
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Buffer config
BUFFER_DIR  = Path(__file__).parent.parent.parent / "data" / "replay_buffer"
BUFFER_FILE = BUFFER_DIR / "experiences.jsonl"
METRICS_FILE = BUFFER_DIR / "engine_metrics.json"
MAX_BUFFER  = 2000       # Maximum experiences to keep in memory
RECENT_WEIGHT = 3        # Recent experiences are sampled 3× more often
MIN_PNL_LABEL_BUY  = 1.5   # % — above this is a "UP" label (class 2)
MIN_PNL_LABEL_SELL = -1.5  # % — below this is a "DOWN" label (class 0)


class ExperienceReplayBuffer:
    """
    Disk-persisted ring buffer of closed trade experiences.

    Each experience contains:
      - symbol: str
      - features: dict of indicator values at entry
      - outcome: {pnl_pct, hit_target, hit_sl, bars_held, direction}
      - engine: which engine generated the signal
      - ts: ISO timestamp
    """

    def __init__(self, maxlen: int = MAX_BUFFER):
        self.maxlen = maxlen
        self.buffer: deque = deque(maxlen=maxlen)
        self._engine_stats: dict = {}   # {engine: {wins, losses, total_pnl}}
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()
        self._load_engine_metrics()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load existing experiences from JSONL file on startup."""
        if not BUFFER_FILE.exists():
            logger.info("Replay buffer: no existing file, starting fresh.")
            return
        try:
            loaded = 0
            with open(BUFFER_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.buffer.append(record)
                        loaded += 1
                    except json.JSONDecodeError:
                        continue
            # Keep only last maxlen records (deque handles this, but file may be huge)
            logger.info(f"Replay buffer: loaded {loaded} experiences from disk.")
        except Exception as e:
            logger.error(f"Replay buffer load failed: {e}")

    def _append_to_disk(self, record: dict):
        """Append a single experience to the JSONL file."""
        try:
            with open(BUFFER_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Replay buffer disk write failed: {e}")

    def _load_engine_metrics(self):
        """Load per-engine win/loss stats from disk."""
        if not METRICS_FILE.exists():
            self._engine_stats = {}
            return
        try:
            with open(METRICS_FILE, "r") as f:
                self._engine_stats = json.load(f)
        except Exception:
            self._engine_stats = {}

    def _save_engine_metrics(self):
        """Persist per-engine metrics to disk."""
        try:
            with open(METRICS_FILE, "w") as f:
                json.dump(self._engine_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Engine metrics save failed: {e}")

    # ── Write ──────────────────────────────────────────────────────────────────

    def push(self, symbol: str, features: dict, outcome: dict,
             engine: str = "unknown"):
        """
        Record a closed trade experience.

        Args:
            symbol:   e.g. "TCS.NS"
            features: dict of indicator values at entry (20 ASTRA features)
            outcome:  {pnl_pct, hit_target, hit_sl, bars_held, direction}
            engine:   which engine generated the signal
        """
        record = {
            "symbol":   symbol,
            "engine":   engine,
            "features": features,
            "outcome":  outcome,
            "ts":       datetime.utcnow().isoformat(),
        }
        self.buffer.append(record)
        self._append_to_disk(record)
        self._update_engine_stats(engine, outcome)
        logger.debug(f"Replay buffer: pushed {symbol} {engine} "
                     f"pnl={outcome.get('pnl_pct', '?'):.2f}%  "
                     f"buffer_size={len(self.buffer)}")

    def _update_engine_stats(self, engine: str, outcome: dict):
        """Track per-engine win rate and cumulative P&L."""
        if engine not in self._engine_stats:
            self._engine_stats[engine] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
                "last_updated": datetime.utcnow().isoformat(),
            }
        pnl = outcome.get("pnl_pct", 0.0)
        won = pnl > 0
        self._engine_stats[engine]["wins"]      += int(won)
        self._engine_stats[engine]["losses"]    += int(not won)
        self._engine_stats[engine]["total_pnl"] += pnl
        self._engine_stats[engine]["last_updated"] = datetime.utcnow().isoformat()
        self._save_engine_metrics()

    # ── Read ───────────────────────────────────────────────────────────────────

    def sample(self, n: int = 256, recent_bias: bool = True) -> list:
        """
        Sample n experiences from the buffer.
        With recent_bias=True, the last 20% of experiences are
        RECENT_WEIGHT times more likely to be selected.
        """
        if len(self.buffer) == 0:
            return []

        buf_list = list(self.buffer)
        if not recent_bias or len(buf_list) < 20:
            k = min(n, len(buf_list))
            return random.sample(buf_list, k)

        # Build weighted index: last 20% of buffer get 3× weight
        cutoff = int(len(buf_list) * 0.80)
        old_part = buf_list[:cutoff]
        new_part = buf_list[cutoff:]
        # Expand new_part by RECENT_WEIGHT
        weighted = old_part + new_part * RECENT_WEIGHT
        k = min(n, len(weighted))
        return random.sample(weighted, k)

    def to_training_data(self, feature_cols: list,
                         lookback: int = 1,
                         min_samples: int = 64
                         ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert buffer to (X, y) numpy arrays for model training.

        For LSTM fine-tuning (lookback > 1): we can only build sequences
        from symbols that appear consecutively, which is complex — so by
        default (lookback=1) we return flat feature vectors suitable for
        RF incremental update or a dense head fine-tune.

        Label encoding:
          pnl_pct > MIN_PNL_LABEL_BUY  → class 2 (UP)
          pnl_pct < MIN_PNL_LABEL_SELL → class 0 (DOWN)
          otherwise                    → class 1 (NEUTRAL)

        Returns (None, None) if fewer than min_samples experiences available.
        """
        if len(self.buffer) < min_samples:
            logger.info(f"Replay buffer: only {len(self.buffer)} samples, "
                        f"need {min_samples} for training.")
            return None, None

        rows, labels = [], []
        for exp in self.buffer:
            feats = exp.get("features", {})
            pnl   = exp["outcome"].get("pnl_pct", 0.0)

            # Build feature vector in the correct column order
            row = []
            valid = True
            for col in feature_cols:
                val = feats.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    valid = False
                    break
                row.append(float(val))

            if not valid:
                continue

            # Direction label
            if pnl > MIN_PNL_LABEL_BUY:
                label = 2   # UP
            elif pnl < MIN_PNL_LABEL_SELL:
                label = 0   # DOWN
            else:
                label = 1   # NEUTRAL

            rows.append(row)
            labels.append(label)

        if len(rows) < min_samples:
            return None, None

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # For LSTM: reshape to (n, lookback, features) if lookback > 1
        if lookback > 1 and len(X) >= lookback:
            # Build sliding windows (no time ordering guaranteed in buffer)
            # — use only for dense/Conv1D head, not for recurrent layers
            X_seq, y_seq = [], []
            for i in range(lookback, len(X)):
                X_seq.append(X[i - lookback:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)

        return X, y

    # ── Introspection ──────────────────────────────────────────────────────────

    def engine_win_rates(self) -> dict:
        """Return per-engine win rate % and cumulative P&L."""
        summary = {}
        for eng, stats in self._engine_stats.items():
            total = stats["wins"] + stats["losses"]
            wr = round(stats["wins"] / total * 100, 1) if total > 0 else 0.0
            summary[eng] = {
                "win_rate":   wr,
                "total_pnl":  round(stats["total_pnl"], 2),
                "trades":     total,
                "last_updated": stats.get("last_updated", ""),
            }
        return summary

    def drift_check(self, window: int = 20) -> dict:
        """
        Check recent win rate vs overall win rate.
        If recent WR drops more than 15pp below overall, flag drift.
        """
        if len(self.buffer) < window * 2:
            return {"drift_detected": False, "reason": "insufficient_data"}

        buf_list = list(self.buffer)
        recent = buf_list[-window:]
        overall = buf_list

        def win_rate(experiences):
            wins = sum(1 for e in experiences if e["outcome"].get("pnl_pct", 0) > 0)
            return wins / len(experiences) * 100 if experiences else 0.0

        recent_wr  = win_rate(recent)
        overall_wr = win_rate(overall)
        drop       = overall_wr - recent_wr
        drifting   = drop > 15.0 or recent_wr < 30.0

        return {
            "drift_detected":  drifting,
            "overall_win_rate": round(overall_wr, 1),
            "recent_win_rate":  round(recent_wr, 1),
            "win_rate_drop":    round(drop, 1),
            "window":          window,
            "recommendation":  "retrain" if drifting else "ok",
        }

    def summary(self) -> dict:
        """Overall buffer summary."""
        if not self.buffer:
            return {"size": 0}
        all_pnl = [e["outcome"].get("pnl_pct", 0) for e in self.buffer]
        wins = [p for p in all_pnl if p > 0]
        return {
            "size":        len(self.buffer),
            "win_rate":    round(len(wins) / len(all_pnl) * 100, 1),
            "avg_pnl":     round(float(np.mean(all_pnl)), 3),
            "avg_win":     round(float(np.mean(wins)), 3) if wins else 0.0,
            "total_pnl":   round(float(np.sum(all_pnl)), 2),
            "engines":     self.engine_win_rates(),
            "drift":       self.drift_check(),
        }

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"ExperienceReplayBuffer(size={len(self)}, maxlen={self.maxlen})"


# Module-level singleton
replay_buffer = ExperienceReplayBuffer()
