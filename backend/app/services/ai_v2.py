"""
ASTRA.AI v2 — live scoring for the walk-forward-validated ranking model.

The model (a numpy-exported tree ensemble) scores every stock in the trading
universe on the v2 feature set (features_v2.py). A stock is a BUY when its
score is in the top (1 − q) of today's universe and, if the config says so,
the market regime (NIFTY above its 200-day average) is bullish. Exits use the
ATR stop/target/time limit chosen in validation. All parameters come from
saved_models/astra_ai_v2.json, written by research/export_model.py.

A universe snapshot (latest feature row per stock + cross-sectional ranks) is
built once and cached for SNAPSHOT_TTL_SEC, so single-stock analysis and the
universe scan share one computation.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "saved_models"
SNAPSHOT_TTL_SEC = 30 * 60


class AIv2:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.cfg: Optional[dict] = None
        self.model = None
        self._snap: Optional[pd.DataFrame] = None
        self._snap_at = 0.0
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        cfg_path = self.models_dir / "astra_ai_v2.json"
        npz = self.models_dir / "astra_ai_v2_trees.npz"
        if not (cfg_path.exists() and npz.exists()):
            logger.info("ASTRA.AI v2 model not found — v1 Random Forest stays active")
            return
        try:
            from app.services.tree_ensemble import NumpyForestRegressor
            self.cfg = json.loads(cfg_path.read_text())
            self.model = NumpyForestRegressor.load(str(npz))
            logger.info(f"Loaded ASTRA.AI v2 ({self.cfg.get('label')}/{self.cfg.get('algo')}, "
                        f"trained to {self.cfg.get('trained_until')})")
        except Exception as e:
            logger.error(f"ASTRA.AI v2 failed to load: {e}")
            self.cfg = self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.cfg is not None

    # ── snapshot ─────────────────────────────────────────────────────────
    def snapshot(self, force: bool = False) -> pd.DataFrame:
        if not force and self._snap is not None and time.time() - self._snap_at < SNAPSHOT_TTL_SEC:
            return self._snap
        with self._lock:
            if not force and self._snap is not None and time.time() - self._snap_at < SNAPSHOT_TTL_SEC:
                return self._snap
            from app.services.features_v2 import market_frame, symbol_features, add_cross_sectional, feature_columns
            from app.services.intraday_universe import INTRADAY_UNIVERSE
            from app.services.market_data import market_data
            from app.strategies.pullback import SECTOR_MAP
            nifty = market_data.get_ohlcv("^NSEI", period="2y", interval="1d")
            if len(nifty) < 220:
                raise RuntimeError("NIFTY history unavailable — can't build the v2 snapshot")
            mkt = market_frame(nifty)

            def one(sym):
                try:
                    raw = market_data.get_ohlcv(f"{sym}.NS", period="2y", interval="1d")
                    if len(raw) < 300:
                        return None
                    f = symbol_features(raw, mkt).iloc[[-1]].copy()
                    f["symbol"], f["sector"] = sym, SECTOR_MAP.get(sym, "OTHER")
                    f["data_source"] = raw.attrs.get("source")
                    f["data_stale"] = bool(raw.attrs.get("stale"))
                    return f
                except Exception as e:
                    logger.debug(f"[v2] {sym}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=6) as pool:
                rows = [r for r in pool.map(one, INTRADAY_UNIVERSE) if r is not None]
            if len(rows) < 30:
                raise RuntimeError(f"Only {len(rows)} universe stocks have data — v2 snapshot incomplete")
            ds = pd.concat(rows)
            ds.index.name = "date"
            ds = ds.reset_index()
            # stocks whose last bar is older than the freshest bar are scored but not ranked against today
            ds["date"] = ds["date"].max()
            ds = add_cross_sectional(ds)
            feats = feature_columns()
            ok = ds[feats].notna().all(axis=1)
            ds["score"] = np.nan
            if ok.any():
                ds.loc[ok, "score"] = self.model.predict(ds.loc[ok, feats].values)
            ds["pct"] = ds["score"].rank(pct=True)
            self._snap, self._snap_at = ds.set_index("symbol"), time.time()
            logger.info(f"[v2] snapshot built: {int(ok.sum())}/{len(ds)} stocks scored")
            return self._snap

    # ── decisions ────────────────────────────────────────────────────────
    def decide(self, symbol: str) -> Optional[dict]:
        """Signal for one stock, or None if it isn't in the v2 universe."""
        if not self.available:
            return None
        sym = symbol.upper().replace(".NS", "")
        snap = self.snapshot()
        if sym not in snap.index or pd.isna(snap.at[sym, "score"]):
            return None
        row = snap.loc[sym]
        q = float(self.cfg.get("q", 0.9))
        regime_ok = (not self.cfg.get("regime")) or float(row.get("mkt_above_200", 1)) > 0.5
        pct = float(row["pct"])
        signal = "BUY" if (pct >= q and regime_ok) else "HOLD"
        reason = None
        if pct >= q and not regime_ok:
            reason = "Top-ranked, but NIFTY is below its 200-day average (regime filter)"
        return {
            "signal": signal,
            "confidence": round(pct * 100, 1),            # universe percentile of the model score
            "score": round(float(row["score"]), 4),
            "rank_pct": round(pct * 100, 1),
            "sl_atr": float(self.cfg.get("stop_k", 1.5)),
            "tp_atr": float(self.cfg.get("tgt_k", 3.0)),
            "max_hold_days": int(self.cfg.get("max_bars", 20)),
            "regime_bull": bool(float(row.get("mkt_above_200", 1)) > 0.5),
            "reason": reason,
        }

    def scan(self, top_k: int = 20) -> list[dict]:
        if not self.available:
            return []
        snap = self.snapshot().dropna(subset=["score"]).sort_values("score", ascending=False)
        q = float(self.cfg.get("q", 0.9))
        regime_ok = (not self.cfg.get("regime")) or float(snap["mkt_above_200"].iloc[0]) > 0.5
        out = []
        for sym, row in snap.head(top_k).iterrows():
            atr, close = float(row["ATR"]), float(row["Close"])
            buy = float(row["pct"]) >= q and regime_ok
            out.append({
                "symbol": f"{sym}.NS", "signal": "BUY" if buy else "HOLD",
                "confidence": round(float(row["pct"]) * 100, 1),
                "entry_price": round(close, 2),
                "stop_loss": round(close - float(self.cfg.get("stop_k", 1.5)) * atr, 2),
                "target": round(close + float(self.cfg.get("tgt_k", 3.0)) * atr, 2),
                "sector": row.get("sector"), "engine": "astra_ai_v2",
            })
        return out


_engine: Optional[AIv2] = None


def get_ai_v2() -> AIv2:
    global _engine
    if _engine is None:
        _engine = AIv2()
    return _engine
