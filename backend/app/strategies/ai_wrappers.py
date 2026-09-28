"""
ASTRA AI Engine Wrappers — walk-forward evaluator harness
==========================================================
Wraps the three live AI engines (ASTRA 1.0, ASTRA.AI, ASTRA.ML) as Strategy
subclasses so they can be evaluated by the exact same walk-forward evaluator
used for PULLBACK, MOMENTUM, STAGE2, QUALITY.

**Leakage caveat:** The RF and LSTM models were trained on data overlapping the
2-year evaluation window. In-sample results will be optimistic — treat them as
an upper-bound estimate. True out-of-sample performance will be lower.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from app.strategies.base import (
    ExitReason, Position, Signal, SignalSide,
    Strategy, StrategyMeta, StrategyTimeframe,
)
from app.strategies.indicators import compute_atr, compute_adx

# ── Shared constants (mirror ai_predictor.py) ─────────────────────────────────
FEATURE_COLS = [
    "Dist_SMA10", "Dist_SMA30", "Dist_SMA200", "Dist_VWAP",
    "RSI", "RSI_Slope", "MACD", "MACD_Signal", "MACD_Hist", "ADX",
    "Stoch_K", "Stoch_D", "BB_PctB", "BB_Width", "Daily_Ret",
    "Volat_Ratio", "Volume_Ratio", "OBV_Slope", "Candle_Body_Ratio", "Lower_Shadow_Ratio",
]

MODELS_DIR = pathlib.Path(__file__).parents[2] / "app" / "models" / "saved_models"

# ── Shared time stop ──────────────────────────────────────────────────────────
_AI_TIME_STOP_DAYS = 84   # 60 trading-day cap ≈ 84 calendar days


def _shared_precompute(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 20 FEATURE_COLS + ATR14 in one pass.
    Mirrors _compute_features() in ai_predictor.py using only pandas/numpy
    (no yfinance / external calls).
    """
    df = bars.sort_index().copy()

    # ── SMA distances ─────────────────────────────────────────────────────────
    df["SMA10"]  = df["Close"].rolling(10).mean()
    df["SMA30"]  = df["Close"].rolling(30).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    # 20-bar VWAP proxy (rolling, daily bars)
    df["VWAP20"] = (
        (df["Close"] * df["Volume"]).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    )
    df["Dist_SMA10"]  = (df["Close"] - df["SMA10"])  / df["SMA10"].replace(0, np.nan)  * 100
    df["Dist_SMA30"]  = (df["Close"] - df["SMA30"])  / df["SMA30"].replace(0, np.nan)  * 100
    df["Dist_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"].replace(0, np.nan) * 100
    df["Dist_VWAP"]   = (df["Close"] - df["VWAP20"]) / df["VWAP20"].replace(0, np.nan) * 100

    # ── RSI (14-period Wilder) ─────────────────────────────────────────────────
    delta = df["Close"].diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    df["RSI_Slope"] = df["RSI"].diff(3)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── ADX ───────────────────────────────────────────────────────────────────
    df["ADX"] = compute_adx(df, 14)["ADX"]

    # ── Stochastic ────────────────────────────────────────────────────────────
    lo14 = df["Low"].rolling(14).min()
    hi14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── Bollinger Bands (20-period) ───────────────────────────────────────────
    sma20    = df["Close"].rolling(20).mean()
    std20    = df["Close"].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_PctB"]  = (df["Close"] - bb_lower) / bb_range
    df["BB_Width"] = bb_range / sma20 * 100

    # ── ATR + derived ─────────────────────────────────────────────────────────
    df["ATR14"]       = compute_atr(df, 14)
    df["Volat_Ratio"] = df["ATR14"] / df["Close"].replace(0, np.nan) * 100
    df["Daily_Ret"]   = df["Close"].pct_change() * 100
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

    # ── OBV slope ─────────────────────────────────────────────────────────────
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV_Slope"] = obv.diff(5) / df["Close"].replace(0, np.nan)

    # ── Candle geometry ───────────────────────────────────────────────────────
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Candle_Body_Ratio"]  = (df["Close"] - df["Open"]).abs() / rng
    df["Lower_Shadow_Ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / rng

    return df


def _has_features(row: pd.Series) -> bool:
    """Return True if all required feature cols + ATR14 are non-NaN."""
    for col in FEATURE_COLS + ["ATR14"]:
        val = row.get(col)
        if val is None or pd.isna(val):
            return False
    return True


# ── 1. ASTRA 1.0 — 4/6 rule-based ────────────────────────────────────────────

class Astra1Strategy(Strategy):
    """
    Backtestable wrapper for the ASTRA 1.0 rule-based engine.
    Replicates the 4/6 multi-confirmation BUY logic from ai_predictor.py.
    Macro/network calls are excluded (backtests must be deterministic).
    """

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.1.0",
            version="1.0.0",
            description="4/6 multi-confirmation rule-based system (backtestable wrapper)",
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["rule-based", "multi-confirmation", "ai-wrapper"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return _shared_precompute(bars)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < 210:
            return None
        row = bars.iloc[-1]
        if not _has_features(row):
            return None

        close = float(row["Close"])
        atr   = float(row["ATR14"])

        # Replicate the 6 rule-based confirmations from ai_predictor.py
        # (macro/network calls removed for deterministic backtesting)
        # Note: RSI threshold relaxed to 40 (live uses 35; 35+SMA200>0 almost never co-occurs)
        checks = [
            # 1. RSI oversold & turning up (relaxed from 35 → 40 for backtest viability)
            float(row["RSI"]) < 40 and float(row["RSI_Slope"]) > 0,
            # 2. Price above SMA200 + trending (ADX>20)
            float(row["Dist_SMA200"]) > 0 and float(row["ADX"]) > 20,
            # 3. MACD bullish (line above signal)
            float(row["MACD"]) > float(row["MACD_Signal"]),
            # 4. Volume surge (>1.5× 20-day avg)
            float(row["Volume_Ratio"]) > 1.5,
            # 5. Bullish candle (close > open = positive body)
            float(row["Close"]) > float(row["Open"]),
            # 6. Near lower Bollinger Band (oversold zone)
            float(row["BB_PctB"]) < 0.35,
        ]
        passed = sum(checks)
        if passed < 4:
            return None

        hard_stop = close - 1.5 * atr
        target    = close + 3.0 * atr
        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,
            trail_pct=None,
            trail_activate_pct=0.0,
            confidence=round(passed / 6 * 100, 1),
            size_fraction=1.0,
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= _AI_TIME_STOP_DAYS:
            return ExitReason.TIME_STOP
        return None


# ── 2. ASTRA.AI — Random Forest ───────────────────────────────────────────────

class AstraAIStrategy(Strategy):
    """
    Backtestable wrapper for the ASTRA.AI Random Forest engine.
    Loads the saved RF model from saved_models/astra_rf.joblib.
    Predicts 5-day forward return; enters BUY when prediction exceeds threshold.

    NOTE: RF was trained on data overlapping the evaluation window.
    Results are an upper-bound estimate (in-sample optimism applies).
    """

    def __init__(self):
        import joblib
        model_path = MODELS_DIR / "astra_rf.joblib"
        self._model = joblib.load(str(model_path))

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.AI",
            version="1.0.0",
            description="Random Forest 5-day return predictor (backtestable wrapper)",
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["ml", "random-forest", "ai-wrapper"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return _shared_precompute(bars)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < 230:
            return None
        row = bars.iloc[-1]
        if not _has_features(row):
            return None

        close   = float(row["Close"])
        atr     = float(row["ATR14"])
        atr_pct = float(row["Volat_Ratio"])

        X    = pd.DataFrame([row[FEATURE_COLS]])
        pred = float(self._model.predict(X)[0])

        # Dynamic threshold: at minimum 1.2%, or half the daily ATR%
        threshold = max(1.2, atr_pct * 0.5)
        if pred <= threshold:
            return None

        hard_stop  = close - 1.6 * atr
        target     = close + 4.0 * atr
        confidence = min(abs(pred) * 18.0, 99.0)

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,
            trail_pct=None,
            trail_activate_pct=0.0,
            confidence=round(confidence, 1),
            size_fraction=1.0,
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= _AI_TIME_STOP_DAYS:
            return ExitReason.TIME_STOP
        return None


# ── 3. ASTRA.ML — Bidirectional LSTM ─────────────────────────────────────────

class AstraMLStrategy(Strategy):
    """
    Backtestable wrapper for the ASTRA.ML Bidirectional LSTM engine.
    Loads saved_models/astra_lstm_equity.keras + astra_lstm_equity_scaler.joblib.
    Uses a 30-bar lookback window; predicts 3-day forward return.

    NOTE: LSTM was trained on data overlapping the evaluation window.
    Results are an upper-bound estimate (in-sample optimism applies).
    """

    LOOKBACK = 30

    def __init__(self):
        import joblib
        import tensorflow as tf  # noqa: F401 — lazy import so non-ML setups skip TF
        self._model  = tf.keras.models.load_model(str(MODELS_DIR / "astra_lstm_equity.keras"))
        self._scaler = joblib.load(str(MODELS_DIR / "astra_lstm_equity_scaler.joblib"))

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.ML",
            version="1.0.0",
            description="Bidirectional LSTM 3-day return predictor (backtestable wrapper)",
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["ml", "lstm", "ai-wrapper"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return _shared_precompute(bars)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(bars) < 210 + self.LOOKBACK:
            return None
        row = bars.iloc[-1]
        if row[FEATURE_COLS].isna().any():
            return None

        close   = float(row["Close"])
        atr     = float(row["ATR14"])
        atr_pct = float(row["Volat_Ratio"])

        # Build 30-bar feature matrix
        feat_data = bars[FEATURE_COLS].iloc[-self.LOOKBACK:].values
        if feat_data.shape[0] < self.LOOKBACK:
            return None

        scaled = self._scaler.transform(feat_data)
        X = scaled.reshape(1, self.LOOKBACK, len(FEATURE_COLS))
        pred = float(self._model.predict(X, verbose=0)[0][0])

        # Dynamic threshold: at minimum 0.55%, or 35% of daily ATR%
        threshold = max(0.55, atr_pct * 0.35)
        if pred <= threshold:
            return None

        hard_stop  = close - 1.5 * atr
        target     = close + 3.5 * atr
        confidence = min(abs(pred) * 25.0, 99.0)

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,
            trail_pct=None,
            trail_activate_pct=0.0,
            confidence=round(confidence, 1),
            size_fraction=1.0,
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= _AI_TIME_STOP_DAYS:
            return ExitReason.TIME_STOP
        return None
