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

import os as _os
# ASTRA_MODELS_DIR lets the evaluator load held-out ("trained before the test
# window") models instead of the production ones.
MODELS_DIR = pathlib.Path(_os.getenv("ASTRA_MODELS_DIR") or
                          pathlib.Path(__file__).parents[2] / "app" / "models" / "saved_models")

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


# ── Live-parity wrappers ──────────────────────────────────────────────────────
# v1.14: the three engines are backtested through ai_engine.decide() — the same
# function the live /api/analyze endpoint uses — on features from the live
# _compute_features(). Market regime (NIFTY vs SMA200) and the weekly-trend
# filter are rebuilt per date without look-ahead. Long-only: the evaluator
# trades BUY signals (positional shorting isn't available in Indian cash equity).

_NIFTY_BULL: Optional[pd.Series] = None


def _nifty_bull_series() -> pd.Series:
    """Date → NIFTY close above its 200-day SMA (the live macro filter)."""
    global _NIFTY_BULL
    if _NIFTY_BULL is None:
        from app.services.market_data import market_data
        n = market_data.get_ohlcv("^NSEI", period="10y", interval="1d")
        if len(n) >= 200:
            c = n["Close"].sort_index()
            c.index = _naive(c.index)
            _NIFTY_BULL = (c > c.rolling(200).mean()).astype(bool)
        else:
            _NIFTY_BULL = pd.Series(dtype=bool)
    return _NIFTY_BULL


def _naive(idx):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _weekly_trend_series(close: pd.Series) -> pd.Series:
    """Live _get_weekly_trend() per day without look-ahead: weekly SMA10/SMA30
    from completed weeks plus the current day's close as the running week."""
    c = close.copy()
    c.index = _naive(c.index)
    wk_label = c.index.to_period("W-FRI")
    weekly = c.groupby(wk_label).last()
    prev9 = weekly.rolling(9).sum().shift(1)       # 9 completed weeks before this one
    prev29 = weekly.rolling(29).sum().shift(1)
    s10 = (prev9.reindex(wk_label).values + c.values) / 10.0
    s30 = (prev29.reindex(wk_label).values + c.values) / 30.0
    bull = (c.values > s30) & (s10 > s30)
    bear = (c.values < s30) & (s10 < s30)
    out = np.where(bull, "BULL", np.where(bear, "BEAR", "NEUTRAL"))
    out = np.where(np.isnan(s30), "NEUTRAL", out)
    return pd.Series(out, index=close.index)


class _LiveEngineStrategy(Strategy):
    ENGINE = "astra"
    NAME = "ASTRA.1.0"
    DESCRIPTION = ""
    TAGS: list = []
    MIN_BARS = 210

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name=self.NAME, version="2.0.0", description=self.DESCRIPTION,
            timeframe=StrategyTimeframe.DAILY, asset_class="EQUITY", long_only=True,
            owner="ASTRA", tags=list(self.TAGS) + ["ai-wrapper", "live-parity"],
        )

    def _models(self):
        """Override to supply (rf_model, lstm_model, lstm_scaler)."""
        return None, None, None

    def _extra_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        return df

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        from app.services.ai_predictor import ai_engine
        bars = bars.sort_index()
        feats = ai_engine._compute_features(bars)
        if feats.empty:
            return feats
        # keep every input bar (warm-up rows just have NaN features → no signal)
        df = feats.reindex(bars.index)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = bars[c]
        df["ATR14"] = df["ATR"] if "ATR" in df.columns else compute_atr(df, 14)
        bull = _nifty_bull_series()
        if len(bull):
            df["_macro_bull"] = bull.reindex(_naive(df.index), method="pad").fillna(True).values
        else:
            df["_macro_bull"] = True
        df["_weekly"] = _weekly_trend_series(df["Close"]).values
        return df

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        from app.services.ai_predictor import ai_engine
        if len(bars) < self.MIN_BARS:
            return None
        row = bars.iloc[-1]
        if row[FEATURE_COLS].isna().any():
            return None
        rf_pred = float(row["_rf_pred"]) if "_rf_pred" in bars.columns and pd.notna(row.get("_rf_pred")) else None
        ml_probs = None
        if "P_UP" in bars.columns:
            if pd.isna(row.get("P_UP")):
                return None
            ml_probs = [float(row["P_DOWN"]), float(row["P_NEUTRAL"]), float(row["P_UP"])]
        side, confidence, m_sl, m_tp = ai_engine.decide(
            self.ENGINE, bars.tail(40), symbol, bool(row["_macro_bull"]), str(row["_weekly"]),
            rf_pred=rf_pred, ml_probs=ml_probs)
        if side != "BUY":
            return None
        close, atr = float(row["Close"]), float(row["ATR14"])
        return Signal(
            symbol=symbol, side=SignalSide.BUY, entry_price=close, timestamp=bars.index[-1],
            hard_stop=close - m_sl * atr, target=close + m_tp * atr,
            trail_pct=None, trail_activate_pct=0.0, confidence=round(confidence, 1), size_fraction=1.0,
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= _AI_TIME_STOP_DAYS:
            return ExitReason.TIME_STOP
        return None


class Astra1Strategy(_LiveEngineStrategy):
    """ASTRA 1.0 — 4/7 multi-confirmation rules (exact live logic)."""
    ENGINE, NAME = "astra", "ASTRA.1.0"
    DESCRIPTION = "4/7 multi-confirmation rule-based system (live-parity backtest)"
    TAGS = ["rule-based", "multi-confirmation"]


class AstraAIStrategy(_LiveEngineStrategy):
    """ASTRA.AI — Random Forest 5-day return model (exact live decision rule)."""
    ENGINE, NAME = "astra_ai", "ASTRA.AI"
    DESCRIPTION = "Random Forest 5-day return predictor (live-parity backtest)"
    TAGS = ["ml", "random-forest"]
    MIN_BARS = 230

    def __init__(self):
        from app.services.tree_ensemble import load_rf_model
        self._model = load_rf_model(MODELS_DIR)
        if self._model is None:
            raise FileNotFoundError(f"ASTRA.AI model not found in {MODELS_DIR}")

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = super().precompute(bars)
        if df.empty:
            return df
        ok = df[FEATURE_COLS].notna().all(axis=1)
        df["_rf_pred"] = np.nan
        if ok.any():
            df.loc[ok, "_rf_pred"] = self._model.predict(df.loc[ok, FEATURE_COLS].values)
        return df

    def signal(self, bars, symbol):
        from app.services.ai_predictor import ai_engine
        ai_engine.rf_model = self._model          # decide() checks a model is loaded
        return super().signal(bars, symbol)


class AstraMLStrategy(_LiveEngineStrategy):
    """ASTRA.ML — 3-class BiLSTM (exact live decision rule)."""
    ENGINE, NAME = "astra_ml", "ASTRA.ML"
    DESCRIPTION = "3-class BiLSTM 20-day direction model (live-parity backtest)"
    TAGS = ["ml", "lstm"]

    def __init__(self):
        import joblib
        from app.services.model_compat import load_keras_model
        from app.services.ai_predictor import _get_luong_attention_cls
        self._model = load_keras_model(str(MODELS_DIR / "astra_lstm_equity.keras"),
                                       custom_objects={"LuongAttention": _get_luong_attention_cls()})
        self._scaler = joblib.load(str(MODELS_DIR / "astra_lstm_equity_scaler.joblib"))
        if int(self._model.output_shape[-1]) != 3:
            raise ValueError("ASTRA.ML model is not a 3-class classifier — retrain it")
        self.LOOKBACK = int(self._model.input_shape[1])
        self.MIN_BARS = 210 + self.LOOKBACK
        self._symbol_hint = None

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        from app.services.ai_predictor import lstm_predict_proba_series
        df = super().precompute(bars)
        if df.empty:
            return df
        sym = bars.attrs.get("symbol", "") if hasattr(bars, "attrs") else ""
        probs = lstm_predict_proba_series(self._model, self._scaler, sym, df)
        return df.join(probs, how="left")
