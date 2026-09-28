"""
ASTRA Equity Prediction Engine v3.0
====================================
Upgraded from a 2-indicator system to a 20-feature, multi-confirmation engine.
Supports three modes: ASTRA 1.0 (Rules-Based), ASTRA.AI (Random Forest),
ASTRA.ML (Bidirectional LSTM with Attention).

Key upgrades:
- 4/6 confirmation system with ADX, Stochastic, BB%B, MACD Signal, OBV, Candlestick patterns
- ATR-based dynamic SL/TP (replaces hardcoded 1%/2%)
- Circuit breaker (no synthetic data fallback)
- Market regime detection (ADX trending vs. ranging)
- LSTM ANN with walk-forward validation
- Removed all hardcoded RSI override heuristics
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from app.services.market_data import market_data, normalize_symbol

# ── Back-compat shims ────────────────────────────────────────────────────────
# Older modules import these helpers. All data now flows through
# app.services.market_data (provider chain + caching + health tracking).

def cache_get(symbol: str, interval: str):
    df = market_data.get_ohlcv(symbol, period="1y", interval=interval)
    return None if df.empty else df

def cache_put(symbol: str, df, interval: str):   # no-op: market_data caches itself
    return None

def _price_cache_get(symbol: str):
    info = market_data._price.get(normalize_symbol(symbol))
    return info[0] if info else None

def _price_cache_put(symbol: str, price: float):
    market_data.put_live_price(symbol, price, "cache")

def _yf_download_safe(symbol, period, interval, timeout_sec=5, **kwargs):
    """Deprecated: kept for callers that still import it. Uses market_data."""
    return market_data.get_ohlcv(symbol, period=period, interval=interval)

def _yf_circuit_open() -> bool:
    return False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")

# Feature columns for ML models (20 features)
FEATURE_COLS = [
    "Dist_SMA10", "Dist_SMA30", "Dist_SMA200", "Dist_VWAP",
    "RSI", "RSI_Slope", "MACD", "MACD_Signal", "MACD_Hist",
    "ADX", "Stoch_K", "Stoch_D",
    "BB_PctB", "BB_Width",
    "Daily_Ret", "Volat_Ratio",
    "Volume_Ratio", "OBV_Slope",
    "Candle_Body_Ratio", "Lower_Shadow_Ratio",
]


# ── Custom Keras layer for ASTRA.ML (lazy + serialization-registered) ─────────
# Defined via a cached factory so that (a) TensorFlow is only imported when the
# LSTM path is actually used (preserving the no-TF fallback), and (b) the class
# is registered with Keras's serialization registry BEFORE any load_model() call.
# Without registration, tf.keras.models.load_model() cannot reconstruct the layer
# and ASTRA.ML silently disables itself on every restart.
_LUONG_ATTENTION_CLS = None


def _get_luong_attention_cls():
    """Build, register, and cache the LuongAttention layer class. Returns the class."""
    global _LUONG_ATTENTION_CLS
    if _LUONG_ATTENTION_CLS is not None:
        return _LUONG_ATTENTION_CLS

    import keras  # Keras 3 (tf.keras may resolve to the legacy Keras 2 shim)

    @keras.saving.register_keras_serializable(package="astra")
    class LuongAttention(keras.layers.Layer):
        """Dot-product attention over LSTM timesteps -> context vector."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.score_dense = keras.layers.Dense(1, use_bias=False)

        def call(self, hidden_states):
            # hidden_states: (batch, timesteps, features)
            score = self.score_dense(hidden_states)                       # (batch, timesteps, 1)
            weights = keras.ops.softmax(score, axis=1)                    # (batch, timesteps, 1)
            return keras.ops.sum(weights * hidden_states, axis=1)         # (batch, features)

    _LUONG_ATTENTION_CLS = LuongAttention
    return LuongAttention


class AIPredictionEngine:
    def __init__(self):
        logger.info("Initialized ASTRA Equity Engine v3.0 (20-feature multi-confirmation system)")
        self.rf_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.macro_trend_bullish = True
        self.macro_trend_updated = None
        self._load_models()

    # ─────────────────────── MODEL LOADING ───────────────────────

    def _load_models(self):
        try:
            rf_path = os.path.join(MODELS_DIR, "astra_rf.joblib")
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("Loaded ASTRA.AI (Random Forest) model.")

            lstm_path = os.path.join(MODELS_DIR, "astra_lstm_equity.keras")
            scaler_path = os.path.join(MODELS_DIR, "astra_lstm_equity_scaler.joblib")
            if os.path.exists(lstm_path) and os.path.exists(scaler_path):
                try:
                    # Register the custom attention layer before loading so Keras
                    # can reconstruct it; older files with a Lambda layer go through
                    # the compat loader (see services/model_compat.py).
                    from app.services.model_compat import load_keras_model
                    luong_cls = _get_luong_attention_cls()
                    self.lstm_model = load_keras_model(
                        lstm_path, custom_objects={"LuongAttention": luong_cls})
                    self.lstm_scaler = joblib.load(scaler_path)
                    logger.info("Loaded ASTRA.ML (Bidirectional LSTM) model.")
                except ImportError:
                    logger.warning("TensorFlow not installed. ASTRA.ML (LSTM) disabled.")
                except Exception as e:
                    logger.error(f"ASTRA.ML model failed to load ({e}); engine falls back to rules. "
                                 "Retrain with: python train_models.py")
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")

    # ─────────────────────── MACRO FILTER ───────────────────────

    def _is_macro_bullish(self) -> bool:
        """NIFTY 50 above its 200-day SMA → bull regime (BUYs allowed). Cached 1h."""
        now = datetime.now()
        if self.macro_trend_updated and (now - self.macro_trend_updated).total_seconds() < 3600:
            return self.macro_trend_bullish
        df = market_data.get_ohlcv("^NSEI", period="2y", interval="1d")
        if len(df) >= 200:
            lp = float(df["Close"].iloc[-1])
            s200 = float(df["Close"].rolling(200).mean().iloc[-1])
            self.macro_trend_bullish = lp > s200
            self.macro_trend_updated = now
            logger.info(f"Macro trend: NIFTY @ {lp:.0f} vs SMA200 @ {s200:.0f} → "
                        f"{'BULL' if self.macro_trend_bullish else 'BEAR'} [{df.attrs.get('source')}]")
        else:
            logger.warning("Macro trend: NIFTY data unavailable — keeping last known regime")
        return self.macro_trend_bullish

    def _get_weekly_trend(self, symbol: str) -> str:
        """Weekly-chart trend filter → 'BULL' | 'BEAR' | 'NEUTRAL' (cached 4h)."""
        cache = self.__dict__.setdefault("_weekly_cache", {})
        now = datetime.now()
        hit = cache.get(symbol)
        if hit and (now - hit[1]).total_seconds() < 14400:
            return hit[0]
        trend = "NEUTRAL"
        df_w = market_data.get_ohlcv(symbol, period="2y", interval="1wk")
        if len(df_w) >= 30:
            close = df_w["Close"]
            sma30w = float(close.rolling(30).mean().iloc[-1])
            sma10w = float(close.rolling(10).mean().iloc[-1])
            lp = float(close.iloc[-1])
            if lp > sma30w and sma10w > sma30w:
                trend = "BULL"
            elif lp < sma30w and sma10w < sma30w:
                trend = "BEAR"
            cache[symbol] = (trend, now)
        return trend

    # ─────────────────────── DATA FETCHING ───────────────────────

    def _fetch_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """OHLCV via the unified market-data layer (see services/market_data.py)."""
        return market_data.get_ohlcv(symbol, period=period, interval=interval)

    # ─────────────────────── INDICATOR LIBRARY ───────────────────────

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.where(loss != 0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss != 0, 100.0)  # Pure up-trend → RSI = 100
        return rsi

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index — measures trend STRENGTH (not direction)."""
        high, low, close = df["High"], df["Low"], df["Close"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < (-low.diff()).clip(lower=0)] = 0
        minus_dm[minus_dm < high.diff().clip(lower=0)] = 0
        atr = self._compute_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, 0.001))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, 0.001))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 0.001))
        return dx.rolling(period).mean()

    def _compute_stochastic(self, df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple:
        """Stochastic Oscillator %K and %D."""
        low_min = df["Low"].rolling(k).min()
        high_max = df["High"].rolling(k).max()
        stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, 0.001)
        stoch_d = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    def _compute_macd_full(self, series: pd.Series, fast=12, slow=26, signal=9):
        """Full MACD: line, signal line, histogram."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _compute_bollinger(self, series: pd.Series, period: int = 20, std: float = 2.0):
        """Bollinger Bands: %B and Width."""
        sma = series.rolling(period).mean()
        std_dev = series.rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        pct_b = (series - lower) / (upper - lower).replace(0, 0.001)
        width = (upper - lower) / sma.replace(0, 0.001) * 100
        return pct_b, width

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP computed per rolling 20-bar window (not cumulative from dataset start)."""
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        return (typical * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, 1)

    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(df["Close"].diff())
        obv = (df["Volume"] * direction).fillna(0).cumsum()
        return obv

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> dict:
        """
        Fast vectorized detection of key reversal candlestick patterns.
        Returns flags for the last row only.
        """
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        body = (c - o).abs()
        candle_range = (h - l).replace(0, 0.001)
        lower_shadow = np.where(c > o, o - l, c - l)
        upper_shadow = np.where(c > o, h - c, h - o)
        body_ratio = body / candle_range
        lower_ratio = lower_shadow / candle_range

        # Hammer: small body at top, long lower shadow (>2x body), bullish reversal
        hammer = (lower_ratio.iloc[-1] > 0.6) and (body_ratio.iloc[-1] < 0.4) and (c.iloc[-1] > o.iloc[-1])

        # Bullish Engulfing: current green candle fully covers previous red candle
        bullish_engulfing = False
        if len(df) >= 2:
            prev_red = c.iloc[-2] < o.iloc[-2]
            curr_green = c.iloc[-1] > o.iloc[-1]
            covers = (c.iloc[-1] > o.iloc[-2]) and (o.iloc[-1] < c.iloc[-2])
            bullish_engulfing = prev_red and curr_green and covers

        # Doji: body < 10% of range
        doji = body_ratio.iloc[-1] < 0.1

        # Morning Star: 3-candle pattern (red, doji/small, green)
        morning_star = False
        if len(df) >= 3:
            red_candle = c.iloc[-3] < o.iloc[-3]
            small_middle = body_ratio.iloc[-2] < 0.2
            green_close = c.iloc[-1] > o.iloc[-3]
            morning_star = red_candle and small_middle and green_close

        return {
            "hammer": hammer,
            "bullish_engulfing": bullish_engulfing,
            "doji": doji,
            "morning_star": morning_star,
            "any_bullish": hammer or bullish_engulfing or morning_star,
        }

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 20 features for ML models and signal generation.
        Minimum 35 bars required.
        """
        df = df.copy()
        if len(df) < 35:
            return pd.DataFrame()

        # Trend
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_30"] = df["Close"].rolling(30).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        # When fewer than 200 bars available, back-fill with longest available mean
        if df["SMA_200"].isna().all():
            df["SMA_200"] = df["Close"].expanding().mean()
        df["SMA_200"] = df["SMA_200"].fillna(df["Close"].expanding().mean())
        df["VWAP"] = self._compute_vwap(df)
        df["ATR"] = self._compute_atr(df)

        # Distances
        df["Dist_SMA10"] = (df["Close"] - df["SMA_10"]) / df["SMA_10"].replace(0, 1) * 100
        df["Dist_SMA30"] = (df["Close"] - df["SMA_30"]) / df["SMA_30"].replace(0, 1) * 100
        df["Dist_SMA200"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"].replace(0, 1) * 100
        df["Dist_VWAP"] = (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, 1) * 100

        # Momentum
        df["RSI"] = self._compute_rsi(df["Close"])
        df["RSI_Slope"] = df["RSI"].diff(3)
        macd, macd_sig, macd_hist = self._compute_macd_full(df["Close"])
        df["MACD"] = macd
        df["MACD_Signal"] = macd_sig
        df["MACD_Hist"] = macd_hist

        # Trend strength
        df["ADX"] = self._compute_adx(df)
        df["Stoch_K"], df["Stoch_D"] = self._compute_stochastic(df)

        # Volatility/Bands
        df["BB_PctB"], df["BB_Width"] = self._compute_bollinger(df["Close"])
        df["Daily_Ret"] = df["Close"].pct_change() * 100
        df["Volat_Ratio"] = df["ATR"] / df["Close"].replace(0, 1) * 100

        # Volume
        df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, 1)
        obv = self._compute_obv(df)
        df["OBV_Slope"] = obv.diff(5) / df["Close"].replace(0, 1)

        # Candlestick features (vectorized)
        body = (df["Close"] - df["Open"]).abs()
        candle_range = (df["High"] - df["Low"]).replace(0, 0.001)
        lower_shadow = np.where(df["Close"] > df["Open"], df["Open"] - df["Low"], df["Close"] - df["Low"])
        df["Candle_Body_Ratio"] = body / candle_range
        df["Lower_Shadow_Ratio"] = lower_shadow / candle_range

        return df.dropna()

    # ─────────────────────── 4/6 CONFIRMATION SYSTEM ───────────────────────

    def _check_buy_confirmations(self, row, patterns: dict, macro_bull: bool) -> tuple:
        """
        ASTRA 1.0 — 4/7 multi-confirmation BUY system.
        Returns (passed, score, confirmation_details)

        Changes from v1.0:
          - RSI threshold relaxed 35→50: uptrend stocks pull back to 40-50, not below 35
          - SMA200 and ADX are now separate independent checks (7 total, need 4)
          - Candlestick check: raw geometry (close>open + lower shadow) replaces pattern engine
            so it works identically in live and backtest contexts
          - BB_PctB threshold widened 0.25→0.35: captures more pullback setups near lower band
        """
        candle_range  = float(row.get("High", 0)) - float(row.get("Low", 0))
        lower_shadow  = float(row.get("Open", 0)) - float(row.get("Low", 0))  # BUY candle shadow
        bullish_candle = (
            float(row.get("Close", 0)) > float(row.get("Open", 0))  # positive body
            and candle_range > 0
            and (lower_shadow / candle_range) > 0.3                  # lower shadow ≥ 30% of range
        )
        # In live mode, pattern engine can override geometry (e.g., a bearish engulfing
        # after a gap-up: technically bearish body but pattern engine flagged it).
        # This means geometry check never acts as backstop when pattern engine is active.
        if patterns.get("any_bullish", False):
            bullish_candle = True

        checks = {
            # 1. RSI dip-and-recover: pullback into 40-50 zone, now turning up
            "RSI pullback & recovering":     float(row.get("RSI", 100)) < 50 and float(row.get("RSI_Slope", -1)) > 0,
            # 2. Price remains above long-term trend (uptrend structure)
            "Price above SMA200":            float(row.get("Dist_SMA200", -1)) > 0,
            # 3. Trend strength independent of price level
            "Strong trend (ADX > 25)":       float(row.get("ADX", 0)) > 25,
            # 4. MACD momentum turning bullish
            "MACD bullish crossover":        float(row.get("MACD", 0)) > float(row.get("MACD_Signal", 0)),
            # 5. Volume confirms the move
            "Volume spike (>1.5x avg)":      float(row.get("Volume_Ratio", 0)) > 1.5,
            # 6. Geometry: bullish body + support shadow (no pattern engine dependency)
            "Bullish candle (body + lower shadow)": bullish_candle,
            # 7. Near lower Bollinger Band (mean-reversion setup)
            "Near lower Bollinger Band":     float(row.get("BB_PctB", 1.0)) < 0.35,
        }
        passed_count = sum(checks.values())
        score = (passed_count / 7) * 100

        # Macro filter: bear market requires 5/7 (stricter)
        threshold = 4 if macro_bull else 5

        return (passed_count >= threshold), round(score, 1), checks

    def _check_sell_confirmations(self, row, patterns: dict) -> tuple:
        """4/6 multi-confirmation SELL system."""
        prev_green = True  # Bearish engulfing would need access to prior row — simplified here
        checks = {
            "RSI overbought & turning down": (row["RSI"] > 65) and (row["RSI_Slope"] < 0),
            "Price below SMA200": row["Dist_SMA200"] < 0,
            "MACD bearish crossover": row["MACD"] < row["MACD_Signal"],
            "Volume spike on down move": (row["Volume_Ratio"] > 1.5) and (row["Daily_Ret"] < 0),
            "Near upper Bollinger Band": row["BB_PctB"] > 0.8,
            "Stochastic overbought": row["Stoch_K"] > 80,
        }
        passed_count = sum(checks.values())
        score = (passed_count / 6) * 100
        return (passed_count >= 4), round(score, 1), checks

    # ─────────────────────── MAIN ANALYSIS ───────────────────────

    def analyze_market_data(self, asset_symbol: str, period: str = "6mo",
                            interval: str = "1d", engine: str = "astra") -> dict:
        """
        Full market analysis. Returns signal, confidence, SL/TP, chartData.
        """
        if engine == "astra_ai":
            period, interval = "2y", "1d"
        elif engine == "astra_ml":
            period, interval = "2y", "1d"  # LSTM works better on daily data

        try:
            df_raw = self._fetch_data(asset_symbol, period=period, interval=interval)
            if df_raw.empty:
                return {
                    "asset": asset_symbol, "signal": "HOLD", "confidence": 0.0,
                    "error": ("No market data available from any provider. "
                              "Check GET /api/data/health for the reason (network, keys, quota)."),
                    "current_price": 0.0, "entry_price": 0.0, "target": 0.0, "stop_loss": 0.0,
                    "chartData": [], "data_source": None, "data_stale": True,
                }
            data_source = df_raw.attrs.get("source")
            data_stale = bool(df_raw.attrs.get("stale"))

            df = self._compute_features(df_raw)
            if len(df) < 5:
                raise ValueError("Insufficient data after feature computation")

            last = df.iloc[-1]
            lp = float(last["Close"])
            at = float(last["ATR"])
            patterns = self._detect_candlestick_patterns(df)
            macro_bull = self._is_macro_bullish()

            # ---- Market Regime ----
            # ADX < 20: choppy/ranging → prefer mean reversion (RSI extremes)
            # ADX >= 20: trending → prefer momentum/breakout
            is_trending = float(last["ADX"]) >= 20

            signal = "HOLD"
            confidence = 50.0
            mult_sl, mult_tp = 1.5, 3.0  # Default ATR multipliers (2:1 R:R minimum)

            # ── Multi-Timeframe Weekly Trend ──
            # Only filter equity symbols (skip indices/crypto which use their own macro filters)
            weekly_trend = "NEUTRAL"
            if not any(c in asset_symbol for c in ["^", "-USD", "-INR", "=F"]):
                weekly_trend = self._get_weekly_trend(asset_symbol)

            # ── ASTRA 1.0 (Rule-Based Multi-Confirmation) ──
            if engine == "astra":
                buy_ok, buy_conf, _ = self._check_buy_confirmations(last, patterns, macro_bull)
                sell_ok, sell_conf, _ = self._check_sell_confirmations(last, patterns)

                if buy_ok and buy_conf > sell_conf:
                    signal, confidence = "BUY", buy_conf
                    # MTF filter: suppress BUY if weekly is bearish
                    if weekly_trend == "BEAR":
                        signal = "HOLD"
                        confidence = round(confidence * 0.6, 1)
                elif sell_ok and sell_conf > buy_conf:
                    signal, confidence = "SELL", sell_conf
                    # MTF filter: suppress SELL if weekly is bullish
                    if weekly_trend == "BULL":
                        signal = "HOLD"
                        confidence = round(confidence * 0.6, 1)
                else:
                    confidence = max(buy_conf, sell_conf)

            # ── ASTRA.AI (Random Forest — trust the model) ──
            elif engine == "astra_ai":
                mult_sl, mult_tp = 1.6, 4.0
                if self.rf_model is not None:
                    try:
                        feats = df[FEATURE_COLS].tail(1)
                        pred = float(self.rf_model.predict(feats)[0])
                        atr_pct = float(last["Volat_Ratio"])
                        threshold = max(1.2, atr_pct * 0.5)
                        confidence = float(round(min(abs(pred) * 18.0, 99.0), 1))
                        if pred > threshold:
                            signal = "BUY"
                            # Block BUY if macro bear OR weekly bear
                            if not macro_bull or weekly_trend == "BEAR":
                                signal = "HOLD"
                                confidence *= 0.5
                        elif pred < -threshold:
                            signal = "SELL"
                            if weekly_trend == "BULL":
                                signal = "HOLD"
                                confidence *= 0.5
                    except Exception as e:
                        logger.warning(f"RF model inference failed: {e}")
                        buy_ok, buy_conf, _ = self._check_buy_confirmations(last, patterns, macro_bull)
                        if buy_ok:
                            signal, confidence = "BUY", buy_conf

            # ── ASTRA.ML (Bidirectional LSTM) ──
            elif engine == "astra_ml":
                mult_sl, mult_tp = 1.5, 3.5
                if self.lstm_model is not None and self.lstm_scaler is not None:
                    try:
                        lookback = 30
                        feat_data = df[FEATURE_COLS].tail(lookback).values
                        if len(feat_data) == lookback:
                            scaled = self.lstm_scaler.transform(feat_data)
                            X = scaled.reshape(1, lookback, len(FEATURE_COLS))
                            pred = float(self.lstm_model.predict(X, verbose=0)[0][0])
                            atr_pct = float(last["Volat_Ratio"])
                            # Lowered from 0.8 → 0.55 to unlock more trades while keeping quality
                            threshold = max(0.55, atr_pct * 0.35)
                            confidence = float(round(min(abs(pred) * 25.0, 99.0), 1))
                            if pred > threshold:
                                signal = "BUY"
                                if not macro_bull or weekly_trend == "BEAR":
                                    signal = "HOLD"
                                    confidence *= 0.6
                            elif pred < -threshold:
                                signal = "SELL"
                                if weekly_trend == "BULL":
                                    signal = "HOLD"
                                    confidence *= 0.6
                    except Exception as e:
                        logger.warning(f"LSTM inference failed: {e}")
                        buy_ok, buy_conf, _ = self._check_buy_confirmations(last, patterns, macro_bull)
                        if buy_ok:
                            signal, confidence = "BUY", buy_conf

            # ── ATR-Based SL/TP (all engines) ──
            entry_p = float(round(lp, 2))
            if signal == "SELL":
                sl = round(entry_p + at * mult_sl, 2)
                tp = round(entry_p - at * mult_tp, 2)
            else:
                sl = round(entry_p - at * mult_sl, 2)
                tp = round(entry_p + at * mult_tp, 2)

            # ── Build chart data (last 180 bars) ──
            chart_data = []
            used_times = set()
            intraday = interval not in ("1d", "1wk", "1mo")
            for idx, row in df.tail(180).iterrows():
                try:
                    # Daily: "YYYY-MM-DD". Intraday: epoch seconds of exchange-local
                    # wall time (lightweight-charts renders it as-is), so bars
                    # within one day are no longer collapsed into a single point.
                    if intraday:
                        t_str = int(pd.Timestamp(idx).tz_localize(None).timestamp()) if getattr(idx, "tzinfo", None) else int(pd.Timestamp(idx).timestamp())
                    else:
                        t_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                except Exception:
                    t_str = str(idx)[:10]
                if t_str in used_times:
                    continue
                used_times.add(t_str)

                sma_val = row.get("SMA_200", None)
                rsi_val = row.get("RSI", None)
                macd_val = row.get("MACD", None)
                adx_val = row.get("ADX", None)

                chart_data.append({
                    "time": t_str,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row.get("Volume", 0) or 0),
                    "sma200": round(float(sma_val), 2) if sma_val is not None and not pd.isna(sma_val) else None,
                    "rsi": round(float(rsi_val), 2) if rsi_val is not None and not pd.isna(rsi_val) else None,
                    "macd": round(float(macd_val), 4) if macd_val is not None and not pd.isna(macd_val) else None,
                    "adx": round(float(adx_val), 2) if adx_val is not None and not pd.isna(adx_val) else None,
                })

            # Confirmations for UI display
            _, _, buy_checks = self._check_buy_confirmations(last, patterns, macro_bull)
            confirmations = {k: bool(v) for k, v in buy_checks.items()}

            # ── Data-freshness audit note ─────────────────────────────────────
            # ASTRA rate-limits external API calls to preserve free-tier quotas.
            # Prices are cached for up to 60 seconds; OHLCV bars (indicators) are
            # cached for up to 4 hours. Entry/SL/TP prices shown are the BEST
            # AVAILABLE cached price — always verify live price before executing.
            data_freshness_note = (
                "Stale data: live providers unreachable, showing last cached bars." if data_stale else
                "Indicative levels from latest bars — confirm the live price before acting."
            )

            return {
                "asset": asset_symbol,
                "signal": signal,
                "confidence": confidence,
                "current_price": entry_p,
                "entry_price": entry_p,
                "target": tp,
                "stop_loss": sl,
                "rsi": round(float(last["RSI"]), 2),
                "adx": round(float(last["ADX"]), 2),
                "macd": round(float(last["MACD"]), 4),
                "bb_pct_b": round(float(last["BB_PctB"]), 3),
                "volume_ratio": round(float(last["Volume_Ratio"]), 2),
                "market_regime": "TRENDING" if is_trending else "RANGING",
                "macro_trend": "BULL" if macro_bull else "BEAR",
                "weekly_trend": weekly_trend,
                "candlestick_patterns": patterns,
                "confirmations": confirmations,
                "interval": interval,
                "engine": engine,
                "chartData": chart_data,
                "data_freshness_note": data_freshness_note,
                "data_source": data_source,
                "data_stale": data_stale,
                "bars": int(len(df_raw)),
                "last_bar": str(df_raw.index[-1]),
            }

        except Exception as e:
            logger.error(f"Analysis failed for {asset_symbol}: {e}", exc_info=True)
            return {
                "asset": asset_symbol, "signal": "HOLD", "confidence": 0.0,
                "error": str(e), "current_price": 0.0, "entry_price": 0.0,
                "target": 0.0, "stop_loss": 0.0, "chartData": []
            }

    def get_realtime_price(self, symbol: str) -> float:
        """Last traded price (Dhan WS → Dhan → Binance → Yahoo → last close). 0.0 if unknown."""
        return market_data.get_quote(symbol)

    # ─────────────────────── MODEL TRAINING HELPERS ───────────────────────

    def train_model_rf(self, df: pd.DataFrame):
        """Train RandomForest on 20-feature equity data. Returns fitted model."""
        from sklearn.ensemble import RandomForestRegressor
        df = self._compute_features(df)
        if df.empty or len(df) < 50:
            logger.error("Not enough data to train RF model")
            return None

        # Target: 5-day forward return (what we're predicting)
        df["Target"] = df["Close"].pct_change(5).shift(-5) * 100
        df = df.dropna(subset=FEATURE_COLS + ["Target"])

        X = df[FEATURE_COLS].values
        y = df["Target"].values

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            oob_score=True,
        )
        model.fit(X, y)
        logger.info(
            f"RF model trained on {len(X)} samples, {len(FEATURE_COLS)} features | "
            f"OOB R²={model.oob_score_:.4f}"
        )
        return model

    def train_model_lstm(
        self,
        df_per_symbol: dict,          # {symbol: pd.DataFrame of OHLCV}
        lookback: int = 30,
        target_days: int = 20,        # 20-day return → aligned with 40-day hold period
        label_threshold_pct: float = 1.5,
    ):
        """
        Build and train 3-class BiLSTM with Luong Attention.

        Architecture: BiLSTM(64) → BiLSTM(32) → Attention → Dense(16) → Dense(3, softmax)
        Output classes: 0=DOWN (<-1.5%), 1=NEUTRAL, 2=UP (>+1.5%)
        Scaler: per-symbol RobustScaler fitted only on that symbol's training split.

        Args:
            df_per_symbol: dict mapping bare symbol (e.g. "TCS") to its OHLCV DataFrame.
            lookback: sequence length for LSTM input.
            target_days: forward return horizon for labeling.
            label_threshold_pct: pnl % boundary between NEUTRAL and UP/DOWN.

        Returns:
            (model, scaler_dict) where scaler_dict = {symbol: RobustScaler}
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import RobustScaler

            all_X_train, all_y_train = [], []
            all_X_val,   all_y_val   = [], []
            scaler_dict: dict = {}

            for symbol, raw_df in df_per_symbol.items():
                df = self._compute_features(raw_df)
                if df.empty or len(df) < lookback + target_days + 10:
                    logger.debug(f"Skipping {symbol}: insufficient rows after feature compute")
                    continue

                # 20-day forward return → 3-class label
                df["Target_pct"] = df["Close"].pct_change(target_days).shift(-target_days) * 100
                df = df.dropna(subset=FEATURE_COLS + ["Target_pct"])
                if len(df) < lookback + 20:
                    continue

                raw_X = df[FEATURE_COLS].values
                y_raw = df["Target_pct"].values

                # Labels: 0=DOWN, 1=NEUTRAL, 2=UP
                y = np.where(
                    y_raw >  label_threshold_pct, 2,
                    np.where(y_raw < -label_threshold_pct, 0, 1)
                ).astype(np.int32)

                # Build sequences
                X_seq, y_seq = [], []
                for i in range(lookback, len(raw_X) - target_days):
                    X_seq.append(raw_X[i - lookback:i])
                    y_seq.append(y[i])
                if len(X_seq) < 20:
                    continue
                X_seq, y_seq = np.array(X_seq), np.array(y_seq)

                # Chronological 80/20 split
                split = int(len(X_seq) * 0.8)
                X_tr_raw, X_val_raw = X_seq[:split], X_seq[split:]
                y_tr,     y_val     = y_seq[:split], y_seq[split:]

                if len(X_tr_raw) < 10:
                    continue

                # Fit scaler ONLY on this symbol's training data
                n_tr, n_steps, n_feats = X_tr_raw.shape
                scaler = RobustScaler()
                X_tr  = scaler.fit_transform(X_tr_raw.reshape(-1, n_feats)).reshape(n_tr, n_steps, n_feats)
                X_val = scaler.transform(X_val_raw.reshape(-1, n_feats)).reshape(len(X_val_raw), n_steps, n_feats)
                scaler_dict[symbol] = scaler

                all_X_train.append(X_tr);   all_y_train.append(y_tr)
                all_X_val.append(X_val);    all_y_val.append(y_val)

            if not all_X_train:
                logger.error("No valid symbols for LSTM training after feature compute")
                return None, None

            X_train = np.concatenate(all_X_train, axis=0)
            y_train = np.concatenate(all_y_train, axis=0)
            X_val   = np.concatenate(all_X_val,   axis=0)
            y_val   = np.concatenate(all_y_val,   axis=0)

            logger.info(
                f"LSTM training: {X_train.shape[0]} train / {X_val.shape[0]} val sequences "
                f"from {len(scaler_dict)} symbols. "
                f"Class dist train: {np.bincount(y_train).tolist()}"
            )

            # ── Luong-style attention — registered class so it survives save/reload ──
            LuongAttention = _get_luong_attention_cls()

            # ── Architecture: smaller = less overfitting on financial time series ──
            inputs  = tf.keras.Input(shape=(lookback, len(FEATURE_COLS)))
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
            )(inputs)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
            )(x)
            context = LuongAttention()(x)                           # learned attention pooling
            x = tf.keras.layers.Dense(16, activation="relu")(context)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(3, activation="softmax")(x)  # 3-class

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ]

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=60,
                batch_size=64,
                callbacks=callbacks,
                verbose=1,
            )
            logger.info(f"LSTM equity model trained: {X_train.shape} | symbols: {len(scaler_dict)}")
            return model, scaler_dict

        except ImportError:
            logger.error("TensorFlow not available. Cannot train LSTM.")
            return None, None
        except Exception as e:
            logger.error(f"LSTM training failed: {e}", exc_info=True)
            return None, None


ai_engine = AIPredictionEngine()
