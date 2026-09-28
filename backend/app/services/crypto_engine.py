"""
ASTRA Crypto Engine v1.0
=========================
Dedicated engine for cryptocurrency markets — fundamentally separate from the equity engine.

Key design decisions:
- CCXT Binance feed for international pairs (BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD)
- yfinance .INR pairs for Indian crypto (BTC-INR, ETH-INR via WazirX-like symbols)
- Indian/International market toggle with different strategy logic
- Crypto-specific ATR multipliers (2.5x SL, 5.0x TP) for high volatility
- Regime-aware: Momentum/Breakout in trending (ADX>25), Mean Reversion in ranging (ADX<20)
- Fear & Greed Index as macro filter
- Donchian Channel breakout for momentum regime
- Bollinger Band squeeze for ranging regime
- Separate Bidirectional LSTM for crypto (trained on BTC + multi-coin features)
"""

import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
import requests

from datetime import datetime, timedelta
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")

# ─────────────────────── ASSET REGISTRIES ───────────────────────

INTERNATIONAL_CRYPTO = {
    "BTC-USD":  {"name": "Bitcoin",   "ccxt_pair": "BTC/USDT", "risk_class": "major"},
    "ETH-USD":  {"name": "Ethereum",  "ccxt_pair": "ETH/USDT", "risk_class": "major"},
    "BNB-USD":  {"name": "BNB",       "ccxt_pair": "BNB/USDT", "risk_class": "major"},
    "SOL-USD":  {"name": "Solana",    "ccxt_pair": "SOL/USDT", "risk_class": "altcoin"},
    "XRP-USD":  {"name": "XRP",       "ccxt_pair": "XRP/USDT", "risk_class": "altcoin"},
    "ADA-USD":  {"name": "Cardano",   "ccxt_pair": "ADA/USDT", "risk_class": "altcoin"},
    "DOGE-USD": {"name": "Dogecoin",  "ccxt_pair": "DOGE/USDT","risk_class": "meme"},
    "AVAX-USD": {"name": "Avalanche", "ccxt_pair": "AVAX/USDT","risk_class": "altcoin"},
}

INDIAN_CRYPTO = {
    "BTC-INR":  {"name": "Bitcoin (INR)",   "yf_symbol": "BTC-INR",  "risk_class": "major"},
    "ETH-INR":  {"name": "Ethereum (INR)",  "yf_symbol": "ETH-INR",  "risk_class": "major"},
    "SOL-INR":  {"name": "Solana (INR)",    "yf_symbol": "SOL-INR",  "risk_class": "altcoin"},
    "XRP-INR":  {"name": "XRP (INR)",       "yf_symbol": "XRP-INR",  "risk_class": "altcoin"},
    "SHIB-INR": {"name": "Shiba Inu (INR)", "yf_symbol": "SHIB-INR", "risk_class": "meme"},
}

# ATR multipliers by risk class
RISK_CONFIG = {
    "major":   {"sl_mult": 2.0, "tp_mult": 4.5, "conf_threshold": 0.8},
    "altcoin": {"sl_mult": 2.8, "tp_mult": 5.5, "conf_threshold": 0.85},
    "meme":    {"sl_mult": 3.5, "tp_mult": 7.0, "conf_threshold": 0.90},
}

# Feature columns for crypto LSTM (20 features — overlapping with equity but crypto-specific)
CRYPTO_FEATURE_COLS = [
    "Dist_SMA10", "Dist_SMA30", "Dist_SMA50",
    "RSI", "RSI_Slope",
    "MACD", "MACD_Signal", "MACD_Hist",
    "ADX",
    "Stoch_K", "Stoch_D",
    "BB_PctB", "BB_Width",
    "Donchian_PctB",
    "Daily_Ret", "Volat_Ratio",
    "Volume_Ratio", "OBV_Slope",
    "Candle_Body_Ratio", "Lower_Shadow_Ratio",
]


class CryptoEngine:
    def __init__(self):
        logger.info("Initialized ASTRA.CRYPTO Engine v1.0")
        self.exchange = None   # v1.13: Binance is reached via market_data (data-api.binance.vision + fallbacks)
        self.fear_greed_cache = None
        self.fear_greed_updated = None
        self.btc_dominance_cache = None
        self.btc_dom_updated = None
        self.lstm_model = None
        self.lstm_scaler = None
        self._load_models()

    # ─────────────────────── SETUP ───────────────────────

    def _init_ccxt(self):
        """Initialize CCXT Binance (public feed, no API key required for OHLCV)."""
        try:
            import ccxt
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            logger.info("CCXT Binance initialized (public feed)")
        except ImportError:
            logger.warning("CCXT not installed. International crypto data will use yfinance fallback.")
        except Exception as e:
            logger.error(f"CCXT init failed: {e}")

    def _load_models(self):
        try:
            lstm_path = os.path.join(MODELS_DIR, "astra_lstm_crypto.keras")
            scaler_path = os.path.join(MODELS_DIR, "astra_lstm_crypto_scaler.joblib")
            if os.path.exists(lstm_path) and os.path.exists(scaler_path):
                from app.services.model_compat import load_keras_model
                model = load_keras_model(lstm_path)
                # The engine expects a 3-class classifier [P_DOWN, P_NEUTRAL, P_UP].
                # The model file shipped in the repo is an older 1-output regressor;
                # feeding it to the classifier logic turned most predictions into SELL.
                if int(model.output_shape[-1]) != 3:
                    logger.warning(
                        f"Crypto LSTM has {model.output_shape[-1]} output(s), expected 3 — ignoring it "
                        "(rule-based crypto engine used). Retrain: python train_models.py")
                else:
                    self.lstm_model = model
                    self.lstm_scaler = joblib.load(scaler_path)
                    logger.info("Loaded ASTRA.CRYPTO LSTM model.")
        except ImportError:
            logger.warning("TensorFlow not available. CRYPTO LSTM disabled.")
        except Exception as e:
            logger.error(f"Crypto model load failed: {e}")

    # ─────────────────────── EXTERNAL DATA ───────────────────────

    def get_fear_greed(self) -> dict:
        """
        Fetch Fear & Greed Index from Alternative.me (free, no API key).
        Returns: {"value": 65, "label": "Greed", "updated": "..."}
        """
        now = datetime.now()
        if self.fear_greed_cache and self.fear_greed_updated:
            if (now - self.fear_greed_updated).total_seconds() < 3600:
                return self.fear_greed_cache
        try:
            res = requests.get(
                "https://api.alternative.me/fng/?limit=1&format=json",
                timeout=6
            )
            data = res.json()
            entry = data["data"][0]
            self.fear_greed_cache = {
                "value": int(entry["value"]),
                "label": entry["value_classification"],
                "updated": entry.get("timestamp", str(int(time.time()))),
            }
            self.fear_greed_updated = now
            return self.fear_greed_cache
        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
            # Remember the failure for 10 min so every analysis doesn't wait on a dead host
            self.fear_greed_cache = {"value": 50, "label": "Neutral (unavailable)", "updated": ""}
            self.fear_greed_updated = now - timedelta(seconds=3000)
            return self.fear_greed_cache

    def get_btc_dominance(self) -> float:
        """
        Fetch BTC dominance from CoinGecko (free tier, rate-limited).
        Returns float (e.g., 52.3 = 52.3%).
        """
        now = datetime.now()
        if self.btc_dominance_cache and self.btc_dom_updated:
            if (now - self.btc_dom_updated).total_seconds() < 3600:
                return self.btc_dominance_cache
        try:
            res = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=8
            )
            data = res.json()
            dom = data["data"]["market_cap_percentage"].get("btc", 50.0)
            self.btc_dominance_cache = round(float(dom), 2)
            self.btc_dom_updated = now
            return self.btc_dominance_cache
        except Exception as e:
            logger.debug(f"BTC dominance fetch failed: {e}")
            self.btc_dominance_cache = 50.0
            self.btc_dom_updated = now - timedelta(seconds=3000)
            return 50.0

    # ─────────────────────── DATA FETCHING ───────────────────────

    def fetch_data(self, symbol: str, market: str = "international",
                   timeframe: str = "1d", limit: int = 500) -> pd.DataFrame:
        """
        OHLCV for a crypto pair via the unified market-data layer.
        international (BTC-USD …): Binance public data → Yahoo → yfinance → Twelve Data
        indian (BTC-INR …):        Yahoo → yfinance
        """
        from app.services.market_data import market_data
        sym = INDIAN_CRYPTO.get(symbol, {}).get("yf_symbol", symbol) if market == "indian" else symbol
        per_day = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}.get(timeframe, 1)
        days = max(30, int(limit / per_day) + 5)
        df = market_data.get_ohlcv(sym, period=f"{days}d", interval=timeframe)
        if df.empty:
            logger.warning(f"⚠️  No crypto data for {sym} ({timeframe}) — see /api/data/health")
        return df

    # ─────────────────────── INDICATOR LIBRARY ───────────────────────

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low = df["High"], df["Low"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < (-low.diff()).clip(lower=0)] = 0
        minus_dm[minus_dm < high.diff().clip(lower=0)] = 0
        atr = self._compute_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, 0.001))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, 0.001))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 0.001))
        return dx.rolling(period).mean()

    def _compute_stochastic(self, df: pd.DataFrame, k: int = 14, d: int = 3):
        low_min = df["Low"].rolling(k).min()
        high_max = df["High"].rolling(k).max()
        stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, 0.001)
        stoch_d = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    def _compute_macd_full(self, series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _compute_bollinger(self, series: pd.Series, period: int = 20, std: float = 2.0):
        sma = series.rolling(period).mean()
        std_dev = series.rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        pct_b = (series - lower) / (upper - lower).replace(0, 0.001)
        width = (upper - lower) / sma.replace(0, 0.001) * 100
        return pct_b, width

    def _compute_donchian(self, df: pd.DataFrame, period: int = 20):
        """
        Donchian Channel — key for breakout detection in trending crypto markets.
        %B: 0 = at lower band (support), 1 = at upper band (resistance/breakout)
        """
        upper = df["High"].rolling(period).max()
        lower = df["Low"].rolling(period).min()
        pct_b = (df["Close"] - lower) / (upper - lower).replace(0, 0.001)
        return pct_b, upper, lower

    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["Close"].diff())
        return (df["Volume"] * direction).fillna(0).cumsum()

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < 35:
            return pd.DataFrame()

        # Trend SMAs
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_30"] = df["Close"].rolling(30).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()

        df["Dist_SMA10"] = (df["Close"] - df["SMA_10"]) / df["SMA_10"].replace(0, 1) * 100
        df["Dist_SMA30"] = (df["Close"] - df["SMA_30"]) / df["SMA_30"].replace(0, 1) * 100
        df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"].replace(0, 1) * 100

        # Momentum
        df["RSI"] = self._compute_rsi(df["Close"])
        df["RSI_Slope"] = df["RSI"].diff(3)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = self._compute_macd_full(df["Close"])
        df["ADX"] = self._compute_adx(df)
        df["Stoch_K"], df["Stoch_D"] = self._compute_stochastic(df)

        # Volatility
        df["BB_PctB"], df["BB_Width"] = self._compute_bollinger(df["Close"])
        df["ATR"] = self._compute_atr(df)
        df["Donchian_PctB"], _, _ = self._compute_donchian(df)
        df["Daily_Ret"] = df["Close"].pct_change() * 100
        df["Volat_Ratio"] = df["ATR"] / df["Close"].replace(0, 1) * 100

        # Volume
        df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, 1)
        obv = self._compute_obv(df)
        df["OBV_Slope"] = obv.diff(5) / df["Close"].replace(0, 1)

        # Candlestick geometry
        body = (df["Close"] - df["Open"]).abs()
        candle_range = (df["High"] - df["Low"]).replace(0, 0.001)
        lower_shadow = np.where(df["Close"] > df["Open"], df["Open"] - df["Low"], df["Close"] - df["Low"])
        df["Candle_Body_Ratio"] = body / candle_range
        df["Lower_Shadow_Ratio"] = lower_shadow / candle_range

        return df.dropna()

    # ─────────────────────── STRATEGY LOGIC ───────────────────────

    def _strategy_momentum_breakout(self, last) -> tuple:
        """
        Momentum/Breakout strategy — for TRENDING markets (ADX > 25).
        Best for: BTC in bull runs, breakout plays.
        BUY when price breaks above Donchian upper channel with volume surge.
        """
        checks = {
            "Donchian breakout (price near upper band)": last["Donchian_PctB"] > 0.85,
            "MACD bullish crossover": last["MACD"] > last["MACD_Signal"],
            "RSI in momentum zone (50-75)": 50 < last["RSI"] < 75,
            "Volume surge (>2x avg)": last["Volume_Ratio"] > 2.0,
            "Above 50 SMA": last["Dist_SMA50"] > 0,
        }
        score = sum(checks.values()) / len(checks) * 100
        signal = "BUY" if score >= 60 else "HOLD"
        return signal, score, checks

    def _strategy_mean_reversion(self, last) -> tuple:
        """
        Mean Reversion strategy — for RANGING markets (ADX < 20).
        Best for: BTC in sideways chop, ETH consolidation.
        BUY at Bollinger Band lower extreme, SELL at upper extreme.
        """
        checks_buy = {
            "Near lower Bollinger Band": last["BB_PctB"] < 0.15,
            "RSI oversold (<30)": last["RSI"] < 30,
            "RSI turning up": last["RSI_Slope"] > 0,
            "Stochastic oversold (<20)": last["Stoch_K"] < 20,
            "OBV slope positive (buying pressure)": last["OBV_Slope"] > 0,
        }
        checks_sell = {
            "Near upper Bollinger Band": last["BB_PctB"] > 0.85,
            "RSI overbought (>70)": last["RSI"] > 70,
            "RSI turning down": last["RSI_Slope"] < 0,
            "Stochastic overbought (>80)": last["Stoch_K"] > 80,
        }
        buy_score = sum(checks_buy.values()) / len(checks_buy) * 100
        sell_score = sum(checks_sell.values()) / len(checks_sell) * 100

        if buy_score >= 60 and buy_score > sell_score:
            return "BUY", buy_score, checks_buy
        elif sell_score >= 60:
            return "SELL", sell_score, checks_sell
        return "HOLD", max(buy_score, sell_score), checks_buy

    def _apply_macro_filters(self, signal: str, fg_value: int,
                              btc_dom: float, symbol: str, market: str) -> tuple:
        """
        Apply macro-level filters:
        - Fear & Greed < 20 (Extreme Fear): Block new BUY signals
        - Fear & Greed > 90 (Extreme Greed): Block new BUY signals (reversal risk)
        - BTC Dominance rising + altcoin: Block altcoin BUYs (money flowing to BTC)
        - Indian market: Skip BTC dominance filter (not applicable to INR pairs)
        """
        reason = None
        risk_class = INTERNATIONAL_CRYPTO.get(symbol, INDIAN_CRYPTO.get(symbol, {})).get("risk_class", "altcoin")

        if signal == "BUY":
            if fg_value < 20:
                reason = f"Blocked: Extreme Fear (F&G={fg_value})"
                signal = "HOLD"
            elif fg_value > 90:
                reason = f"Blocked: Extreme Greed — reversal risk (F&G={fg_value})"
                signal = "HOLD"
            elif market == "international" and risk_class == "altcoin" and btc_dom > 58:
                reason = f"Blocked: BTC dominance high ({btc_dom}%) — altcoin headwind"
                signal = "HOLD"

        return signal, reason

    # ─────────────────────── MAIN ANALYSIS ───────────────────────

    def analyze(self, symbol: str, market: str = "international",
                timeframe: str = "1d", engine: str = "astra_crypto") -> dict:
        """
        Main crypto analysis entry point.
        market: "international" | "indian"
        engine: "astra_crypto" (rule-based), "astra_crypto_ml" (LSTM)
        """
        currency = "INR" if market == "indian" else "USD"
        asset_info = INTERNATIONAL_CRYPTO.get(symbol) or INDIAN_CRYPTO.get(symbol) or {}
        risk_class = asset_info.get("risk_class", "altcoin")
        risk_cfg = RISK_CONFIG[risk_class]

        try:
            df_raw = self.fetch_data(symbol, market=market, timeframe=timeframe)
            if df_raw.empty:
                return self._error_response(symbol, "Real-time crypto data currently unavailable from all sources (Binance/Yahoo).")

            df = self._compute_features(df_raw)
            if len(df) < 5:
                return self._error_response(symbol, "Insufficient data after feature computation")

            last = df.iloc[-1]
            lp = float(last["Close"])
            at = float(last["ATR"])

            # ── Fetch macro context ──
            fg = self.get_fear_greed()
            fg_value = fg["value"]
            btc_dom = self.get_btc_dominance() if market == "international" else 50.0

            # ── Market Regime Detection ──
            adx = float(last["ADX"])
            is_trending = adx >= 25
            is_ranging = adx < 20

            signal = "HOLD"
            confidence = 50.0
            strategy_used = "neutral"
            checks = {}

            if engine == "astra_crypto":
                if is_trending:
                    signal, confidence, checks = self._strategy_momentum_breakout(last)
                    strategy_used = "momentum_breakout"
                elif is_ranging:
                    signal, confidence, checks = self._strategy_mean_reversion(last)
                    strategy_used = "mean_reversion"
                else:
                    # Transitional — use lighter mean reversion
                    signal, confidence, checks = self._strategy_mean_reversion(last)
                    strategy_used = "transitional_mean_reversion"
                    confidence *= 0.8  # Reduce confidence in uncertain regime

            elif engine == "astra_crypto_ml":
                if self.lstm_model is not None and self.lstm_scaler is not None:
                    try:
                        lookback = 20  # Matches new classifier lookback
                        feat_data = df[CRYPTO_FEATURE_COLS].tail(lookback).values
                        if len(feat_data) == lookback:
                            scaled = self.lstm_scaler.transform(feat_data)
                            X = scaled.reshape(1, lookback, len(CRYPTO_FEATURE_COLS))
                            probs = self.lstm_model.predict(X, verbose=0)[0]  # [P_DOWN, P_NEUTRAL, P_UP]
                            pred_class = int(np.argmax(probs))
                            confidence = float(round(max(probs) * 100, 1))
                            # Only signal when model is confident (>50% on a class)
                            if pred_class == 2 and probs[2] > risk_cfg["conf_threshold"]:
                                signal = "BUY"
                            elif pred_class == 0 and probs[0] > risk_cfg["conf_threshold"]:
                                signal = "SELL"
                            strategy_used = "lstm_crypto_classifier"
                    except Exception as e:
                        logger.warning(f"Crypto LSTM failed: {e}. Falling back to rule-based.")
                        if is_trending:
                            signal, confidence, checks = self._strategy_momentum_breakout(last)
                        else:
                            signal, confidence, checks = self._strategy_mean_reversion(last)
                        strategy_used = "rule_based_fallback"
                else:
                    # LSTM not loaded — fall back to rule-based
                    if is_trending:
                        signal, confidence, checks = self._strategy_momentum_breakout(last)
                    else:
                        signal, confidence, checks = self._strategy_mean_reversion(last)
                    strategy_used = "rule_based_fallback"

            # ── Apply Macro Filters ──
            signal, filter_reason = self._apply_macro_filters(
                signal, fg_value, btc_dom, symbol, market
            )
            if filter_reason:
                confidence *= 0.5

            # ── ATR-based SL/TP (crypto-specific multipliers) ──
            mult_sl = risk_cfg["sl_mult"]
            mult_tp = risk_cfg["tp_mult"]
            entry_p = round(lp, 2)
            if signal == "SELL":
                sl = round(entry_p + at * mult_sl, 2)
                tp = round(entry_p - at * mult_tp, 2)
            else:
                sl = round(entry_p - at * mult_sl, 2)
                tp = round(entry_p + at * mult_tp, 2)

            # ── Build Chart Data ──
            chart_data = []
            used_times = set()
            intraday = timeframe not in ("1d", "1wk")
            for idx, row in df.tail(180).iterrows():
                # Daily → "YYYY-MM-DD"; intraday → epoch seconds (keeps every bar)
                t_date = int(pd.Timestamp(idx).timestamp()) if intraday else pd.Timestamp(idx).strftime("%Y-%m-%d")
                if t_date in used_times:
                    continue
                used_times.add(t_date)

                sma_val = row.get("SMA_50", None)
                rsi_val = row.get("RSI", None)
                bb_val = row.get("BB_PctB", None)

                chart_data.append({
                    "time": t_date,
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": float(row.get("Volume", 0) or 0),
                    "sma50": round(float(sma_val), 4) if sma_val is not None and not pd.isna(sma_val) else None,
                    "rsi": round(float(rsi_val), 2) if rsi_val is not None and not pd.isna(rsi_val) else None,
                    "bb_pct_b": round(float(bb_val), 3) if bb_val is not None and not pd.isna(bb_val) else None,
                })

            return {
                "asset": symbol,
                "name": asset_info.get("name", symbol),
                "market": market,
                "currency": currency,
                "risk_class": risk_class,
                "signal": signal,
                "confidence": round(confidence, 1),
                "current_price": entry_p,
                "entry_price": entry_p,
                "target": tp,
                "stop_loss": sl,
                "rsi": round(float(last["RSI"]), 2),
                "adx": round(float(last["ADX"]), 2),
                "bb_pct_b": round(float(last["BB_PctB"]), 3),
                "volume_ratio": round(float(last["Volume_Ratio"]), 2),
                "donchian_pct_b": round(float(last["Donchian_PctB"]), 3),
                "market_regime": "TRENDING" if is_trending else ("RANGING" if is_ranging else "TRANSITIONAL"),
                "strategy_used": strategy_used,
                "fear_greed": fg,
                "btc_dominance": btc_dom,
                "macro_filter_reason": filter_reason,
                "confirmations": {k: bool(v) for k, v in checks.items()},
                "atr": round(at, 4),
                "engine": engine,
                "chartData": chart_data,
                "data_freshness_note": (
                    "Stale data: live providers unreachable, showing last cached bars."
                    if df_raw.attrs.get("stale") else
                    "Indicative levels from latest bars — confirm the live price before acting."
                ),
                "data_source": df_raw.attrs.get("source"),
                "data_stale": bool(df_raw.attrs.get("stale")),
            }

        except Exception as e:
            logger.error(f"Crypto analysis failed for {symbol}: {e}", exc_info=True)
            return self._error_response(symbol, str(e))

    def get_watchlist_prices(self, market: str = "international") -> list:
        """Price + 24h change for the watchlist strip (parallel, cached, never hangs)."""
        from concurrent.futures import ThreadPoolExecutor
        from app.services.market_data import market_data
        assets = INTERNATIONAL_CRYPTO if market == "international" else INDIAN_CRYPTO

        def one(item):
            symbol, info = item
            sym = info.get("yf_symbol", symbol)
            row = {"symbol": symbol, "name": info["name"], "price": 0.0, "change_24h": 0.0,
                   "risk_class": info.get("risk_class", "major"),
                   "currency": "INR" if market == "indian" else "USD", "source": None}
            try:
                q = market_data.get_quote_info(sym)
                df = market_data.get_ohlcv(sym, period="7d", interval="1d")
                price = q["price"] or (float(df["Close"].iloc[-1]) if not df.empty else 0.0)
                if len(df) >= 2 and price:
                    prev = float(df["Close"].iloc[-2])
                    row["change_24h"] = round((price / prev - 1) * 100, 2) if prev else 0.0
                row["price"] = round(float(price), 4)
                row["source"] = q["source"] or df.attrs.get("source")
            except Exception as e:
                logger.debug(f"Watchlist price failed for {symbol}: {e}")
            return row

        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(one, assets.items()))

    def _error_response(self, symbol: str, error: str) -> dict:
        return {
            "asset": symbol, "signal": "HOLD", "confidence": 0.0,
            "error": error, "current_price": 0.0, "entry_price": 0.0,
            "target": 0.0, "stop_loss": 0.0,
            "fear_greed": {"value": 50, "label": "Neutral", "updated": ""},
            "btc_dominance": 50.0, "chartData": [],
        }

    # ─────────────────────── LSTM TRAINING ───────────────────────

    def train_lstm_crypto(self, symbols: list = None, lookback: int = 20):
        """
        Train dedicated Bidirectional LSTM for crypto.
        Trains on multiple coins simultaneously for generalization.

        Target: 3-class direction (0=DOWN, 1=NEUTRAL, 2=UP) using 1.5% threshold.
        Classification is far more stable than magnitude regression across regime changes.
        Shorter lookback (20 vs 30) produces more sequences from the same data.
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import RobustScaler

            if symbols is None:
                symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

            DIRECTION_THRESHOLD = 1.5  # % — moves below this are NEUTRAL

            all_X, all_y = [], []

            for sym in symbols:
                logger.info(f"Fetching crypto training data for {sym}...")
                df_raw = self.fetch_data(sym, market="international", timeframe="1d", limit=800)
                if df_raw.empty:
                    continue
                df = self._compute_features(df_raw)
                if len(df) < lookback + 20:
                    continue

                # Target: 3-class direction (DOWN=0, NEUTRAL=1, UP=2)
                fwd_return = df["Close"].pct_change(3).shift(-3) * 100
                df["Target"] = np.where(fwd_return > DIRECTION_THRESHOLD, 2,
                               np.where(fwd_return < -DIRECTION_THRESHOLD, 0, 1))
                df = df.dropna(subset=CRYPTO_FEATURE_COLS + ["Target"])
                df["Target"] = df["Target"].astype(int)

                for i in range(lookback, len(df) - 1):
                    all_X.append(df[CRYPTO_FEATURE_COLS].values[i - lookback:i])
                    all_y.append(df["Target"].values[i])

            if len(all_X) < 100:
                logger.error("Not enough crypto training data across symbols.")
                return None, None

            X_arr = np.array(all_X)
            y_arr = np.array(all_y)

            n_samples, n_steps, n_feats = X_arr.shape

            # Chronological split before scaling (no data leakage)
            split = int(n_samples * 0.8)
            X_train_raw, X_val_raw = X_arr[:split], X_arr[split:]
            y_train, y_val = y_arr[:split], y_arr[split:]

            # Fit scaler only on train data
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_feats)).reshape(split, n_steps, n_feats)
            X_val = scaler.transform(X_val_raw.reshape(-1, n_feats)).reshape(n_samples - split, n_steps, n_feats)

            # ── BiLSTM Classifier (3-class: DOWN/NEUTRAL/UP) ──
            inputs = tf.keras.Input(shape=(lookback, n_feats))
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3,
                                     recurrent_dropout=0.1,
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            )(inputs)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2,
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            )(x)
            context = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(32, activation="relu",
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(context)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(3, activation="softmax")(x)  # 3-class

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy"),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6),
            ]

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            logger.info(f"Crypto LSTM classifier trained on {len(X_train)} samples from {len(symbols)} coins")
            return model, scaler

        except ImportError:
            logger.error("TensorFlow not available.")
            return None, None
        except Exception as e:
            logger.error(f"Crypto LSTM training failed: {e}", exc_info=True)
            return None, None


crypto_engine = CryptoEngine()
