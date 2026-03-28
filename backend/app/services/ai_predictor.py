import logging
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MARKET_PROFILES = {
    "DEFAULT": {"atr_multiplier_sl": 1.5, "atr_multiplier_tp": 4.0},
    "CRYPTO":  {"atr_multiplier_sl": 2.0, "atr_multiplier_tp": 6.0},
    "STOCKS": {".NS": {"sl": 1.5, "tp": 4.5}, "US": {"sl": 1.2, "tp": 3.6}},
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")

class AIPredictionEngine:
    def __init__(self):
        logger.info("Initialized Three-Pillar AI Prediction Engine")
        self.rf_model = None
        self.ann_model = None
        self.macro_trend_bullish = True
        self.macro_trend_updated = None
        self._load_models()

    def _load_models(self):
        try:
            rf_path = os.path.join(MODELS_DIR, "astra_rf.joblib")
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("Loaded pre-trained Astra.ai 1.0 (RF) model.")
            
            ann_path = os.path.join(MODELS_DIR, "astra_ann.joblib")
            if os.path.exists(ann_path):
                self.ann_model = joblib.load(ann_path)
                logger.info("Loaded pre-trained Astra.ml (ANN) model.")
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")

    def _is_macro_bullish(self) -> bool:
        now = datetime.now()
        if self.macro_trend_updated and (now - self.macro_trend_updated).total_seconds() < 3600:
            return self.macro_trend_bullish
            
        try:
            df = yf.download("^NSEI", period="1y", interval="1d", progress=False, timeout=5)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                lp = float(df.iloc[-1]["Close"])
                s200 = float(df["Close"].rolling(200).mean().iloc[-1])
                self.macro_trend_bullish = lp > s200
                self.macro_trend_updated = now
        except Exception:
            pass
        return self.macro_trend_bullish

    def _fetch_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        df = None
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
        except: pass
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
            
        if ".NS" not in symbol and "^" not in symbol and "=" not in symbol:
            try:
                df = yf.download(symbol + ".NS", period=period, interval=interval, auto_adjust=True, progress=False, timeout=10)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    return df
            except: pass
        
        raise ValueError(f"CRITICAL: Failed to fetch real market data for {symbol}. Circuit breaker triggered.")

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, 1)

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < 35: return pd.DataFrame()
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_30"] = df["Close"].rolling(30).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean() if len(df) >= 200 else df["SMA_30"]
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["RSI"] = self._compute_rsi(df["Close"], 14)
        df["ATR"] = self._compute_atr(df)
        df["VWAP"] = self._compute_vwap(df)
        df["Dist_SMA10"]  = (df["Close"] - df["SMA_10"]) / df["SMA_10"].replace(0, 1) * 100
        df["Dist_SMA30"]  = (df["Close"] - df["SMA_30"]) / df["SMA_30"].replace(0, 1) * 100
        df["Dist_SMA200"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"].replace(0, 1) * 100
        df["Dist_VWAP"]   = (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, 1) * 100
        df["Daily_Ret"]   = df["Close"].pct_change() * 100
        df["Volat_Ratio"] = df["ATR"] / df["Close"].replace(0, 1) * 100
        return df.dropna()

    def analyze_market_data(self, asset_symbol: str, period: str = "6mo", interval: str = "1d", engine: str = "astra") -> dict:
        if engine == "astra_ai": period, interval = "2y", "1d"
        elif engine == "astra_ml": period, interval = "2y", "1h"
        
        try:
            df = self._fetch_data(asset_symbol, period=period, interval=interval)
            raw_candles = df.copy()
            df = self._compute_features(df)
            if len(df) < 5: raise ValueError("Insufficient features data")

            lp = float(df.iloc[-1]["Close"])
            lr = float(df.iloc[-1]["RSI"])
            s2 = float(df.iloc[-1]["SMA_200"])
            at = float(df.iloc[-1]["ATR"])
            is_up = lp > s2
            
            signal, confidence = "HOLD", 50.0
            mult_sl, mult_tp = 2.0, 4.0

            feats = ["Dist_SMA10", "Dist_SMA30", "Dist_SMA200", "Dist_VWAP", "RSI", "MACD", "Daily_Ret", "Volat_Ratio"]

            if engine == "astra":
                if lr < 35 and is_up: signal = "BUY"
                elif lr > 65 and not is_up: signal = "SELL"
                confidence = 75.0
                if signal == "BUY" and not self._is_macro_bullish(): signal = "HOLD"
            elif engine == "astra_ai":
                mult_sl, mult_tp = 1.6, 4.2
                if self.rf_model:
                    pred = float(self.rf_model.predict(df[feats].tail(1))[0])
                    confidence = float(round(min(abs(pred) * 20.0, 99.0), 1))
                    if pred > 1.2: signal = "BUY"
                    elif pred < -1.2: signal = "SELL"
                if signal == "HOLD" and lr < 35 and is_up: signal = "BUY"
                if signal == "BUY" and not self._is_macro_bullish(): signal = "HOLD"
            elif engine == "astra_ml":
                mult_sl, mult_tp = 1.5, 3.0
                if self.ann_model:
                    X_sc = self.ann_model.scaler.transform(df[feats].tail(1))
                    pred = float(self.ann_model.predict(X_sc)[0])
                    confidence = float(round(min(abs(pred) * 25.0, 99.0), 1))
                    if pred > 0.8: signal = "BUY"
                    elif pred < -1.0: signal = "SELL"
                if signal == "HOLD" and lr < 38: signal = "BUY"
                if signal == "BUY" and not self._is_macro_bullish(): signal = "HOLD"

            entry_p = float(round(lp, 2))
            if engine == "astra":
                m = 0.01 if signal == "BUY" else -0.01
                sl, tp = round(entry_p * (1.0 - m), 2), round(entry_p * (1.0 + m*2.0), 2)
            else:
                if signal == "SELL":
                    sl, tp = round(entry_p + at*mult_sl, 2), round(entry_p - at*mult_tp, 2)
                else:
                    sl, tp = round(entry_p - at*mult_sl, 2), round(entry_p + at*mult_tp, 2)

            return {
                "asset": asset_symbol, "signal": signal, "confidence": confidence,
                "current_price": entry_p, "entry_price": entry_p, "target": tp, "stop_loss": sl,
                "rsi": round(lr, 2), "interval": interval, "engine": engine,
                "prices": [round(float(p), 2) for p in raw_candles["Close"].tolist()[-180:]]
            }
        except Exception as e:
            logger.error(f"Analysis failed for {asset_symbol}: {e}")
            return {"asset": asset_symbol, "signal": "HOLD", "error": str(e)}

    def get_realtime_price(self, symbol: str) -> float:
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.fast_info.get("lastPrice")
            if price: return round(float(price), 2)
            df = ticker.history(period="1d")
            if not df.empty: return round(float(df.iloc[-1]["Close"]), 2)
        except: pass
        return 0.0

ai_engine = AIPredictionEngine()
