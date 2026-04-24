"""
ASTRA Commodities Engine
=========================
Generates signals for MCX commodities and international futures
using regime-aware technical analysis.

Instruments:
  MCX:  Gold (GC=F), Silver (SI=F), Crude Oil (CL=F),
        Natural Gas (NG=F), Copper (HG=F), Aluminium, Zinc
  COMEX: Same contracts mapped to yfinance symbols

Strategy:
  - Gold/Silver: Safe-haven regime filter (VIX > 20 → fear → gold BUY)
  - Crude:       Supply/demand regime (EIA inventory proxy via momentum)
  - Nat Gas:     Seasonal + momentum (winter → bullish, summer → bearish)
  - Base metals: Global growth proxy (corr with NIFTY 500)

Data Source: yfinance (commodity futures continuous contracts)
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

# ── Commodity universe ─────────────────────────────────────────────────────────

COMMODITIES = {
    # yfinance symbol → display name, asset class, MCX lot size (for reference)
    "GC=F":   {"name": "Gold",         "class": "precious",   "mcx_lot": 100,   "unit": "oz"},
    "SI=F":   {"name": "Silver",       "class": "precious",   "mcx_lot": 30000, "unit": "oz"},
    "CL=F":   {"name": "Crude Oil",    "class": "energy",     "mcx_lot": 100,   "unit": "bbl"},
    "NG=F":   {"name": "Natural Gas",  "class": "energy",     "mcx_lot": 1250,  "unit": "mmBtu"},
    "HG=F":   {"name": "Copper",       "class": "base",       "mcx_lot": 2500,  "unit": "lbs"},
    "ALI=F":  {"name": "Aluminium",    "class": "base",       "mcx_lot": 5000,  "unit": "lbs"},
    "ZC=F":   {"name": "Corn",         "class": "agri",       "mcx_lot": 5000,  "unit": "bu"},
    "ZW=F":   {"name": "Wheat",        "class": "agri",       "mcx_lot": 5000,  "unit": "bu"},
}

# Risk config per asset class
RISK_CONFIG = {
    "precious": {"sl_atr": 1.5, "tp_atr": 3.0, "conf_threshold": 0.60},
    "energy":   {"sl_atr": 2.0, "tp_atr": 4.0, "conf_threshold": 0.65},  # high vol
    "base":     {"sl_atr": 1.8, "tp_atr": 3.5, "conf_threshold": 0.58},
    "agri":     {"sl_atr": 1.5, "tp_atr": 3.0, "conf_threshold": 0.60},
}

IST = pytz.timezone("Asia/Kolkata")


class CommoditiesEngine:
    """
    Signal generator for commodity futures.
    Uses a 3-layer confirmation system:
      1. Macro regime filter (USD index, VIX proxy, global growth)
      2. Technical confluence (MACD + RSI + ATR trend)
      3. Seasonality bias (for energy and agri)
    """

    def __init__(self):
        self._cache: dict = {}   # {symbol: (ts, df)}
        self._cache_ttl = 3600   # 1-hour cache for commodity data

    # ── Data layer ─────────────────────────────────────────────────────────────

    def _fetch(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch commodity futures data with caching."""
        now = datetime.utcnow().timestamp()
        if symbol in self._cache:
            ts, df = self._cache[symbol]
            if now - ts < self._cache_ttl:
                return df

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            self._cache[symbol] = (now, df)
            return df
        except Exception as e:
            logger.error(f"Commodity data fetch failed for {symbol}: {e}")
            raise

    def _fetch_vix(self) -> float:
        """Fetch VIX (fear gauge) as macro filter."""
        try:
            vix = yf.Ticker("^VIX").history(period="5d")
            return float(vix["Close"].iloc[-1]) if not vix.empty else 20.0
        except Exception:
            return 20.0  # default to neutral

    def _fetch_dxy(self) -> float:
        """Fetch DXY (US Dollar Index) — inverse proxy for gold/silver."""
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
            if dxy.empty:
                return 103.0
            closes = dxy["Close"].dropna()
            if len(closes) < 2:
                return 103.0
            # Return DXY trend: positive = dollar strengthening (bearish for gold)
            return float(closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0.0
        except Exception:
            return 0.0

    # ── Feature computation ────────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on commodity OHLCV data."""
        df = df.copy()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # Trend
        df["SMA_20"]  = close.rolling(20).mean()
        df["SMA_50"]  = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean().fillna(close.expanding().mean())
        df["EMA_9"]   = close.ewm(span=9).mean()
        df["EMA_21"]  = close.ewm(span=21).mean()

        # ATR
        hl  = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low  - close.shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.where(loss != 0, np.nan)
        df["RSI"] = (100 - 100 / (1 + rs)).where(loss != 0, 100.0)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["BB_Upper"] = sma20 + 2 * std20
        df["BB_Lower"] = sma20 - 2 * std20
        df["BB_PctB"]  = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

        # ADX
        plus_dm  = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        atr14    = tr.rolling(14).mean()
        pdi      = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
        mdi      = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
        dx       = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
        df["ADX"] = dx.rolling(14).mean()

        # Volume trend
        df["Vol_SMA20"] = vol.rolling(20).mean()
        df["Vol_Ratio"] = vol / (df["Vol_SMA20"] + 1e-9)

        # Seasonality (month encoding)
        df["Month"] = df.index.month
        df["DayOfWeek"] = df.index.dayofweek

        return df.dropna(subset=["SMA_20", "ATR", "RSI", "MACD"])

    # ── Macro regime filters ───────────────────────────────────────────────────

    def _macro_filter(self, asset_class: str) -> dict:
        """
        Returns macro context for signal filtering.
        Gold/Silver: bullish when VIX high + DXY weak
        Energy: bullish when global demand proxy positive
        Base metals: bullish when NIFTY in uptrend
        """
        vix = self._fetch_vix()
        dxy_trend = self._fetch_dxy()

        context = {
            "vix": round(vix, 1),
            "dxy_trend_pct": round(dxy_trend, 2),
            "fear_regime": vix > 25,
            "dollar_strengthening": dxy_trend > 0.5,
        }

        if asset_class == "precious":
            # Gold/Silver: BUY bias when fear high OR dollar weak
            context["macro_bias"] = "BULL" if (vix > 22 or dxy_trend < -0.3) else \
                                    "BEAR" if (vix < 15 and dxy_trend > 0.5) else "NEUTRAL"
        elif asset_class == "energy":
            # Crude/NatGas: neutral macro (supply/demand driven, not fear)
            context["macro_bias"] = "NEUTRAL"
        elif asset_class == "base":
            # Base metals: follow global growth (approximate with NIFTY trend)
            try:
                nifty = yf.Ticker("^NSEI").history(period="6mo")
                if not nifty.empty:
                    nifty_sma50 = nifty["Close"].rolling(50).mean().iloc[-1]
                    nifty_now   = nifty["Close"].iloc[-1]
                    context["macro_bias"] = "BULL" if nifty_now > nifty_sma50 else "BEAR"
                else:
                    context["macro_bias"] = "NEUTRAL"
            except Exception:
                context["macro_bias"] = "NEUTRAL"
        else:
            context["macro_bias"] = "NEUTRAL"

        return context

    # ── Seasonality bias ───────────────────────────────────────────────────────

    def _seasonality_bias(self, symbol: str, month: int) -> str:
        """
        Known seasonal patterns for commodity futures.
        Returns 'BULL', 'BEAR', or 'NEUTRAL'.
        """
        seasonal_map = {
            "GC=F": {  # Gold: strong Jan–Feb (safe haven), Aug–Sep (festival/jewellery India)
                "BULL": [1, 2, 8, 9, 10],
                "BEAR": [3, 4, 5],
            },
            "CL=F": {  # Crude: summer driving (Apr–Jun) bullish, Jan–Feb bearish
                "BULL": [4, 5, 6, 7],
                "BEAR": [1, 2, 11, 12],
            },
            "NG=F": {  # Natural Gas: winter (Nov–Feb) demand bullish
                "BULL": [11, 12, 1, 2],
                "BEAR": [5, 6, 7, 8],
            },
            "ZW=F": {  # Wheat: harvest pressure Jun–Jul bearish
                "BULL": [2, 3, 4],
                "BEAR": [6, 7, 8],
            },
        }
        mapping = seasonal_map.get(symbol, {})
        if month in mapping.get("BULL", []):
            return "BULL"
        elif month in mapping.get("BEAR", []):
            return "BEAR"
        return "NEUTRAL"

    # ── Core signal generation ─────────────────────────────────────────────────

    def _generate_signal(self, df: pd.DataFrame, symbol: str,
                         asset_class: str, macro: dict) -> dict:
        """
        3-layer confirmation:
          Layer 1: Trend (price vs SMA50 and SMA200)
          Layer 2: Momentum (MACD crossover + RSI)
          Layer 3: Volatility (ADX trend strength)
        """
        last = df.iloc[-1]
        close = float(last["Close"])
        atr   = float(last["ATR"])

        buy_score  = 0
        sell_score = 0
        reasons    = []

        # Layer 1: Trend
        if close > float(last["SMA_50"]) > float(last["SMA_200"]):
            buy_score += 2
            reasons.append("price_above_SMA50_200")
        elif close < float(last["SMA_50"]) < float(last["SMA_200"]):
            sell_score += 2
            reasons.append("price_below_SMA50_200")

        # EMA 9/21 cross (recent trend flip)
        if float(last["EMA_9"]) > float(last["EMA_21"]):
            buy_score += 1
            reasons.append("EMA9_above_EMA21")
        else:
            sell_score += 1
            reasons.append("EMA9_below_EMA21")

        # Layer 2: Momentum
        rsi  = float(last["RSI"])
        macd = float(last["MACD"])
        macd_sig = float(last["MACD_Signal"])

        if macd > macd_sig and macd > 0:
            buy_score += 2
        elif macd < macd_sig and macd < 0:
            sell_score += 2

        if 40 < rsi < 65:
            buy_score += 1 if rsi > 50 else 0
        elif rsi > 70:
            sell_score += 1
        elif rsi < 30:
            buy_score += 2   # oversold — stronger buy signal

        # BB squeeze
        bb_pctb = float(last["BB_PctB"])
        if bb_pctb < 0.15:
            buy_score += 1    # near lower band
        elif bb_pctb > 0.85:
            sell_score += 1   # near upper band

        # Layer 3: ADX trend strength
        adx = float(last["ADX"])
        if adx < 20:
            # No strong trend — halve the signal strength
            buy_score  = buy_score  // 2
            sell_score = sell_score // 2

        # Macro filter
        macro_bias = macro.get("macro_bias", "NEUTRAL")
        if macro_bias == "BULL":
            buy_score += 1
        elif macro_bias == "BEAR":
            sell_score += 1

        # Seasonality
        month = int(last["Month"])
        seasonal = self._seasonality_bias(symbol, month)
        if seasonal == "BULL":
            buy_score += 1
        elif seasonal == "BEAR":
            sell_score += 1

        # Determine signal (need 4+ score out of ~8)
        threshold = 4
        risk = RISK_CONFIG[asset_class]

        if buy_score >= threshold and buy_score > sell_score:
            signal     = "BUY"
            raw_conf   = min(0.95, 0.50 + buy_score * 0.06)
            sl         = close - atr * risk["sl_atr"]
            tp         = close + atr * risk["tp_atr"]
        elif sell_score >= threshold and sell_score > buy_score:
            signal     = "SELL"
            raw_conf   = min(0.95, 0.50 + sell_score * 0.06)
            sl         = close + atr * risk["sl_atr"]
            tp         = close - atr * risk["tp_atr"]
        else:
            signal   = "HOLD"
            raw_conf = 0.50
            sl = tp  = close

        confidence = round(raw_conf * 100, 1)
        if signal != "HOLD" and raw_conf < risk["conf_threshold"]:
            signal = "HOLD"

        return {
            "signal":      signal,
            "confidence":  confidence,
            "entry_price": round(close, 4),
            "stop_loss":   round(sl, 4),
            "target":      round(tp, 4),
            "buy_score":   buy_score,
            "sell_score":  sell_score,
            "adx":         round(adx, 1),
            "rsi":         round(rsi, 1),
            "atr":         round(atr, 4),
            "seasonal":    seasonal,
            "macro_bias":  macro_bias,
            "reasons":     reasons,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, symbol: str) -> dict:
        """
        Full analysis pipeline for a single commodity symbol.
        Returns signal dict with entry, SL, TP, confidence, macro context.
        """
        if symbol not in COMMODITIES:
            raise ValueError(f"Unknown commodity: {symbol}. Choose from: {list(COMMODITIES)}")

        meta       = COMMODITIES[symbol]
        asset_class = meta["class"]

        try:
            df_raw = self._fetch(symbol)
            df     = self._compute_features(df_raw)

            if df.empty:
                raise ValueError("Feature computation returned empty DataFrame")

            macro  = self._macro_filter(asset_class)
            sig    = self._generate_signal(df, symbol, asset_class, macro)

            # Build chart data (last 90 bars)
            chart_df = df.tail(90)[["Close", "SMA_20", "SMA_50", "ATR", "RSI",
                                     "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
                                     "ADX", "Volume"]].copy()
            chart_df.index = chart_df.index.strftime("%Y-%m-%d")
            chart_data = chart_df.reset_index().rename(columns={"index": "date"}).to_dict("records")

            return {
                "symbol":       symbol,
                "name":         meta["name"],
                "asset_class":  asset_class,
                "mcx_lot":      meta["mcx_lot"],
                "unit":         meta["unit"],
                "signal":       sig["signal"],
                "confidence":   sig["confidence"],
                "entry_price":  sig["entry_price"],
                "stop_loss":    sig["stop_loss"],
                "target":       sig["target"],
                "atr":          sig["atr"],
                "rsi":          sig["rsi"],
                "adx":          sig["adx"],
                "seasonal_bias": sig["seasonal"],
                "macro":        macro,
                "chart_data":   chart_data,
                "timestamp":    datetime.now(IST).isoformat(),
            }

        except Exception as e:
            logger.error(f"Commodity analysis failed for {symbol}: {e}")
            raise

    def scan_all(self, signal_filter: str | None = None) -> list[dict]:
        """
        Scan all commodity symbols and return signals.
        signal_filter: 'BUY' | 'SELL' | None (return all)
        """
        results = []
        for symbol in COMMODITIES:
            try:
                r = self.analyze(symbol)
                if signal_filter is None or r["signal"] == signal_filter:
                    results.append(r)
            except Exception as e:
                logger.warning(f"Commodity scan skipped {symbol}: {e}")
                results.append({"symbol": symbol, "signal": "ERROR", "error": str(e)})

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results


# Module-level singleton
commodities_engine = CommoditiesEngine()
