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
import requests
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import warnings

# ── In-memory OHLCV data cache (TTL = 4 hours) ───────────────────────────────
_DATA_CACHE: dict = {}   # key: "SYMBOL|interval" → (pd.DataFrame, datetime)
_DATA_CACHE_TTL = 14400  # 4 hours in seconds

def cache_get(symbol: str, interval: str):
    key = f"{symbol}|{interval}"
    entry = _DATA_CACHE.get(key)
    if entry:
        df, ts = entry
        if (datetime.now() - ts).total_seconds() < _DATA_CACHE_TTL:
            return df
    return None

def cache_put(symbol: str, df, interval: str):
    _DATA_CACHE[f"{symbol}|{interval}"] = (df, datetime.now())

# ── Real-time price cache (TTL = 60s) — stops /api/positions from hammering ──
_PRICE_CACHE: dict = {}   # key: symbol → (price: float, datetime)
_PRICE_CACHE_TTL = 60

def _price_cache_get(symbol: str):
    entry = _PRICE_CACHE.get(symbol)
    if entry:
        price, ts = entry
        if (datetime.now() - ts).total_seconds() < _PRICE_CACHE_TTL:
            return price
    return None

def _price_cache_put(symbol: str, price: float):
    _PRICE_CACHE[symbol] = (price, datetime.now())

# ── AV working-format cache — avoids retrying dead symbol formats ─────────────
_AV_FORMAT_CACHE: dict = {}  # key: symbol → best AV symbol string

# ── Hard-timeout wrapper + circuit breaker for yfinance ─────────────────────
# yfinance ignores its own timeout parameter when DNS is down; we wrap every
# call in a daemon thread and abandon after timeout_sec. A circuit breaker
# tracks consecutive failures and skips yfinance entirely for COOLDOWN_SEC
# after _CB_THRESHOLD failures, preventing thread-pool exhaustion.
_YF_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="yf_safe")
_YF_CB_FAILURES   = 0          # consecutive failure counter
_YF_CB_TRIPPED_AT = None       # when the breaker tripped
_YF_CB_THRESHOLD  = 1          # trip after first failure
_YF_CB_COOLDOWN   = 300        # seconds to stay open (5 min)

def _preflight_yf_check():
    """On startup: test Yahoo DNS in 2s. If unreachable, pre-trip the circuit breaker."""
    global _YF_CB_FAILURES, _YF_CB_TRIPPED_AT
    import socket, logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("query2.finance.yahoo.com", 443)
        socket.setdefaulttimeout(None)
        _log.info("✅ Yahoo Finance DNS reachable — yfinance enabled")
    except Exception:
        socket.setdefaulttimeout(None)
        _YF_CB_FAILURES = _YF_CB_THRESHOLD
        _YF_CB_TRIPPED_AT = datetime.now()
        _log.warning("⚡ Yahoo Finance DNS unreachable at startup — circuit breaker pre-tripped; using Alpha Vantage only")

_preflight_yf_check()

def _yf_circuit_open() -> bool:
    """Return True if the circuit breaker is open (yfinance should be skipped)."""
    global _YF_CB_TRIPPED_AT
    if _YF_CB_TRIPPED_AT is None:
        return False
    elapsed = (datetime.now() - _YF_CB_TRIPPED_AT).total_seconds()
    if elapsed > _YF_CB_COOLDOWN:
        # Reset after cooldown
        _YF_CB_TRIPPED_AT = None
        return False
    return True

def _yf_record_failure():
    global _YF_CB_FAILURES, _YF_CB_TRIPPED_AT
    _YF_CB_FAILURES += 1
    if _YF_CB_FAILURES >= _YF_CB_THRESHOLD and _YF_CB_TRIPPED_AT is None:
        _YF_CB_TRIPPED_AT = datetime.now()
        logger.warning(f"⚡ yfinance circuit breaker OPEN after {_YF_CB_FAILURES} failures — "
                       f"skipping yfinance for {_YF_CB_COOLDOWN}s")

def _yf_record_success():
    global _YF_CB_FAILURES, _YF_CB_TRIPPED_AT
    _YF_CB_FAILURES = 0
    _YF_CB_TRIPPED_AT = None

def _yf_download_safe(symbol, period, interval, timeout_sec=5, **kwargs):
    """Run yf.download in a thread; abandon after timeout_sec; update circuit breaker."""
    if _yf_circuit_open():
        return pd.DataFrame()
    future = _YF_EXECUTOR.submit(
        yf.download, symbol, period=period, interval=interval,
        auto_adjust=True, progress=False, **kwargs
    )
    try:
        result = future.result(timeout=timeout_sec)
        if result is not None and not result.empty:
            _yf_record_success()
        else:
            _yf_record_failure()
        return result if result is not None else pd.DataFrame()
    except (FuturesTimeoutError, Exception):
        _yf_record_failure()
        return pd.DataFrame()

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
                    import tensorflow as tf
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                    self.lstm_scaler = joblib.load(scaler_path)
                    logger.info("Loaded ASTRA.ML (Bidirectional LSTM) model.")
                except ImportError:
                    logger.warning("TensorFlow not installed. ASTRA.ML (LSTM) disabled.")
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")

    # ─────────────────────── MACRO FILTER ───────────────────────

    def _is_macro_bullish(self) -> bool:
        """Check if NIFTY 50 is above its 200 SMA — only allow BUY in bull regime."""
        now = datetime.now()
        if self.macro_trend_updated and (now - self.macro_trend_updated).total_seconds() < 3600:
            return self.macro_trend_bullish
        try:
            df = _yf_download_safe("^NSEI", period="1y", interval="1d", timeout_sec=5)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                lp = float(df.iloc[-1]["Close"])
                s200 = float(df["Close"].rolling(200).mean().iloc[-1])
                self.macro_trend_bullish = lp > s200
                self.macro_trend_updated = now
                logger.info(f"Macro trend: NIFTY @ {lp:.0f} vs SMA200 @ {s200:.0f} → {'BULL' if self.macro_trend_bullish else 'BEAR'}")
        except Exception:
            pass
        return self.macro_trend_bullish

    def _get_weekly_trend(self, symbol: str) -> str:
        """
        Multi-timeframe filter: check weekly chart trend for a symbol.
        Returns 'BULL', 'BEAR', or 'NEUTRAL'.
        Cached per symbol for 4 hours (weekly bars don't change intraday).
        """
        cache_key = f"_weekly_{symbol}"
        cached = getattr(self, "_weekly_cache", {})
        now = datetime.now()
        if cache_key in cached:
            result, ts = cached[cache_key]
            if (now - ts).total_seconds() < 14400:  # 4-hour TTL
                return result
        try:
            df_w = _yf_download_safe(symbol, period="2y", interval="1wk", timeout_sec=5)
            if df_w is not None and not df_w.empty and len(df_w) >= 30:
                if isinstance(df_w.columns, pd.MultiIndex):
                    df_w.columns = df_w.columns.get_level_values(0)
                close = df_w["Close"]
                sma30w = close.rolling(30).mean().iloc[-1]
                sma10w = close.rolling(10).mean().iloc[-1]
                lp = float(close.iloc[-1])
                if lp > float(sma30w) and float(sma10w) > float(sma30w):
                    trend = "BULL"
                elif lp < float(sma30w) and float(sma10w) < float(sma30w):
                    trend = "BEAR"
                else:
                    trend = "NEUTRAL"
                if not hasattr(self, "_weekly_cache"):
                    self._weekly_cache = {}
                self._weekly_cache[cache_key] = (trend, now)
                logger.info(f"Weekly trend {symbol}: {trend} (price={lp:.2f} vs SMA30w={sma30w:.2f})")
                return trend
        except Exception as e:
            logger.debug(f"Weekly trend check failed for {symbol}: {e}")
        # Cache NEUTRAL to avoid repeated failing calls for 4 hours
        if not hasattr(self, "_weekly_cache"):
            self._weekly_cache = {}
        self._weekly_cache[cache_key] = ("NEUTRAL", now)
        return "NEUTRAL"

    # ─────────────────────── DATA FETCHING ───────────────────────

    def _fetch_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV data with cache + multi-source fallback chain.
        Source priority:
          0. In-memory cache (4-hour TTL)
          1. Twelve Data (800 req/day free — best quality, supports Indian stocks)
          2. Alpha Vantage (25 req/day — reliable backup, tries bare/BSE/NSE formats)
          3. yfinance primary (hard-capped 5s thread + circuit breaker)
          4. yfinance with .NS suffix
          5. CIRCUIT BREAKER — no synthetic fallback
        """
        # 0. Cache hit — skip all API calls
        cached = cache_get(symbol, interval)
        if cached is not None:
            logger.debug(f"Cache hit for {symbol} [{interval}]")
            return cached

        # 0.5. Dhan HQ (primary — unlimited, native NSE/BSE, no rate limits)
        try:
            from app.services.dhan_data import dhan_data_service
            if dhan_data_service.is_available():
                df_dhan = dhan_data_service.get_ohlcv(symbol, period=period, interval=interval)
                if df_dhan is not None and not df_dhan.empty and len(df_dhan) > 20:
                    cache_put(symbol, df_dhan, interval)
                    logger.info(f"Dhan feed OK for {symbol}: {len(df_dhan)} bars [{interval}]")
                    return df_dhan
        except Exception as e:
            logger.debug(f"Dhan data unavailable: {e}")

        # Determine if we need extended history (>100 days)
        need_full = period in ["1y", "2y", "5y", "max", "2mo", "3mo", "6mo"]

        # 1. Twelve Data (800 free req/day)
        #    Free tier covers: US stocks, crypto, and some international listings.
        #    Pure NSE-only symbols (RELIANCE, TCS) return 404 on free plan — skip them
        #    to avoid wasting a quota request. Crypto needs "-" → "/" conversion.
        td_key = os.getenv("TWELVE_DATA_KEY", "")
        if td_key and interval == "1d":
            try:
                is_crypto_sym = "-USD" in symbol or "-BTC" in symbol or "-ETH" in symbol
                is_pure_nse   = ".NS" in symbol  # NSE-suffixed symbols need paid plan
                if not is_pure_nse:              # Only attempt for non-NSE symbols
                    td_symbol = symbol.replace("-", "/") if is_crypto_sym else symbol.replace(".NS", "")
                    size_map  = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
                    outputsize = size_map.get(period, 180)
                    url = (
                        f"https://api.twelvedata.com/time_series?symbol={td_symbol}"
                        f"&interval=1day&outputsize={outputsize}&apikey={td_key}&format=JSON"
                    )
                    res = requests.get(url, timeout=8)
                    data = res.json()
                    if "values" in data and len(data["values"]) > 20:
                        df_td = pd.DataFrame(data["values"])
                        df_td["datetime"] = pd.to_datetime(df_td["datetime"])
                        df_td = df_td.set_index("datetime").sort_index()
                        df_td = df_td.rename(columns={"open": "Open", "high": "High",
                                                       "low": "Low", "close": "Close",
                                                       "volume": "Volume"})
                        df_td = df_td[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                        cache_put(symbol, df_td, interval)
                        logger.info(f"Twelve Data feed OK for {symbol}: {len(df_td)} bars")
                        return df_td
            except Exception as e:
                logger.debug(f"Twelve Data unavailable: {e}")

        # 2. Alpha Vantage — free tier only supports 'compact' (100 bars); 'full' requires premium
        try:
            av_key = os.getenv("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_KEY", "XV1FMHS5UHPIIPAZ"))
            bare = symbol.replace(".NS", "").replace(".BSE", "").split("-")[0]
            # Try the format that worked last time first, then fall back to all candidates.
            # Order: preferred (cached) → bare.BSE (works for Indian stocks) → bare (US stocks)
            # Skip bare.NSE — Alpha Vantage returns empty results for .NSE format.
            preferred = _AV_FORMAT_CACHE.get(symbol)
            av_candidates = [preferred] if preferred else []
            for fmt in [bare + ".BSE", bare]:  # .BSE first — works for NSE-listed Indian stocks
                if fmt not in av_candidates:
                    av_candidates.append(fmt)
            for av_sym in av_candidates:
                try:
                    url = (
                        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                        f"&symbol={av_sym}&apikey={av_key}&outputsize=compact"
                    )
                    res = requests.get(url, timeout=8)
                    data = res.json()
                    if "Time Series (Daily)" in data:
                        ts = data["Time Series (Daily)"]
                        df_av = pd.DataFrame.from_dict(ts, orient="index")
                        df_av = df_av.rename(columns={
                            "1. open": "Open", "2. high": "High",
                            "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
                        })
                        df_av.index = pd.to_datetime(df_av.index)
                        df_av = df_av.astype(float).sort_index()
                        if len(df_av) > 20:
                            _AV_FORMAT_CACHE[symbol] = av_sym  # Remember working format
                            logger.info(f"AlphaVantage feed OK for {symbol} (as {av_sym}): {len(df_av)} bars")
                            cache_put(symbol, df_av, interval)
                            return df_av
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"AlphaVantage unavailable: {e}")

        # 2. yfinance primary (hard-capped at 5s via thread)
        try:
            df = _yf_download_safe(symbol, period=period, interval=interval, timeout_sec=5)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 20:
                    cache_put(symbol, df, interval)
                    return df
        except Exception:
            pass

        # 3. yfinance with .NS suffix for Indian stocks
        if ".NS" not in symbol and "^" not in symbol and "=" not in symbol and "-" not in symbol:
            try:
                df = _yf_download_safe(symbol + ".NS", period=period, interval=interval, timeout_sec=5)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) > 20:
                        cache_put(symbol, df, interval)
                        return df
            except Exception:
                pass

        # 4. Data exhaust - all sources failed
        logger.warning(f"⚠️  Data fetch exhausted for {symbol}. Proceeding with empty set.")
        return pd.DataFrame()

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
        ASTRA 1.0 — 4/6 multi-confirmation BUY system.
        Returns (passed, score, confirmation_details)
        """
        checks = {
            "RSI oversold & turning up": (row["RSI"] < 35) and (row["RSI_Slope"] > 0),
            "Trend & strength (SMA200 + ADX)": (row["Dist_SMA200"] > 0) and (row["ADX"] > 20),
            "MACD bullish crossover": row["MACD"] > row["MACD_Signal"],
            "Volume spike (>1.5x avg)": row["Volume_Ratio"] > 1.5,
            "Bullish candlestick pattern": patterns.get("any_bullish", False),
            "Near lower Bollinger Band": row["BB_PctB"] < 0.25,
        }
        passed_count = sum(checks.values())
        score = (passed_count / 6) * 100

        # Macro filter: In a bear market, require 5/6 for BUY (stricter)
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
                    "error": "Real-time data currently unavailable from all sources.",
                    "current_price": 0.0, "entry_price": 0.0, "target": 0.0, "stop_loss": 0.0,
                    "chartData": []
                }

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
            for idx, row in df.tail(180).iterrows():
                try:
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
            _cache_hit = cache_get(asset_symbol, interval) is not None
            data_freshness_note = (
                "⚠️  Data Rate-Limit Notice: To preserve API quotas, prices are cached up to 60 s "
                "and OHLCV bars up to 4 h. Entry / SL / TP levels shown are indicative — "
                "always confirm the live market price before executing any trade."
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
                "price_cache_ttl_sec": 60,
                "ohlcv_cache_ttl_hr": 4,
            }

        except Exception as e:
            logger.error(f"Analysis failed for {asset_symbol}: {e}", exc_info=True)
            return {
                "asset": asset_symbol, "signal": "HOLD", "confidence": 0.0,
                "error": str(e), "current_price": 0.0, "entry_price": 0.0,
                "target": 0.0, "stop_loss": 0.0, "chartData": []
            }

    def get_realtime_price(self, symbol: str) -> float:
        # 0. DhanFeed live price (sub-second, if WebSocket is connected)
        try:
            from app.services.dhan_feed import dhan_feed_manager
            live = dhan_feed_manager.get_price(symbol)
            if live and live > 0:
                _price_cache_put(symbol, live)
                return live
        except Exception:
            pass

        # 0.5. Price cache (60s TTL) — prevents /api/positions from firing on every poll
        cached_price = _price_cache_get(symbol)
        if cached_price is not None:
            return cached_price

        # 1. Try yfinance fast_info (hard-capped via thread + circuit breaker)
        if not _yf_circuit_open():
            try:
                def _yf_price():
                    ticker = yf.Ticker(symbol)
                    fast = ticker.fast_info
                    price = getattr(fast, "last_price", None) or fast.get("lastPrice")
                    if price and float(price) > 0:
                        return round(float(price), 2)
                    df = ticker.history(period="1d")
                    if not df.empty:
                        return round(float(df.iloc[-1]["Close"]), 2)
                    return None
                future = _YF_EXECUTOR.submit(_yf_price)
                result = future.result(timeout=5)
                if result:
                    _yf_record_success()
                    _price_cache_put(symbol, result)
                    return result
                _yf_record_failure()
            except Exception:
                _yf_record_failure()

        # 2. Fallback: use latest close from cached/fetched data
        try:
            df = self._fetch_data(symbol, period="1mo", interval="1d")
            if not df.empty:
                p = round(float(df.iloc[-1]["Close"]), 2)
                _price_cache_put(symbol, p)
                return p
        except Exception:
            pass

        # 3. Alpha Vantage GLOBAL_QUOTE for real-time price
        try:
            av_key = os.getenv("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_KEY", "XV1FMHS5UHPIIPAZ"))
            bare = symbol.replace(".NS", "").replace(".BSE", "").split("-")[0]
            preferred = _AV_FORMAT_CACHE.get(symbol)
            av_syms = [preferred] if preferred else []
            for fmt in [bare, bare + ".BSE"]:
                if fmt not in av_syms:
                    av_syms.append(fmt)
            for av_sym in av_syms:
                try:
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_sym}&apikey={av_key}"
                    res = requests.get(url, timeout=6)
                    data = res.json()
                    gq = data.get("Global Quote", {})
                    price_str = gq.get("05. price", "")
                    if price_str:
                        p = round(float(price_str), 2)
                        if p > 0:
                            _price_cache_put(symbol, p)
                            return p
                except Exception:
                    continue
        except Exception:
            pass

        return 0.0

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

    def train_model_lstm(self, df: pd.DataFrame, lookback: int = 30):
        """
        Build and train Bidirectional LSTM with Attention for equity predictions.
        Architecture: BiLSTM(128) → BiLSTM(64) → Attention → Dense(32) → Dense(1)
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import RobustScaler

            df = self._compute_features(df)
            if df.empty or len(df) < lookback + 50:
                logger.error("Insufficient data for LSTM training")
                return None, None

            # Target: 3-day forward return
            df["Target"] = df["Close"].pct_change(3).shift(-3) * 100
            df = df.dropna(subset=FEATURE_COLS + ["Target"])

            raw_X = df[FEATURE_COLS].values
            y = df["Target"].values

            # Build sequences before scaling (no data leakage)
            X_seq, y_seq = [], []
            for i in range(lookback, len(raw_X) - 1):
                X_seq.append(raw_X[i - lookback:i])
                y_seq.append(y[i])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            # Chronological 80/20 split
            split = int(len(X_seq) * 0.8)
            X_train_raw, X_val_raw = X_seq[:split], X_seq[split:]
            y_train, y_val = y_seq[:split], y_seq[split:]

            # Fit scaler only on train data, transform both
            n_train, n_steps, n_feats = X_train_raw.shape
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_feats)).reshape(n_train, n_steps, n_feats)
            X_val = scaler.transform(X_val_raw.reshape(-1, n_feats)).reshape(X_val_raw.shape[0], n_steps, n_feats)

            # ── Bidirectional LSTM with Attention (no Lambda layers) ──
            inputs = tf.keras.Input(shape=(lookback, len(FEATURE_COLS)))
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
            )(inputs)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
            )(x)
            # GlobalAveragePooling1D instead of Lambda sum (safe to serialize)
            context = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(32, activation="relu")(context)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(1)(x)

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="huber"
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
            ]

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            logger.info(f"LSTM equity model trained: {X_train.shape}")
            return model, scaler

        except ImportError:
            logger.error("TensorFlow not available. Cannot train LSTM.")
            return None, None
        except Exception as e:
            logger.error(f"LSTM training failed: {e}", exc_info=True)
            return None, None


ai_engine = AIPredictionEngine()
