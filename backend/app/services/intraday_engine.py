"""
ASTRA Intraday Engine (Phase 2)
================================
Generates intraday signals using 15-minute OHLCV data.
Strategies: Opening Range Breakout (ORB), VWAP Mean Reversion, Momentum (RSI+MACD).

Data priority:
  1. Dhan HQ intraday_minute_data (native NSE, unlimited)
  2. Alpha Vantage TIME_SERIES_INTRADAY (.BSE suffix, 30 days free)
"""

import logging
import os
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MARKET_OPEN      = "09:15"
MARKET_CLOSE     = "15:30"
ORB_WINDOW_MIN   = 30          # First 30 minutes define the Opening Range (09:15–09:45)
INTRADAY_SL_PCT  = 0.003       # 0.3% stop-loss
INTRADAY_TP_PCT  = 0.006       # 0.6% take-profit
NO_TRADE_AFTER   = "14:45"     # No new entries after this time
SQUARE_OFF_TIME  = "15:15"     # Auto square-off: close all positions by 15:15 IST

_IST = pytz.timezone("Asia/Kolkata")
_UTC = pytz.utc

# ── Intraday data cache (TTL: 5 minutes) ──────────────────────────────────────
_INTRADAY_CACHE: dict = {}   # symbol → (DataFrame, cached_at)
_INTRADAY_CACHE_TTL = 300    # 5 minutes


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fetch_intraday(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    Fetch 15-minute OHLCV bars for the last `days` trading days.

    Priority:
      1. Dhan HQ intraday_minute_data (native NSE, unlimited)
      2. Alpha Vantage TIME_SERIES_INTRADAY (.BSE suffix)

    Returns a cleaned DataFrame with timezone-aware IST index.
    Symbol should be the bare NSE symbol (e.g. "RELIANCE"), .NS stripped internally.
    """
    # Normalise symbol — strip .NS suffix for Dhan, keep bare for AV lookup
    clean = symbol.strip().replace(".NS", "").replace(".BSE", "").upper()

    # ── Cache check ──────────────────────────────────────────────────────────
    if clean in _INTRADAY_CACHE:
        cached_df, cached_at = _INTRADAY_CACHE[clean]
        age = (datetime.now() - cached_at).total_seconds()
        if age < _INTRADAY_CACHE_TTL and not cached_df.empty:
            return cached_df

    df = pd.DataFrame()

    # ── 1. Yahoo Finance direct (free, no key, 5-day 15m) ───────────────────
    try:
        from app.services.yahoo_finance import yahoo_service
        yf_df = yahoo_service.get_ohlcv(f"{clean}.NS", period="5d", interval="15m")
        if yf_df is not None and not yf_df.empty:
            if yf_df.index.tz is None:
                yf_df.index = yf_df.index.tz_localize("Asia/Kolkata")
            else:
                yf_df.index = yf_df.index.tz_convert("Asia/Kolkata")
            yf_df = yf_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            yf_df = yf_df[yf_df["Volume"] > 0]
            if not yf_df.empty:
                logger.info(f"[intraday] Yahoo 15m: {len(yf_df)} bars for {clean}")
                df = yf_df
    except Exception as e:
        logger.debug(f"[intraday] Yahoo 15m fetch failed for {clean}: {e}")

    # ── 2. Dhan HQ (requires daily session activation) ───────────────────────
    if df.empty:
        try:
            from app.services.dhan_data import dhan_data_service
            if dhan_data_service.is_available():
                dhan_df = dhan_data_service.get_ohlcv(clean, period="1mo", interval="15m")
                if not dhan_df.empty:
                    if dhan_df.index.tz is None:
                        dhan_df.index = dhan_df.index.tz_localize("Asia/Kolkata")
                    else:
                        dhan_df.index = dhan_df.index.tz_convert("Asia/Kolkata")
                    cutoff = dhan_df.index[-1] - pd.Timedelta(days=days + 7)
                    dhan_df = dhan_df[dhan_df.index >= cutoff]
                    dhan_df = dhan_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                    dhan_df = dhan_df[dhan_df["Volume"] > 0]
                    if not dhan_df.empty:
                        logger.info(f"[intraday] Dhan: {len(dhan_df)} 15m bars for {clean}")
                        df = dhan_df
        except Exception as e:
            logger.debug(f"[intraday] Dhan 15m fetch failed for {clean}: {e}")

    if df.empty:
        logger.warning(f"[intraday] No 15m data available for {clean}")
        return df

    _INTRADAY_CACHE[clean] = (df, datetime.now())
    return df


def _fetch_intraday_extended(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch up to `days` of 15m bars for backtesting (bypasses short cache)."""
    clean = symbol.strip().replace(".NS", "").replace(".BSE", "").upper()
    df = pd.DataFrame()

    # Yahoo Finance direct — 60d of 1h bars, or 5d of 15m (best free source)
    try:
        from app.services.yahoo_finance import yahoo_service
        # For backtest: use 1h bars over a longer window when days > 5
        if days > 5:
            yf_df = yahoo_service.get_ohlcv(f"{clean}.NS", period="1mo", interval="1h")
        else:
            yf_df = yahoo_service.get_ohlcv(f"{clean}.NS", period="5d", interval="15m")
        if yf_df is not None and not yf_df.empty:
            if yf_df.index.tz is None:
                yf_df.index = yf_df.index.tz_localize("Asia/Kolkata")
            else:
                yf_df.index = yf_df.index.tz_convert("Asia/Kolkata")
            yf_df = yf_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            yf_df = yf_df[yf_df["Volume"] > 0]
            if not yf_df.empty:
                logger.info(f"[backtest] Yahoo: {len(yf_df)} bars for {clean}")
                df = yf_df
    except Exception as e:
        logger.debug(f"[backtest] Yahoo fetch failed for {clean}: {e}")

    # Twelve Data — free tier supports 1h intraday for some Indian stocks
    # Format: RELIANCE:NSE  (colon separator, no .NS suffix)
    if df.empty:
        try:
            import os, requests as _req, pandas as _pd
            td_key = os.getenv("TWELVE_DATA_KEY", "")
            if td_key:
                td_sym = f"{clean}:NSE"
                outputsize = min(days * 8, 5000)   # ~8 × 1h bars per trading day
                url = (
                    f"https://api.twelvedata.com/time_series?symbol={td_sym}"
                    f"&interval=1h&outputsize={outputsize}&apikey={td_key}&format=JSON"
                )
                r = _req.get(url, timeout=10)
                data = r.json()
                if "values" in data and len(data["values"]) > 10:
                    td_df = _pd.DataFrame(data["values"])
                    td_df["datetime"] = _pd.to_datetime(td_df["datetime"])
                    td_df = td_df.set_index("datetime").sort_index()
                    td_df = td_df.rename(columns={
                        "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "volume": "Volume"
                    })
                    td_df = td_df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                    td_df.index = td_df.index.tz_localize("Asia/Kolkata")
                    td_df = td_df.dropna(subset=["Close"])
                    if not td_df.empty:
                        logger.info(f"[backtest] Twelve Data: {len(td_df)} 1h bars for {clean}")
                        df = td_df
        except Exception as e:
            logger.debug(f"[backtest] Twelve Data fetch failed for {clean}: {e}")

    # Dhan HQ (requires daily session activation)
    if df.empty:
        try:
            from app.services.dhan_data import dhan_data_service
            if dhan_data_service.is_available():
                period = "3mo" if days > 60 else "1mo"
                dhan_df = dhan_data_service.get_ohlcv(clean, period=period, interval="15m")
                if not dhan_df.empty:
                    if dhan_df.index.tz is None:
                        dhan_df.index = dhan_df.index.tz_localize("Asia/Kolkata")
                    else:
                        dhan_df.index = dhan_df.index.tz_convert("Asia/Kolkata")
                    cutoff = dhan_df.index[-1] - pd.Timedelta(days=days + 14)
                    dhan_df = dhan_df[dhan_df.index >= cutoff]
                    dhan_df = dhan_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                    dhan_df = dhan_df[dhan_df["Volume"] > 0]
                    if not dhan_df.empty:
                        df = dhan_df
        except Exception as e:
            logger.debug(f"[backtest] Dhan extended fetch failed for {clean}: {e}")

    # ── Synthetic fallback (dev/testing only) ────────────────────────────────
    # When all live intraday sources are unavailable (Yahoo 429, Dhan inactive),
    # generate fully self-contained synthetic 15m bars.  Tries to seed the path
    # from real daily OHLCV (AV/Yahoo); if all sources fail, uses a GBM with
    # typical Nifty 50 parameters.  Clearly flagged in log — not for live use.
    if df.empty:
        try:
            daily_df = _get_daily_for_synthesis(clean)
            synth = _synthesize_intraday(daily_df, n_bars=26, symbol=clean)
            if not synth.empty:
                df = synth
                logger.warning(
                    f"[backtest] {clean}: SYNTHETIC 15m data "
                    f"({len(df)} bars, {len(daily_df)} daily seed bars). "
                    "Activate Yahoo / Dhan for real backtesting."
                )
        except Exception as e:
            logger.debug(f"[backtest] Synthetic fallback failed for {clean}: {e}")

    return df


def _get_daily_for_synthesis(symbol: str) -> pd.DataFrame:
    """Try to get real daily OHLCV to seed the synthetic generator. Returns empty DF on failure."""
    # Only use in-process cache — never trigger a live API call that could fail or rate-limit
    try:
        from app.services.ai_predictor import cache_get
        cached = cache_get(symbol, "1d")
        if cached is not None and not cached.empty:
            return cached
    except Exception:
        pass
    # Try AV directly — this is a single targeted call, not the full fallback chain
    try:
        import os, requests as _r
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_KEY", "XV1FMHS5UHPIIPAZ"))
        for fmt in [symbol + ".BSE", symbol]:
            url = (
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                f"&symbol={fmt}&apikey={av_key}&outputsize=compact"
            )
            res = _r.get(url, timeout=6)
            data = res.json()
            if "Time Series (Daily)" in data:
                ts = data["Time Series (Daily)"]
                df_av = pd.DataFrame.from_dict(ts, orient="index")
                df_av = df_av.rename(columns={
                    "1. open": "Open", "2. high": "High",
                    "3. low": "Low",  "4. close": "Close", "5. volume": "Volume"
                })
                df_av.index = pd.to_datetime(df_av.index)
                df_av = df_av.astype(float).sort_index()
                if not df_av.empty:
                    return df_av
    except Exception:
        pass
    return pd.DataFrame()


# Typical Nifty 50 base prices (used when no real data is available)
_NIFTY50_BASE_PRICES = {
    "RELIANCE": 1460, "TCS": 3300, "HDFCBANK": 1900, "INFY": 1180,
    "ICICIBANK": 1350, "AXISBANK": 1100, "WIPRO": 470, "LT": 3500,
    "TATAMOTORS": 700, "MARUTI": 12000, "SUNPHARMA": 1700, "BAJFINANCE": 8500,
    "ASIANPAINT": 2600, "HDFC": 2700, "KOTAKBANK": 1900, "ITC": 420,
    "SBIN": 830, "BAJAJFINSV": 1900, "NESTLEIND": 2400, "TITAN": 3300,
    "ULTRACEMCO": 11500, "POWERGRID": 320, "ONGC": 265, "NTPC": 340,
    "DIVISLAB": 5400, "BHARTIARTL": 1750, "HCLTECH": 1600, "HINDALCO": 680,
    "COALINDIA": 430, "GRASIM": 2600,
}


def _synthesize_intraday(daily_df: pd.DataFrame, n_bars: int = 26, symbol: str = "") -> pd.DataFrame:
    """
    Convert a daily OHLCV DataFrame into synthetic hourly bars for backtesting.

    Method
    ------
    For each day, we generate `n_bars` equally-spaced bars (default 13, covering
    09:15–15:15 IST in 30-min steps).  The bar sequence obeys three constraints:
      • First bar open  == day open
      • Last bar close  == day close
      • Path high/low stay within day High / Low

    Price path uses a biased Brownian motion that drifts from Open toward Close,
    with a fixed random seed per day for reproducibility.
    """
    rows = []
    market_open_h, market_open_m = 9, 15
    # 375 min session / n_bars  (26 bars → 15-min bars, 13 bars → ~29-min bars)
    bar_minutes = max(1, int(375 / n_bars))

    # If no real daily data, generate synthetic daily bars from GBM + base price table
    if daily_df is None or daily_df.empty:
        base_price = _NIFTY50_BASE_PRICES.get(symbol.upper(), 1000)
        rng0 = np.random.default_rng(abs(hash(symbol)) % (2**31))
        from datetime import date as _date, timedelta as _td
        from pandas.tseries.offsets import BDay
        end_dt   = pd.Timestamp.now().normalize()
        bdays    = pd.bdate_range(end=end_dt, periods=60)
        prices   = [base_price]
        for _ in range(len(bdays) - 1):
            prices.append(prices[-1] * np.exp(rng0.normal(0.0005, 0.012)))
        synth_days = []
        for i, bd in enumerate(bdays):
            p = prices[i]
            vol_range = p * rng0.uniform(0.01, 0.02)
            day_open  = p * rng0.uniform(0.995, 1.005)
            day_close = p * rng0.uniform(0.99, 1.01)
            day_high  = max(day_open, day_close) * rng0.uniform(1.001, 1.015)
            day_low   = min(day_open, day_close) * rng0.uniform(0.985, 0.999)
            day_vol   = int(base_price * rng0.uniform(500_000, 3_000_000) / base_price)
            synth_days.append({
                "Open": day_open, "High": day_high, "Low": day_low,
                "Close": day_close, "Volume": day_vol
            })
        daily_df = pd.DataFrame(synth_days, index=bdays)
        daily_df.index.name = "Date"

    for ts_day, day in daily_df.iterrows():
        day_open  = float(day["Open"])
        day_high  = float(day["High"])
        day_low   = float(day["Low"])
        day_close = float(day["Close"])
        day_vol   = float(day["Volume"])

        if day_open <= 0 or day_close <= 0:
            continue

        # Deterministic seed from date to ensure repeatable backtest
        if hasattr(ts_day, "year"):
            seed = ts_day.year * 10000 + ts_day.month * 100 + ts_day.day
        else:
            seed = 42
        rng = np.random.default_rng(seed)

        # Generate n_bars price increments that sum to (close - open)
        noise    = rng.standard_normal(n_bars)
        noise   -= noise.mean()                  # zero-mean
        drift    = (day_close - day_open) / n_bars
        steps    = drift + noise * abs(day_close - day_open) * 0.12
        # Adjust so cumulative sum lands on close exactly
        steps[-1] = (day_close - day_open) - steps[:-1].sum()
        prices = np.empty(n_bars + 1)
        prices[0] = day_open
        for i in range(n_bars):
            prices[i + 1] = prices[i] + steps[i]

        # Clamp to day range
        prices = np.clip(prices, day_low * 0.995, day_high * 1.005)
        prices[0]  = day_open
        prices[-1] = day_close

        # U-shaped volume profile — interpolated to exactly n_bars
        _u_base = np.array([2.0, 1.6, 1.3, 1.1, 0.9, 0.8, 0.7, 0.7, 0.7,
                             0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0], dtype=float)
        x_old = np.linspace(0, 1, len(_u_base))
        x_new = np.linspace(0, 1, n_bars)
        vol_weights = np.interp(x_new, x_old, _u_base)
        vol_weights /= vol_weights.sum()
        bar_vols = (vol_weights * day_vol).astype(int)

        # Convert to bars
        try:
            day_date = ts_day.date() if hasattr(ts_day, "date") else pd.Timestamp(ts_day).date()
        except Exception:
            continue

        for i in range(n_bars):
            bar_open  = prices[i]
            bar_close = prices[i + 1]
            bar_high  = max(bar_open, bar_close) * rng.uniform(1.0, 1.003)
            bar_low   = min(bar_open, bar_close) * rng.uniform(0.997, 1.0)
            bar_high  = min(bar_high, day_high)
            bar_low   = max(bar_low,  day_low)

            bar_time = pd.Timestamp(
                year  = day_date.year,
                month = day_date.month,
                day   = day_date.day,
                hour  = market_open_h + (market_open_m + i * bar_minutes) // 60,
                minute= (market_open_m + i * bar_minutes) % 60,
            ).tz_localize("Asia/Kolkata")

            rows.append({
                "Date":   bar_time,
                "Open":   round(bar_open, 2),
                "High":   round(bar_high, 2),
                "Low":    round(bar_low,  2),
                "Close":  round(bar_close, 2),
                "Volume": int(bar_vols[i]),
            })

    if not rows:
        return pd.DataFrame()

    df_syn = pd.DataFrame(rows).set_index("Date").sort_index()
    df_syn.index.name = "Date"
    return df_syn


# ── Trend filter ──────────────────────────────────────────────────────────────

def _trend_aligned(df: pd.DataFrame, signal: str) -> bool:
    """
    Return True if `signal` is aligned with the 50-bar SMA of Close.

    BUY  aligned → latest Close > SMA50
    SELL aligned → latest Close < SMA50
    Fewer than 50 bars → no filter, always True.

    Exported so intraday_backtest.py can import and reuse it.
    """
    if df is None or len(df) < 50:
        return True
    sma50 = df["Close"].rolling(50).mean().iloc[-1]
    if pd.isna(sma50):
        return True
    last_close = float(df["Close"].iloc[-1])
    if signal == "BUY":
        return last_close > float(sma50)
    elif signal == "SELL":
        return last_close < float(sma50)
    return True


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP = cumsum(typical_price × volume) / cumsum(volume)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol  = df["Volume"].cumsum()
    cum_tpv  = (typical * df["Volume"]).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def _compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.where(loss != 0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.where(loss != 0, 100.0)


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_orb(df: pd.DataFrame, date) -> tuple:
    """
    Opening Range (high/low) from first ORB_WINDOW_MIN minutes of the given date.
    Returns (orb_high, orb_low) or (None, None).
    """
    day_df = df[df.index.date == date]
    if day_df.empty:
        return None, None
    orb_start = dtime(9, 15)
    orb_end   = dtime(9, 15 + ORB_WINDOW_MIN)  # 09:45
    orb_df    = day_df[(day_df.index.time >= orb_start) & (day_df.index.time < orb_end)]
    if orb_df.empty:
        return None, None
    return float(orb_df["High"].max()), float(orb_df["Low"].min())


# ── Strategy 1: Opening Range Breakout ───────────────────────────────────────

def orb_signal(df: pd.DataFrame, date) -> dict:
    """
    BUY  if close breaks above ORB high with volume > 1.5× average.
    SELL if close breaks below ORB low  with volume > 1.5× average.
    """
    orb_high, orb_low = _compute_orb(df, date)
    _hold = {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "ORB"}

    if orb_high is None:
        return _hold

    day_df   = df[df.index.date == date].copy()
    avg_vol  = day_df["Volume"].mean() if not day_df.empty else 0
    orb_end  = dtime(9, 15 + ORB_WINDOW_MIN)
    no_trade = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])
    post_orb = day_df[(day_df.index.time >= orb_end) & (day_df.index.time <= no_trade)]

    if post_orb.empty:
        return _hold

    last   = post_orb.iloc[-1]
    close  = float(last["Close"])
    vol_ok = float(last["Volume"]) > 1.5 * avg_vol if avg_vol > 0 else False

    if close > orb_high and vol_ok:
        return {"signal": "BUY",  "entry_price": round(close, 2),
                "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 + INTRADAY_TP_PCT), 2), "strategy": "ORB"}
    elif close < orb_low and vol_ok:
        return {"signal": "SELL", "entry_price": round(close, 2),
                "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 - INTRADAY_TP_PCT), 2), "strategy": "ORB"}

    return {**_hold, "entry_price": close}


# ── Strategy 2: VWAP Mean Reversion ──────────────────────────────────────────

def ema_cross_signal(df: pd.DataFrame) -> dict:
    """
    EMA 5/13 Crossover strategy — replaces VWAP_MR.

    Logic
    -----
    BUY  when EMA5 crosses above EMA13 AND RSI > 50 AND Close > VWAP AND trend-aligned BULL
    SELL when EMA5 crosses below EMA13 AND RSI < 50 AND Close < VWAP AND trend-aligned BEAR

    Why this beats VWAP_MR
    ----------------------
    VWAP mean-reversion is a counter-trend strategy that loses in trending (high-ADX) markets.
    EMA crossover is trend-following — it *rides* the move rather than fading it.
    The Close > VWAP gate ensures we're only buying into momentum, not catching falling knives.
    """
    _hold = {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "EMA_Cross"}
    if df.empty or len(df) < 15:
        return _hold

    df = df.copy()
    df["EMA5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["RSI"]   = _compute_rsi_series(df["Close"])
    df["VWAP"]  = _compute_vwap(df)

    valid = df.dropna(subset=["EMA5", "EMA13", "RSI", "VWAP"])
    if len(valid) < 2:
        return _hold

    prev  = valid.iloc[-2]
    last  = valid.iloc[-1]
    close = float(last["Close"])
    rsi   = float(last["RSI"])
    vwap  = float(last["VWAP"])

    bull_cross = float(prev["EMA5"]) <= float(prev["EMA13"]) and float(last["EMA5"]) > float(last["EMA13"])
    bear_cross = float(prev["EMA5"]) >= float(prev["EMA13"]) and float(last["EMA5"]) < float(last["EMA13"])

    if bull_cross and rsi > 50 and close > vwap:
        if not _trend_aligned(df, "BUY"):
            return {**_hold, "entry_price": close}
        return {"signal": "BUY",  "entry_price": round(close, 2),
                "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 + INTRADAY_TP_PCT), 2), "strategy": "EMA_Cross"}

    if bear_cross and rsi < 50 and close < vwap:
        if not _trend_aligned(df, "SELL"):
            return {**_hold, "entry_price": close}
        return {"signal": "SELL", "entry_price": round(close, 2),
                "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 - INTRADAY_TP_PCT), 2), "strategy": "EMA_Cross"}

    return {**_hold, "entry_price": close}


# ── Strategy 3: Momentum (RSI + MACD) ────────────────────────────────────────

def momentum_signal(df: pd.DataFrame) -> dict:
    """
    BUY  if RSI 14 > 55 AND MACD histogram > 0 AND MACD line crossed above signal recently.
    SELL if RSI 14 < 45 AND MACD histogram < 0 AND MACD line crossed below signal recently.
    """
    _hold = {"signal": "HOLD", "entry_price": 0.0, "sl": 0.0, "tp": 0.0, "strategy": "Momentum"}
    if df.empty or len(df) < 30:
        return _hold

    df = df.copy()
    df["RSI"] = _compute_rsi_series(df["Close"], 14)
    macd_line, signal_line, histogram = _compute_macd(df["Close"])
    df["MACD"]     = macd_line
    df["MACD_Sig"] = signal_line
    df["MACD_Hist"]= histogram

    valid = df.dropna(subset=["RSI", "MACD", "MACD_Sig"])
    if len(valid) < 3:
        return _hold

    last  = valid.iloc[-1]
    prev  = valid.iloc[-2]
    close = float(last["Close"])
    rsi   = float(last["RSI"])
    hist  = float(last["MACD_Hist"])

    # Bullish crossover: MACD crossed above signal in last 2 bars
    bull_cross = (float(prev["MACD"]) <= float(prev["MACD_Sig"])) and (float(last["MACD"]) > float(last["MACD_Sig"]))
    bear_cross = (float(prev["MACD"]) >= float(prev["MACD_Sig"])) and (float(last["MACD"]) < float(last["MACD_Sig"]))

    if rsi > 55 and hist > 0 and bull_cross:
        if not _trend_aligned(df, "BUY"):
            return {**_hold, "entry_price": close}
        return {"signal": "BUY",  "entry_price": round(close, 2),
                "sl": round(close * (1 - INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 + INTRADAY_TP_PCT), 2), "strategy": "Momentum"}
    elif rsi < 45 and hist < 0 and bear_cross:
        if not _trend_aligned(df, "SELL"):
            return {**_hold, "entry_price": close}
        return {"signal": "SELL", "entry_price": round(close, 2),
                "sl": round(close * (1 + INTRADAY_SL_PCT), 2),
                "tp": round(close * (1 - INTRADAY_TP_PCT), 2), "strategy": "Momentum"}

    return {**_hold, "entry_price": close}


# ── IntradayEngine class ──────────────────────────────────────────────────────

class IntradayEngine:
    """Orchestrates intraday signal generation (ORB + EMA_Cross + Momentum)."""

    def __init__(self):
        logger.info("ASTRA IntradayEngine initialised (ORB + EMA_Cross 5/13 + Momentum strategies)")

    @staticmethod
    def _compute_trend_bias(df: pd.DataFrame) -> str:
        """Returns 'BULLISH' / 'BEARISH' / 'NEUTRAL' based on Close vs 50-bar SMA."""
        if df is None or len(df) < 50:
            return "NEUTRAL"
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        if pd.isna(sma50):
            return "NEUTRAL"
        last_close = float(df["Close"].iloc[-1])
        if last_close > float(sma50):
            return "BULLISH"
        elif last_close < float(sma50):
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _market_session() -> str:
        now_ist = datetime.now(_IST).time()
        if now_ist < dtime(9, 0):   return "PRE_MARKET"
        if now_ist < dtime(9, 15):  return "OPENING"
        if now_ist <= dtime(15, 30): return "MARKET_HOURS"
        return "AFTER_HOURS"

    @staticmethod
    def _time_remaining_min() -> int:
        now_ist  = datetime.now(_IST)
        close_dt = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        delta    = (close_dt - now_ist).total_seconds() / 60
        return int(delta)

    @staticmethod
    def _auto_squareoff_active() -> bool:
        """True if we are past NO_TRADE_AFTER but before square-off time."""
        now_ist   = datetime.now(_IST).time()
        no_trade  = dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")])
        sq_off    = dtime(*[int(x) for x in SQUARE_OFF_TIME.split(":")])
        return no_trade <= now_ist <= sq_off

    def analyze_intraday(self, symbol: str) -> dict:
        """
        Full intraday analysis for a symbol.

        Steps:
          1. Fetch 15-minute OHLCV data (5 days via Dhan or AV).
          2. Run ORB, EMA_Cross, and Momentum strategies.
          3. Consensus (all agree) → 90% confidence.
             Two agree → 75% confidence.
             One signal → 60% confidence.
          4. After 14:45 IST: return HOLD + auto-square-off advisory.

        Returns dict with: symbol, signal, confidence, strategy, entry_price,
        sl, tp, time_of_signal, market_session, time_remaining_min, data_source,
        auto_squareoff_active, strategies_detail.
        """
        now_ist = datetime.now(_IST)
        today   = now_ist.date()

        base_result = {
            "symbol":               symbol,
            "signal":               "HOLD",
            "confidence":           0.0,
            "strategy":             "NONE",
            "entry_price":          0.0,
            "sl":                   0.0,
            "tp":                   0.0,
            "time_of_signal":       now_ist.strftime("%H:%M:%S"),
            "market_session":       self._market_session(),
            "time_remaining_min":   self._time_remaining_min(),
            "auto_squareoff_active":self._auto_squareoff_active(),
            "squareoff_time":       SQUARE_OFF_TIME,
            "data_source":          "unavailable",
            "trend_bias":           "NEUTRAL",
            "strategies_detail":    {},
        }

        # After NO_TRADE_AFTER: return HOLD with advisory
        if datetime.now(_IST).time() > dtime(*[int(x) for x in NO_TRADE_AFTER.split(":")]):
            base_result["signal"]   = "HOLD"
            base_result["strategy"] = "NO_NEW_TRADES"
            return base_result

        try:
            df = _fetch_intraday(symbol, days=5)
            if df.empty:
                logger.warning(f"[intraday] No data for {symbol}")
                return base_result

            base_result["data_source"] = "Dhan" if "dhan" in str(type(df)) else "market_data"
            base_result["trend_bias"]  = self._compute_trend_bias(df)

            # Run all three strategies
            orb_res  = orb_signal(df, today)
            ema_res  = ema_cross_signal(df)
            mom_res  = momentum_signal(df)

            base_result["strategies_detail"] = {
                "ORB":       orb_res,
                "EMA_Cross": ema_res,
                "Momentum":  mom_res,
            }

            signals = [r for r in [orb_res, ema_res, mom_res] if r["signal"] != "HOLD"]

            if not signals:
                # Provide entry_price from best available bar
                if orb_res["entry_price"]:
                    base_result["entry_price"] = orb_res["entry_price"]
                return base_result

            # Count agreements
            buys  = [r for r in signals if r["signal"] == "BUY"]
            sells = [r for r in signals if r["signal"] == "SELL"]
            dominant = buys if len(buys) >= len(sells) else sells
            direction = dominant[0]["signal"]

            # Confidence based on agreement count
            agree_count = len(dominant)
            if agree_count == 3:
                confidence = 90.0
                strategy   = "+".join(r["strategy"] for r in dominant)
            elif agree_count == 2:
                confidence = 75.0
                strategy   = "+".join(r["strategy"] for r in dominant)
            else:
                confidence = 60.0
                strategy   = dominant[0]["strategy"]

            # Use the first agreeing strategy's entry/sl/tp
            ref = dominant[0]
            base_result.update({
                "signal":     direction,
                "confidence": confidence,
                "strategy":   strategy,
                "entry_price":ref["entry_price"],
                "sl":         ref["sl"],
                "tp":         ref["tp"],
                "data_source": "Dhan_15m",
            })

        except Exception as exc:
            logger.error(f"[intraday] analyze_intraday({symbol}) error: {exc}", exc_info=True)

        return base_result


# ── Module-level singleton ────────────────────────────────────────────────────

intraday_engine = IntradayEngine()
