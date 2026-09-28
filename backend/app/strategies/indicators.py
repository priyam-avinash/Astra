"""
Shared technical indicator helpers used by ASTRA strategies.
All functions are pure (no side effects) and operate on pandas Series/DataFrames.
"""
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range via EWM smoothing."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_supertrend(df: pd.DataFrame, atr_period: int = 10,
                       multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend indicator.
    Returns a boolean Series: True = bullish (price above Supertrend line).
    Uses iterative logic to maintain band memory across bars.
    """
    atr   = compute_atr(df, atr_period)
    hl2   = (df["High"] + df["Low"]) / 2.0
    close = df["Close"]

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(df)
    upper_arr = upper_basic.values.copy()
    lower_arr = lower_basic.values.copy()
    close_arr = close.values

    for i in range(1, n):
        upper_arr[i] = (upper_basic.iloc[i]
                        if upper_basic.iloc[i] < upper_arr[i-1]
                           or close_arr[i-1] > upper_arr[i-1]
                        else upper_arr[i-1])
        lower_arr[i] = (lower_basic.iloc[i]
                        if lower_basic.iloc[i] > lower_arr[i-1]
                           or close_arr[i-1] < lower_arr[i-1]
                        else lower_arr[i-1])

    trend = np.zeros(n, dtype=bool)

    for i in range(1, n):
        prev_bullish = trend[i-1]
        if prev_bullish:
            if close_arr[i] < lower_arr[i]:
                trend[i] = False
            else:
                trend[i] = True
        else:
            if close_arr[i] > upper_arr[i]:
                trend[i] = True
            else:
                trend[i] = False

    return pd.Series(trend, index=df.index, name="supertrend_bullish")


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) via Wilder smoothing.
    Returns ADX values in [0, 100]. Higher = stronger trend.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high - prev_high
    down_move = prev_low - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_dm_s = pd.Series(pos_dm, index=df.index).ewm(alpha=1.0/period, adjust=False).mean()
    neg_dm_s = pd.Series(neg_dm, index=df.index).ewm(alpha=1.0/period, adjust=False).mean()
    atr_s    = tr.ewm(alpha=1.0/period, adjust=False).mean()

    pos_di = 100 * pos_dm_s / atr_s.replace(0, np.nan)
    neg_di = 100 * neg_dm_s / atr_s.replace(0, np.nan)

    di_sum  = pos_di + neg_di
    dx      = (100 * (pos_di - neg_di).abs() / di_sum.replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1.0/period, adjust=False).mean()
    return pd.DataFrame({
        "ATR":      atr_s,
        "ADX":      adx,
        "DI_plus":  pos_di,
        "DI_minus": neg_di,
    }, index=df.index)


def compute_rs(stock_close: pd.Series, bench_close: pd.Series,
               period: int = 20) -> pd.Series:
    """
    Relative Strength of stock vs benchmark over `period` bars.
    Positive = stock outperforming. Both series must share the same index
    or bench_close will be forward-filled to match stock_close.
    """
    bench_aligned = bench_close.reindex(stock_close.index, method="ffill")
    stock_ret = stock_close.pct_change(period)
    bench_ret = bench_aligned.pct_change(period)
    return stock_ret - bench_ret


def compute_hv_rank(close: pd.Series,
                    hv_period: int = 20,
                    rank_lookback: int = 252) -> pd.Series:
    """
    Historical Volatility Rank — proxy for IV Percentile.
    Returns 0.0–1.0: where today's 20-day realised vol sits in its 252-day history.
    Low rank (< 0.40) = calm vol regime → better mean-reversion entries.
    High rank (> 0.70) = elevated vol → gap risk, stop violations likely.
    """
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(hv_period).std() * np.sqrt(252)
    hv_rank = hv.rolling(rank_lookback).rank(pct=True)
    return hv_rank.rename("hv_rank")


def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Typical-price VWAP approximation from daily OHLCV.
    Uses rolling 5-day window: sum(HLC/3 × Vol) / sum(Vol).
    Entry quality filter: next-day open > today's VWAP → institutional demand.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol  = typical * df["Volume"]
    vwap = tp_vol.rolling(5).sum() / df["Volume"].rolling(5).sum()
    return vwap.rename("vwap")
