"""
ASTRA Enhancement Filters — Phase 1 backtestable market-regime gates.

Three filters derived from freely available daily data:
  1. HV Rank   — proxy for IV percentile; skip when vol is elevated
  2. India VIX — macro fear gauge; skip when VIX > threshold
  3. VWAP      — execution quality; skip when next-bar open < VWAP

All filters are OFF by default. Set USE_*_FILTER = True for the enhanced backtest run.
The same flags are read by the evaluator, so toggling them reruns cleanly.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# ── Feature flags (flip to True for enhanced run) ─────────────────────────────
USE_HV_RANK_FILTER = False
HV_RANK_THRESHOLD  = 0.45     # skip if 20-day HV is above 45th pct of its 252-day history

USE_VIX_FILTER     = False
VIX_MAX            = 20.0     # skip new entries when India VIX > 20

USE_VWAP_FILTER    = False
# skip entry if next bar's open ≤ previous bar's 5-day VWAP (no demand confirmation)

# ── India VIX cache (lazy, once per process) ──────────────────────────────────
_VIX_CACHE: Optional[pd.Series] = None


def _load_india_vix() -> Optional[pd.Series]:
    global _VIX_CACHE
    if _VIX_CACHE is not None:
        return _VIX_CACHE
    try:
        import yfinance as yf
        raw = yf.download("^INDIAVIX", period="3y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].sort_index()
            _VIX_CACHE = s
            return _VIX_CACHE
    except Exception:
        pass
    return None


def _vix_on(date) -> Optional[float]:
    """Return India VIX value on or just before `date`. None if unavailable."""
    vix = _load_india_vix()
    if vix is None:
        return None
    try:
        idx = vix.index.get_indexer([date], method="pad")[0]
        if idx < 0:
            return None
        return float(vix.iloc[idx])
    except Exception:
        return None


# ── Public gate function called from each strategy's signal() ─────────────────

def passes_enhancement_filters(row: pd.Series, prev_open: Optional[float] = None) -> bool:
    """
    Returns False (block signal) if any active filter is triggered.

    Args:
        row:        current bar from enriched DataFrame (has hv_rank, vwap columns if computed)
        prev_open:  next bar's open price (for VWAP filter); pass None to skip VWAP check
    """
    if USE_HV_RANK_FILTER:
        hv_rank = row.get("hv_rank")
        if hv_rank is not None and not pd.isna(hv_rank):
            if float(hv_rank) > HV_RANK_THRESHOLD:
                return False

    if USE_VIX_FILTER:
        date = getattr(row, "name", None)
        if date is not None:
            vix_val = _vix_on(date)
            if vix_val is not None and vix_val > VIX_MAX:
                return False

    if USE_VWAP_FILTER and prev_open is not None:
        vwap = row.get("vwap")
        if vwap is not None and not pd.isna(vwap):
            if float(prev_open) <= float(vwap):
                return False

    return True


def clear_vix_cache() -> None:
    global _VIX_CACHE
    _VIX_CACHE = None
