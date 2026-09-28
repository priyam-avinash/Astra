"""
ASTRA Broker Plugin Abstract Layer
====================================
Defines the unified contract every broker integration must implement.
Concrete clients (Upstox, Dhan, Zerodha, Fyers, Angel) inherit from BrokerClient.

The platform uses this abstraction in three places:
  1. Data fetching: get_ohlcv / get_quote — read-only, no live account needed
  2. Paper trading: simulated execution against real quotes
  3. Live execution: place_order / cancel_order — gated by PaperTradingEngine

Live execution is BLOCKED at the broker_service singleton level (see broker.py).
This abstraction allows the API to remain uniform regardless of mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ── Enums and value objects ─────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"   # MIS — squared off same day
    DELIVERY = "DELIVERY"   # CNC — held overnight


@dataclass
class OrderRequest:
    symbol:        str
    side:          OrderSide
    quantity:      int
    order_type:    OrderType = OrderType.MARKET
    limit_price:   Optional[float] = None
    product_type:  ProductType = ProductType.INTRADAY


@dataclass
class OrderResult:
    """Result returned by place_order. Always includes a broker-specific order_id."""
    order_id:       str
    status:         str          # "submitted", "filled", "rejected"
    filled_price:   Optional[float] = None
    filled_qty:     int           = 0
    rejection_msg:  Optional[str] = None
    broker:         str           = ""
    mode:           str           = "PAPER"   # "PAPER" | "LIVE"


@dataclass
class BrokerPosition:
    symbol:        str
    side:          OrderSide
    quantity:      int
    avg_price:     float
    last_price:    Optional[float] = None
    pnl_unrealized: float = 0.0


# ── Abstract client ────────────────────────────────────────────────────────

class BrokerClient(ABC):
    """
    Abstract broker. Subclasses MUST implement all data methods.
    `place_order` defaults to NotImplementedError so paper-trading-only
    clients (Yahoo, Stooq) don't need to define it.
    """

    # Identity
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable broker name, e.g. 'Upstox', 'Dhan', 'Zerodha'."""

    @abstractmethod
    def is_available(self) -> bool:
        """True iff credentials are present and the API is reachable."""

    # Data (read-only)
    @abstractmethod
    def get_ohlcv(self, symbol: str, period: str = "1mo",
                  interval: str = "1d") -> pd.DataFrame:
        """Historical OHLCV. Returns empty DataFrame on failure (no exceptions)."""

    def get_quote(self, symbol: str) -> Optional[float]:
        """Real-time last-traded price. Optional override — default returns None."""
        return None

    # Trading (optional — paper-only clients raise NotImplementedError)
    def place_order(self, req: OrderRequest) -> OrderResult:
        """Place a live order. Raises NotImplementedError if not supported."""
        raise NotImplementedError(
            f"{self.name} does not support live order execution via this plugin"
        )

    def get_positions(self) -> list[BrokerPosition]:
        """Return live positions. Default = empty (paper-only clients)."""
        return []

    # Auth flow (browser-based, OAuth2 — same pattern as Upstox)
    def get_login_url(self) -> Optional[str]:
        """Return OAuth login URL for browser flow, or None if not needed."""
        return None

    def exchange_code(self, code: str) -> Optional[str]:
        """Exchange OAuth code for an access token. Returns token or None."""
        return None
