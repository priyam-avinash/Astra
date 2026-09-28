"""
ASTRA Paper Trading Engine
===========================
Simulates order execution at REAL market prices (0.05% slippage, 0.03% brokerage).

PAPER TRADING ONLY. As of v1.13 there is no live-order code path in ASTRA:
the Dhan live broker stub was removed, and PAPER_TRADING=false is ignored.
Dhan credentials, if configured, are used for market DATA only.
"""

import logging
import os
import random
import time
from datetime import datetime

import app.core.config  # noqa: F401  (loads .env)

logger = logging.getLogger(__name__)

PAPER_MODE = True   # hard-wired; there is no live mode

if os.getenv("PAPER_TRADING", "true").strip().lower() == "false":
    logger.warning("PAPER_TRADING=false is ignored — ASTRA only supports paper trading.")


# ── Paper Trading Engine ───────────────────────────────────────────────────
class PaperTradingEngine:
    """
    Executes simulated trades at real market prices with minimal slippage.
    Tracks every order in memory for the session. Persisted P&L lives in the DB.
    """
    SLIPPAGE    = 0.0005   # 0.05% — realistic for large-cap NSE stocks
    BROKERAGE   = 0.0003   # 0.03% per leg (Zerodha-equivalent flat fee)

    def __init__(self):
        self.mode    = "PAPER"
        self._orders = []  # type: list
        logger.info("📝 ASTRA Paper Trading Engine initialised — no real money involved")

    def _get_ltp(self, asset: str, fallback: float) -> float:
        """Live LTP from the market-data layer; raises if only stale data is available."""
        from app.services.market_data import market_data
        info = market_data.get_quote_info(asset)
        if info["price"] > 0 and not info["stale"]:
            return float(info["price"])
        # Never fill at a stale cached price or at the (possibly days-old) signal
        # price: that creates fake P&L. Reject instead and say why.
        raise ValueError(
            f"No live price for {asset} right now (all data providers failing"
            f"{'; last cached ' + str(info['price']) if info['price'] else ''}). "
            "Order not filled. Check Settings → Market Data Providers.")

    def _fill_price(self, asset: str, action: str, user_price: float) -> float:
        """LTP ± slippage depending on order direction."""
        ltp = self._get_ltp(asset, user_price)
        if action.upper() == "BUY":
            return round(ltp * (1 + self.SLIPPAGE), 2)   # pay slightly above LTP
        else:
            return round(ltp * (1 - self.SLIPPAGE), 2)   # receive slightly below LTP

    def get_exit_price(self, asset: str, direction: str) -> float:
        """
        Real LTP for position exit. Used by squareoff and auto-exit logic.
        BUY exits at bid (LTP - slippage). SELL exits at ask (LTP + slippage).
        Returns 0.0 if price is unavailable.
        """
        try:
            from app.services.market_data import market_data
            ltp = market_data.get_quote(asset)
            if ltp and ltp > 0:
                if direction == "BUY":
                    return round(ltp * (1 - self.SLIPPAGE), 2)
                else:
                    return round(ltp * (1 + self.SLIPPAGE), 2)
        except Exception as e:
            logger.warning(f"Exit price fetch failed for {asset}: {e}")
        return 0.0

    def execute_trade(self, asset: str, action: str, quantity: int, price: float) -> dict:
        """Place a paper order. Returns order dict with executed_price from real LTP."""
        if quantity < 1:
            raise ValueError(f"Invalid quantity {quantity} for {asset}")

        fill      = self._fill_price(asset, action, price)
        brokerage = round(fill * quantity * self.BROKERAGE, 2)
        order_id  = f"PAPER-{int(time.time())}-{random.randint(1000, 9999)}"

        order = {
            "order_id":        order_id,
            "asset":           asset,
            "action":          action.upper(),
            "quantity":        quantity,
            "requested_price": price,
            "executed_price":  fill,
            "brokerage":       brokerage,
            "mode":            "PAPER",
            "timestamp":       datetime.utcnow().isoformat(),
        }
        self._orders.append(order)

        logger.info(
            f"📝 PAPER | {action.upper():4s} {quantity:>5} × {asset:<20s} "
            f"@ ₹{fill:>10.2f}  (requested ₹{price:.2f}  brok ₹{brokerage:.2f})"
        )
        return {
            "status":         "success",
            "order_id":       order_id,
            "asset":          asset,
            "action":         action.upper(),
            "executed_price": fill,
            "brokerage":      brokerage,
            "mode":           "PAPER",
        }

    def get_order_history(self) -> list[dict]:
        return list(self._orders)

    def get_status(self) -> dict:
        return {
            "mode":          "PAPER",
            "paper_trading": True,
            "total_orders":  len(self._orders),
            "message":       "Paper trading active — no real money at risk",
        }


# ── Live broker: permanently blocked ───────────────────────────────────────
class DhanLiveBroker:
    """Kept only as a tripwire for safety tests. Contains no order code: every
    call raises, whatever credentials or env vars are set."""
    mode = "BLOCKED"

    def execute_trade(self, *args, **kwargs):
        raise RuntimeError(
            "LIVE ORDER BLOCKED: ASTRA is in permanent paper-trading mode. "
            "Real order execution is disabled. All trades are simulated.")

    def get_exit_price(self, *args, **kwargs):
        raise RuntimeError("LIVE ORDER BLOCKED: ASTRA is paper-trading only.")

    def get_status(self) -> dict:
        return {"mode": "BLOCKED", "paper_trading": True}


# ── Module singleton ───────────────────────────────────────────────────────
broker_service = PaperTradingEngine()
logger.info("Broker mode: 📝 PAPER TRADING (live orders disabled in code)")
