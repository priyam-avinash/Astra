"""
ASTRA Paper Trading Engine
===========================
Simulates order execution using REAL market prices fetched via the AI engine.
No real money is ever touched. All fills use live LTP with realistic 0.05% slippage.

Switch to live Dhan execution by setting PAPER_TRADING=false in .env and providing
valid DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN. The Dhan security-ID lookup table must
also be populated before live orders will succeed.
"""

import logging
import random
import os
import time
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

PAPER_MODE = os.getenv("PAPER_TRADING", "true").lower() != "false"


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
        """Fetch real LTP from the AI engine. Fall back to user_price if unavailable."""
        try:
            from app.services.ai_predictor import ai_engine
            ltp = ai_engine.get_realtime_price(asset)
            if ltp and ltp > 0:
                return float(ltp)
        except Exception as e:
            logger.warning(f"LTP fetch failed for {asset}: {e}")
        return fallback

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
            from app.services.ai_predictor import ai_engine
            ltp = ai_engine.get_realtime_price(asset)
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


# ── Live Dhan Engine (stub — requires security IDs + credentials) ──────────
class DhanLiveBroker:
    """
    Placeholder for live Dhan HQ execution.
    NOT SAFE FOR PRODUCTION until SECURITY_ID_MAP is populated.

    To activate:
      1. Set PAPER_TRADING=false in .env
      2. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env
      3. Download https://images.dhan.co/api-data/api-scrip-master.csv
         and populate SECURITY_ID_MAP with (symbol → numeric_id) pairs.
    """

    SECURITY_ID_MAP: dict[str, str] = {
        "RELIANCE.NS":  "1333",
        "TCS.NS":       "11536",
        "HDFCBANK.NS":  "1330",
        "INFY.NS":      "1594",
        "ICICIBANK.NS": "4963",
        # TODO: add all 141 NSE symbols from Dhan scrip master CSV
    }

    def __init__(self):
        self.client_id    = os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN", "")
        self.live = False
        self.mode = "LIVE"
        if self.client_id and not self.client_id.startswith("ENTER"):
            try:
                from dhanhq import dhanhq
                self.dhan = dhanhq(str(self.client_id), str(self.access_token))
                self.live = True
                logger.info("🔴 LIVE Dhan broker initialised")
            except Exception as e:
                logger.error(f"Dhan init failed: {e}")

    def get_exit_price(self, asset: str, direction: str) -> float:
        return 0.0  # Would use Dhan quote API in production

    def execute_trade(self, asset: str, action: str, quantity: int, price: float) -> dict:
        if not self.live:
            raise RuntimeError("Dhan credentials not configured")
        sec_id = self.SECURITY_ID_MAP.get(asset)
        if not sec_id:
            raise ValueError(
                f"No Dhan security_id for {asset}. "
                "Add it to SECURITY_ID_MAP or populate from the scrip master CSV."
            )
        from dhanhq import dhanhq
        direction_flag = self.dhan.BUY if action.upper() == "BUY" else self.dhan.SELL
        response = self.dhan.place_order(
            security_id=sec_id,
            exchange_segment=self.dhan.NSE,
            transaction_type=direction_flag,
            quantity=quantity,
            order_type=self.dhan.MARKET,
            product_type=self.dhan.CNC,   # CNC = delivery (not intraday)
            price=0,
        )
        if response.get("status") == "success":
            order_id = response.get("data", {}).get("orderId", "DHAN-UNKNOWN")
            return {"status": "success", "order_id": order_id, "asset": asset,
                    "action": action.upper(), "executed_price": price, "mode": "LIVE"}
        raise Exception(response.get("remarks", "Dhan order rejected"))

    def get_status(self) -> dict:
        return {"mode": "LIVE", "paper_trading": False}


# ── Module singleton ───────────────────────────────────────────────────────
broker_service = PaperTradingEngine() if PAPER_MODE else DhanLiveBroker()
logger.info(f"Broker mode: {'📝 PAPER TRADING' if PAPER_MODE else '🔴 LIVE TRADING'}")
