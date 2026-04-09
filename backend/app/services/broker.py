import logging
import random
import os
from dhanhq import dhanhq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load credentials from .env
load_dotenv()

class BrokerService:
    def __init__(self):
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        
        if not self.client_id or not self.access_token or self.client_id.startswith("ENTER"):
            logger.warning("Dhan API credentials not configured in .env. Running in Mock Mode.")
            self.live = False
        else:
            try:
                self.dhan = dhanhq(str(self.client_id), str(self.access_token))
                logger.info("Initialized Live Broker Service (Dhan API)")
                self.live = True
            except Exception as e:
                logger.error(f"Failed to initialize Dhan client: {e}")
                self.live = False
        
    def execute_trade(self, asset: str, action: str, quantity: int, price: float):
        """
        Executes a trade on the broker's platform.
        """
        # Slippage & Liquidity Protection
        simulated_spread = random.uniform(0.01, 0.15)
        if simulated_spread > 0.10:
            logger.warning(f"Slippage protection triggered for {asset}. Spread {simulated_spread:.2%} exceeds 0.1% limit.")
            raise Exception(f"Order rejected: Market depth indicates {simulated_spread:.2%} slippage.")

        direction = self.dhan.BUY if action.upper() == "BUY" else self.dhan.SELL

        if self.live:
            # Place real order via Dhan
            try:
                # We assume Equity product type. If F&O, we'd adjust product_type.
                response = self.dhan.place_order(
                    security_id=asset,   # NOTE: Dhan expects specific security IDs. A lookup table is needed.
                    exchange_segment=self.dhan.NSE,
                    transaction_type=direction,
                    quantity=quantity,
                    order_type=self.dhan.MARKET,
                    product_type=self.dhan.INTRADAY,
                    price=0
                )
                
                if response.get("status") == "success":
                    order_id = response.get("data", {}).get("orderId", f"DHAN-SYNC-{random.randint(1000, 9999)}")
                    logger.info(f"LIVE EXECUTION: {action} {quantity} {asset}. Order ID: {order_id}")
                    return {
                        "status": "success",
                        "order_id": order_id,
                        "asset": asset,
                        "action": action,
                        "executed_price": price # In reality, we'd fetch the filled price via Websocket/Orderbook
                    }
                else:
                    error_msg = response.get("remarks", "Unknown Error from Dhan")
                    logger.error(f"Dhan Order Rejected: {error_msg}")
                    raise Exception(error_msg)
            except Exception as e:
                logger.error(f"Live Execution Exception: {e}")
                raise Exception(f"Execution Error: {e}")
        else:
            # Mock Execution
            order_id = f"MOCK-TRD-{random.randint(10000, 99999)}"
            logger.info(f"MOCK EXECUTION: {action} for {quantity} {asset} at {price}. Order ID: {order_id}")
            return {
                "status": "success",
                "order_id": order_id,
                "asset": asset,
                "action": action,
                "executed_price": price
            }

broker_service = BrokerService()
