from celery import Celery
from celery.schedules import crontab
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Setup Celery
celery_app = Celery("astra_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    "monitor-positions-every-15-seconds": {
        "task": "monitor_active_positions",
        "schedule": 15.0,
    },
}

@celery_app.task(name="analyze_asset", bind=True)
def run_ai_analysis(self, symbol: str, interval: str = "1d", period: str = "6mo"):
    """
    Offline celery task to process heavy ML calculations instead of blocking the FastAPI server thread.
    """
    from app.services.ai_predictor import ai_engine
    try:
        result = ai_engine.analyze_market_data(symbol, period=period, interval=interval)
        return result
    except Exception as e:
        logger.error(f"Analysis task failed for {symbol}: {e}")
        return {"error": str(e), "signal": "HOLD"}

@celery_app.task(name="monitor_active_positions")
def monitor_active_positions():
    """
    Background worker that checks SL/TP for all open positions.
    """
    from app.models.database import SessionLocal, ActivePosition, TradeRecord
    from app.services.ai_predictor import ai_engine
    from app.services.broker import broker_service
    
    db = SessionLocal()
    try:
        # 1. Fetch all open positions
        positions = db.query(ActivePosition).filter(ActivePosition.status == "OPEN").all()
        if not positions:
            return "No open positions to monitor."

        results = []
        for p in positions:
            # 2. Get current price
            current_price = ai_engine.get_realtime_price(p.asset)
            if current_price == 0.0:
                continue

            # Trailing Stop Loss (TSL): Auto-Breakeven at 50% Target
            if p.target_price and p.entry_price:
                halfway = abs(p.target_price - p.entry_price) * 0.5
                if p.direction == "BUY" and current_price >= p.entry_price + halfway:
                    if not p.stop_loss or p.stop_loss < p.entry_price:
                        p.stop_loss = p.entry_price
                        db.commit()
                        logger.info(f"TSL Activated for {p.asset}: SL moved to breakeven {p.entry_price}")
                elif p.direction == "SELL" and current_price <= p.entry_price - halfway:
                    if not p.stop_loss or p.stop_loss > p.entry_price:
                        p.stop_loss = p.entry_price
                        db.commit()
                        logger.info(f"TSL Activated for {p.asset}: SL moved to breakeven {p.entry_price}")

            # 3. Check SL/TP
            triggered = False
            exit_reason = ""
            
            if p.direction == "BUY":
                if p.target_price and current_price >= p.target_price:
                    triggered = True; exit_reason = "Target Hit (Auto)"
                elif p.stop_loss and current_price <= p.stop_loss:
                    triggered = True; exit_reason = "Stop Loss Hit (Auto)"
            else: # SELL
                if p.target_price and current_price <= p.target_price:
                    triggered = True; exit_reason = "Target Hit (Auto)"
                elif p.stop_loss and current_price >= p.stop_loss:
                    triggered = True; exit_reason = "Stop Loss Hit (Auto)"

            if triggered:
                # 4. Execute Square Off
                exit_action = "SELL" if p.direction == "BUY" else "BUY"
                try:
                    broker_res = broker_service.execute_trade(
                        asset=p.asset,
                        action=exit_action,
                        quantity=p.quantity,
                        price=current_price
                    )
                    
                    # Update DB
                    p.status = "CLOSED"
                    p.exit_price = current_price
                    p.exit_time = datetime.utcnow()
                    
                    pnl = (current_price - p.entry_price) * p.quantity if p.direction == "BUY" else (p.entry_price - current_price) * p.quantity
                    
                    new_history = TradeRecord(
                        user_id=p.user_id,
                        trade_id=f"AUTO-{p.id}-{datetime.now().strftime('%M%S')}",
                        asset=p.asset,
                        action=f"{exit_action} ({exit_reason})",
                        price=current_price,
                        quantity=p.quantity,
                        pnl=round(pnl, 2),
                        status="Closed (Worker)"
                    )
                    db.add(new_history)
                    db.commit()
                    logger.info(f"AUTO-EXIT: {p.asset} {exit_reason} at {current_price}")
                    results.append(f"Closed {p.asset} due to {exit_reason}")
                except Exception as e:
                    logger.error(f"Failed to auto-exit {p.asset}: {e}")
                    db.rollback()
        
        return results if results else "Monitor cycle complete. No exits triggered."
        
    finally:
        db.close()
