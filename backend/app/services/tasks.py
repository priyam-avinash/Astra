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

@celery_app.task(name="reflect_on_closed_trade", bind=True)
def reflect_on_closed_trade(
    self,
    trade_id:       int,
    symbol:         str,
    original_signal: str,
    confidence:     float,
    entry_price:    float,
    exit_price:     float,
    pnl:            float,
    pnl_pct:        float,
    exit_reason:    str,
    news_sentiment: str = "NEUTRAL",
    fund_grade:     str = "C",
    debate_verdict: str = "CONFIRMED",
    risk_verdict:   str = "GREEN",
):
    """
    Async task: generate a reflection for a closed trade and store it + lessons.
    Triggered automatically after every position close.
    """
    from app.models.database import SessionLocal, TradeRecord
    from app.services.reflector_agent import reflector_agent
    from datetime import datetime

    db = SessionLocal()
    try:
        reflection_text = reflector_agent.reflect(
            symbol=symbol,
            original_signal=original_signal,
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            news_sentiment=news_sentiment,
            fund_grade=fund_grade,
            debate_verdict=debate_verdict,
            risk_verdict=risk_verdict,
            trade_id=trade_id,
            db=db,
        )

        # Store reflection on the TradeRecord
        trade = db.query(TradeRecord).filter(TradeRecord.id == trade_id).first()
        if trade:
            trade.reflection      = reflection_text
            trade.reflection_at   = datetime.utcnow()
            trade.outcome_return_pct = round(pnl_pct, 3)
            db.commit()
            logger.info(f"Reflection stored for trade {trade_id} ({symbol}): {reflection_text[:60]}…")

        return {"trade_id": trade_id, "symbol": symbol, "reflection": reflection_text}

    except Exception as e:
        logger.error(f"Reflection task failed for trade {trade_id}: {e}")
        return {"error": str(e)}
    finally:
        db.close()


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
                    pnl_pct = ((current_price - p.entry_price) / p.entry_price) * (1 if p.direction == "BUY" else -1) * 100

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
                    db.refresh(new_history)
                    logger.info(f"AUTO-EXIT: {p.asset} {exit_reason} at {current_price}")
                    results.append(f"Closed {p.asset} due to {exit_reason}")

                    # ── Trigger async reflection / learning ──────────────
                    try:
                        reflect_on_closed_trade.delay(
                            trade_id=new_history.id,
                            symbol=p.asset,
                            original_signal=p.direction,
                            confidence=50.0,   # best effort
                            entry_price=p.entry_price,
                            exit_price=current_price,
                            pnl=round(pnl, 2),
                            pnl_pct=round(pnl_pct, 3),
                            exit_reason=exit_reason,
                        )
                    except Exception as rf_err:
                        logger.warning(f"Reflection task dispatch failed (non-critical): {rf_err}")

                    # ── Push to replay buffer for self-learning ──────────────
                    try:
                        from app.services.replay_buffer import replay_buffer
                        entry_features = {}
                        if hasattr(p, "entry_features_json") and p.entry_features_json:
                            import json as _json
                            entry_features = _json.loads(p.entry_features_json)
                        replay_buffer.push(
                            symbol=p.asset,
                            features=entry_features,
                            outcome={
                                "pnl_pct":    round(pnl_pct, 3),
                                "hit_target": "Target" in exit_reason,
                                "hit_sl":     "Stop" in exit_reason,
                                "bars_held":  0,   # Could compute from entry_time
                                "direction":  p.direction,
                            },
                            engine=getattr(p, "engine", "unknown"),
                        )
                        # Check for drift and schedule emergency retrain if needed
                        drift = replay_buffer.drift_check()
                        if drift["drift_detected"]:
                            logger.warning(
                                f"🚨 DRIFT DETECTED: recent WR={drift['recent_win_rate']:.1f}% "
                                f"vs overall={drift['overall_win_rate']:.1f}%. "
                                f"Consider retraining."
                            )
                    except Exception as rb_err:
                        logger.warning(f"Replay buffer push failed (non-critical): {rb_err}")

                except Exception as e:
                    logger.error(f"Failed to auto-exit {p.asset}: {e}")
                    db.rollback()
        
        return results if results else "Monitor cycle complete. No exits triggered."
        
    finally:
        db.close()
