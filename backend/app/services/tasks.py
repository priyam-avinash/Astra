from celery import Celery
from celery.schedules import crontab
import os
import logging
import threading
import time
from datetime import datetime

import app.core.config  # noqa: F401  (loads .env)

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"

_BROKER_CHECK = {"ok": None, "at": 0.0}


def broker_available() -> bool:
    """Is Redis (Celery broker) reachable? Cached 60s; 1s connect timeout."""
    from app.core.config import SERVERLESS
    if SERVERLESS and not os.getenv("REDIS_URL"):
        return False
    now = time.time()
    if _BROKER_CHECK["ok"] is not None and now - _BROKER_CHECK["at"] < 60:
        return _BROKER_CHECK["ok"]
    ok = False
    try:
        import redis
        ok = bool(redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1).ping())
    except Exception:
        ok = False
    _BROKER_CHECK.update(ok=ok, at=now)
    return ok


def dispatch(task, **kwargs):
    """Queue on Celery when Redis is up; otherwise run in a background thread.
    (Calling .delay() with Redis down blocks on kombu's connection retries.)"""
    if broker_available():
        try:
            return task.apply_async(kwargs=kwargs, retry=False)
        except Exception as e:
            logger.warning(f"Celery dispatch failed ({e}); running {task.name} in-process")
    threading.Thread(target=lambda: task(**kwargs), name=f"task-{task.name}", daemon=True).start()
    return None

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
    position_id:    int = None,
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

        # Store reflection + label closed trade for self-learning
        trade = db.query(TradeRecord).filter(TradeRecord.id == trade_id).first()
        if trade:
            trade.reflection         = reflection_text
            trade.reflection_at      = datetime.utcnow()
            trade.outcome_return_pct = round(pnl_pct, 3)

            # Derive signal label: was the original signal direction correct?
            # pnl_pct is in PERCENT (e.g. 2.3 = +2.3%). v1.12 compared against
            # 0.003 (a fraction), which labelled almost every trade.
            if pnl_pct > 0.3:            # > +0.3% — clearly correct
                trade.signal_label = "CORRECT"
            elif pnl_pct < -0.3:         # < -0.3% — clearly wrong
                trade.signal_label = "INCORRECT"
            else:
                trade.signal_label = "NEUTRAL"  # scratch / noise

            # Copy entry features from the matching ActivePosition (if captured)
            try:
                from app.models.database import ActivePosition
                if position_id is not None:
                    pos = db.query(ActivePosition).filter(ActivePosition.id == position_id).first()
                else:
                    pos = db.query(ActivePosition).filter(
                        ActivePosition.asset == symbol, ActivePosition.status == "CLOSED",
                        ActivePosition.user_id == trade.user_id,
                    ).order_by(ActivePosition.exit_time.desc()).first()
                if pos and pos.entry_features_json:
                    trade.entry_features_json = pos.entry_features_json
            except Exception as _fe:
                logger.debug(f"Feature copy failed for trade {trade_id}: {_fe}")

            db.commit()
            logger.info(
                f"Trade {trade_id} ({symbol}) labelled {trade.signal_label} "
                f"({pnl_pct:+.2f}%). Reflection: {reflection_text[:60]}…"
            )

        return {"trade_id": trade_id, "symbol": symbol, "reflection": reflection_text}

    except Exception as e:
        logger.error(f"Reflection task failed for trade {trade_id}: {e}")
        return {"error": str(e)}
    finally:
        db.close()




def check_and_close_position(db, p, current_price: float) -> bool:
    """SL / target / breakeven-trailing check for one OPEN position at a LIVE price.
    Closes it through the shared atomic path. Returns True if it is (being) closed."""
    from app.services.broker import broker_service
    from app.services.positions import claim_position, release_position, book_close

    # Trailing stop: move SL to breakeven once price has covered half the target distance
    if p.target_price and p.entry_price:
        halfway = abs(p.target_price - p.entry_price) * 0.5
        if p.direction == "BUY" and current_price >= p.entry_price + halfway:
            if not p.stop_loss or p.stop_loss < p.entry_price:
                p.stop_loss = p.entry_price
                db.commit()
                logger.info(f"TSL activated for {p.asset}: SL → breakeven {p.entry_price}")
        elif p.direction == "SELL" and current_price <= p.entry_price - halfway:
            if not p.stop_loss or p.stop_loss > p.entry_price:
                p.stop_loss = p.entry_price
                db.commit()
                logger.info(f"TSL activated for {p.asset}: SL → breakeven {p.entry_price}")

    reason = None
    if p.direction == "BUY":
        if p.target_price and current_price >= p.target_price:
            reason = "Target Hit"
        elif p.stop_loss and current_price <= p.stop_loss:
            reason = "Stop Loss Hit"
    else:
        if p.target_price and current_price <= p.target_price:
            reason = "Target Hit"
        elif p.stop_loss and current_price >= p.stop_loss:
            reason = "Stop Loss Hit"
    if not reason:
        return False
    if not claim_position(db, p.id):
        return True                      # another closer won the race
    db.refresh(p)
    exit_action = "SELL" if p.direction == "BUY" else "BUY"
    try:
        res = broker_service.execute_trade(asset=p.asset, action=exit_action, quantity=p.quantity, price=current_price)
        fill = float(res["executed_price"])
        pnl, rec = book_close(db, p, fill, f"{reason} (Auto)", "Closed (Auto)", order=res)
    except Exception as e:
        db.rollback()
        release_position(db, p.id)
        logger.warning(f"Auto-exit of {p.asset} failed ({e}); will retry")
        return False
    logger.info(f"AUTO-EXIT: {p.asset} {reason} at {fill} (pnl {pnl})")
    pnl_pct = ((fill - p.entry_price) / p.entry_price) * (1 if p.direction == "BUY" else -1) * 100
    try:
        dispatch(reflect_on_closed_trade, trade_id=rec.id, symbol=p.asset, original_signal=p.direction,
                 confidence=50.0, entry_price=p.entry_price, exit_price=fill, pnl=pnl,
                 pnl_pct=round(pnl_pct, 3), exit_reason=reason, position_id=p.id)
    except Exception as rf_err:
        logger.debug(f"Reflection dispatch failed (non-critical): {rf_err}")
    try:
        from app.services.replay_buffer import replay_buffer
        import json as _json
        feats = _json.loads(p.entry_features_json) if getattr(p, "entry_features_json", None) else {}
        replay_buffer.push(symbol=p.asset, features=feats,
                           outcome={"pnl_pct": round(pnl_pct, 3), "hit_target": reason == "Target Hit",
                                    "hit_sl": reason == "Stop Loss Hit", "bars_held": 0, "direction": p.direction},
                           engine=getattr(p, "engine", "unknown"))
    except Exception as e:
        logger.debug(f"Replay buffer push failed: {e}")
    return True


@celery_app.task(name="monitor_active_positions")
def monitor_active_positions():
    """Background worker: SL/TP/trailing for all open positions (live prices only)."""
    from app.models.database import SessionLocal, ActivePosition
    from app.services.market_data import market_data
    db = SessionLocal()
    try:
        positions = db.query(ActivePosition).filter(ActivePosition.status == "OPEN").all()
        if not positions:
            return "No open positions to monitor."
        closed = 0
        for p in positions:
            q = market_data.get_quote_info(p.asset)
            if q["price"] <= 0 or q["stale"]:
                continue
            if check_and_close_position(db, p, float(q["price"])):
                closed += 1
        return f"Monitored {len(positions)} positions, closed {closed}."
    finally:
        db.close()
