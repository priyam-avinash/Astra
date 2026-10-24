"""
Closing paper positions — one code path for the background monitor, the
serverless auto-exit in GET /api/positions and manual square-off.

claim_position() atomically moves OPEN → CLOSING so only one closer can win;
book_close() records the exit net of both legs' brokerage.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.database import ActivePosition, TradeRecord


def claim_position(db: Session, pos_id: int) -> bool:
    """Atomically move OPEN → CLOSING. Only one closer (monitor, auto-exit on GET,
    manual square-off) can win, so a position is never booked twice."""
    n = db.query(ActivePosition).filter(ActivePosition.id == pos_id, ActivePosition.status == "OPEN") \
          .update({ActivePosition.status: "CLOSING"}, synchronize_session=False)
    db.commit()
    return n == 1


def release_position(db: Session, pos_id: int):
    db.query(ActivePosition).filter(ActivePosition.id == pos_id, ActivePosition.status == "CLOSING") \
      .update({ActivePosition.status: "OPEN"}, synchronize_session=False)
    db.commit()


def book_close(db: Session, pos, exit_price: float, label: str, status: str, order: Optional[dict] = None):
    """Finish closing a claimed position: P&L net of both legs' brokerage."""
    from app.services.broker import PaperTradingEngine
    from app.services.trading_utils import px
    brok = PaperTradingEngine.BROKERAGE
    gross = (exit_price - pos.entry_price) * pos.quantity if pos.direction == "BUY" \
        else (pos.entry_price - exit_price) * pos.quantity
    fees = (pos.entry_price + exit_price) * pos.quantity * brok
    pnl = round(gross - fees, 2)
    pos.status = "CLOSED"
    pos.exit_price = px(exit_price)
    pos.exit_time = datetime.utcnow()
    exit_action = "SELL" if pos.direction == "BUY" else "BUY"
    rec = TradeRecord(
        user_id=pos.user_id,
        trade_id=(order or {}).get("order_id") or f"EXIT-{pos.id}-{uuid.uuid4().hex[:8].upper()}",
        asset=pos.asset, action=f"{exit_action} ({label})", price=px(exit_price),
        quantity=pos.quantity, pnl=pnl, status=status,
        outcome_return_pct=round(gross / (pos.entry_price * pos.quantity) * 100, 3) if pos.entry_price and pos.quantity else None,
    )
    db.add(rec)
    db.commit()
    return pnl, rec


