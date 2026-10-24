"""Mutual fund endpoints (paper investing only)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.endpoints import get_current_user
from app.models.database import MFSip, User, get_db
from app.services.mutual_funds import (MFError, MAX_ACTIVE_SIPS, book_purchase, book_redemption, canon_code,
                                       mf_service, portfolio)

router = APIRouter(prefix="/api/mf", tags=["mutual-funds"])


def _err(e: Exception):
    raise HTTPException(status_code=404 if "not found" in str(e).lower() else 503 if "unavailable" in str(e).lower() else 400,
                        detail=str(e))


@router.get("/search")
def search(q: str, current_user: User = Depends(get_current_user)):
    try:
        return {"results": mf_service.search(q)}
    except MFError as e:
        _err(e)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Mutual fund search unavailable: {type(e).__name__}")


@router.get("/portfolio")
def mf_portfolio(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return portfolio(db, current_user.id)


@router.get("/scheme/{code}")
def scheme(code: str, current_user: User = Depends(get_current_user)):
    try:
        return mf_service.scheme(code)
    except MFError as e:
        _err(e)


@router.get("/scheme/{code}/chart")
def chart(code: str, period: str = "5y", current_user: User = Depends(get_current_user)):
    try:
        return {"points": mf_service.chart(code, period)}
    except MFError as e:
        _err(e)


@router.get("/scheme/{code}/sip-calc")
def sip_calc(code: str, amount: float = 10000, years: int = 5, current_user: User = Depends(get_current_user)):
    if not (100 <= amount <= 1e7) or not (1 <= years <= 30):
        raise HTTPException(status_code=400, detail="amount ₹100–₹1,00,00,000 and 1–30 years")
    try:
        _, s = mf_service.nav_history(code)
        return mf_service.sip_projection(s, amount, years)
    except MFError as e:
        _err(e)


class InvestReq(BaseModel):
    scheme_code: str
    amount: float = Field(..., gt=0)


class RedeemReq(BaseModel):
    scheme_code: str
    units: Optional[float] = Field(None, gt=0)
    amount: Optional[float] = Field(None, gt=0)


class SipReq(BaseModel):
    scheme_code: str
    amount: float = Field(..., ge=100, le=1e7)
    day_of_month: int = Field(5, ge=1, le=28)
    start_date: Optional[str] = None      # YYYY-MM-DD, default today (back-dated SIPs book past instalments)


@router.post("/invest")
def invest(req: InvestReq, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        tx = book_purchase(db, current_user.id, req.scheme_code, req.amount)
        db.commit()
        return {"ok": True, "status": tx.status, "units": tx.units, "nav": tx.nav, "nav_date": tx.nav_date,
                "order_date": tx.order_date, "paper": True,
                "message": ("Booked at NAV " + str(tx.nav)) if tx.status == "SETTLED"
                           else "Order placed — it will be priced at the NAV declared for " + tx.order_date}
    except MFError as e:
        db.rollback()
        _err(e)


@router.post("/redeem")
def redeem(req: RedeemReq, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        tx = book_redemption(db, current_user.id, req.scheme_code, units=req.units, amount=req.amount)
        db.commit()
        return {"ok": True, "status": tx.status, "units": tx.units, "nav": tx.nav, "proceeds": tx.amount,
                "realised_pnl": tx.realised_pnl, "order_date": tx.order_date, "paper": True}
    except MFError as e:
        db.rollback()
        _err(e)


@router.post("/sip")
def create_sip(req: SipReq, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        code = canon_code(req.scheme_code)
        meta, _ = mf_service.nav_history(code)
    except MFError as e:
        _err(e)
    active = db.query(MFSip).filter(MFSip.user_id == current_user.id, MFSip.active == True).count()  # noqa: E712
    if active >= MAX_ACTIVE_SIPS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_ACTIVE_SIPS} active SIPs")
    start = req.start_date or date.today().isoformat()
    try:
        sd = date.fromisoformat(start)
    except ValueError:
        raise HTTPException(status_code=400, detail="start_date must be YYYY-MM-DD")
    if sd > date.today() or (date.today() - sd).days > 366 * 10:
        raise HTTPException(status_code=400, detail="start_date must be within the last 10 years and not in the future")
    sip = MFSip(user_id=current_user.id, scheme_code=code, scheme_name=meta.get("scheme_name"),
                amount=req.amount, day_of_month=req.day_of_month, start_date=start, active=True)
    db.add(sip)
    db.commit()
    return {"ok": True, "id": sip.id, "paper": True}


@router.delete("/sip/{sip_id}")
def stop_sip(sip_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sip = db.query(MFSip).filter(MFSip.id == sip_id, MFSip.user_id == current_user.id).first()
    if not sip:
        raise HTTPException(status_code=404, detail="SIP not found")
    sip.active = False
    db.commit()
    return {"ok": True}
