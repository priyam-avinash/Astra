from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database import get_db, TradeRecord, TargetSignal, User, ActivePosition, AppSettings, AgentMemory
from app.services.ai_predictor import ai_engine
from app.services.broker import broker_service, PAPER_MODE
from app.services.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from jose import JWTError, jwt
import os
import logging
import numpy as np
from datetime import datetime, date

# ── Financials cache (5-min TTL) — prevents repeated yf.Ticker.info calls ────
_FINANCIALS_CACHE: dict = {}   # key: symbol → (data_dict, datetime)
_FINANCIALS_CACHE_TTL = 300    # 5 minutes

# ── Numpy serialization helper ────────────────────────────────────────────────
def _sanitize(obj):
    """Recursively convert numpy scalars/arrays to Python native types so
    FastAPI's jsonable_encoder never chokes on numpy.bool_ / numpy.int64 etc."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Global trading state ─────────────────────────────────────────────────────
_TRADING_HALTED = False
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS",  "5"))
MAX_DAILY_LOSS  = float(os.getenv("MAX_DAILY_LOSS", "10000"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # --- AUTH BYPASS (intentional for paper-trading phase) ---
    admin_user = db.query(User).filter(User.username == "admin_bypass").first()
    if not admin_user:
        admin_user = User(username="admin_bypass", hashed_password="bypass_hash")
        db.add(admin_user)
        db.commit()
    return admin_user

# ── Kill Switch endpoints (defined AFTER get_current_user) ───────────────────
@router.get("/api/emergency/status")
def trading_status(current_user: User = Depends(get_current_user)):
    return {
        "halted":         _TRADING_HALTED,
        "mode":           "PAPER" if PAPER_MODE else "LIVE",
        "paper_trading":  PAPER_MODE,
        "max_positions":  MAX_POSITIONS,
        "max_daily_loss": MAX_DAILY_LOSS,
    }

@router.post("/api/emergency/halt")
def halt_trading(current_user: User = Depends(get_current_user)):
    global _TRADING_HALTED
    _TRADING_HALTED = True
    logger.warning(f"⛔ TRADING HALTED by {current_user.username}")
    return {"halted": True, "message": "All trading halted. No new orders will be placed."}

@router.post("/api/emergency/resume")
def resume_trading(current_user: User = Depends(get_current_user)):
    global _TRADING_HALTED
    _TRADING_HALTED = False
    logger.info(f"▶ Trading resumed by {current_user.username}")
    return {"halted": False, "message": "Trading resumed."}

class UserCreate(BaseModel):
    username: str
    password: str

@router.post("/api/auth/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@router.post("/api/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

class TradeAction(BaseModel):
    id: int # Queue signal ID
    asset: str
    action: str
    quantity: int
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

@router.get("/api/signals")
def get_ai_signals(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    signals = db.query(TargetSignal).filter(
        TargetSignal.status == "Pending Approval",
        TargetSignal.user_id == current_user.id
    ).all()
    if not signals:
        # Seed mock AI signals — entry_price and target_price are DIFFERENT values
        # BUY:  target > entry  (profit when price rises)
        # SELL: target < entry  (profit when price falls)
        mock_seeds = [
            {"asset": "RELIANCE.NS", "type": "Equity",    "signal": "BUY",  "entry": 1480.0, "target": 1535.0, "stop_loss": 1410.0, "confidence": 74.2, "engine": "astra_ai"},
            {"asset": "TCS.NS",      "type": "Equity",    "signal": "BUY",  "entry": 4050.0, "target": 4210.0, "stop_loss": 3870.0, "confidence": 68.5, "engine": "astra_ai"},
            {"asset": "HDFCBANK.NS", "type": "Equity",    "signal": "SELL", "entry": 1580.0, "target": 1510.0, "stop_loss": 1645.0, "confidence": 61.0, "engine": "astra_ml"},
            {"asset": "GC=F",        "type": "Commodity", "signal": "BUY",  "entry": 2720.0, "target": 2835.0, "stop_loss": 2640.0, "confidence": 71.3, "engine": "astra_ai"},
            {"asset": "^NSEI",       "type": "F&O",       "signal": "HOLD", "entry": 23500.0,"target": 24400.0,"stop_loss": 22800.0,"confidence": 55.0, "engine": "astra"},
        ]
        for s in mock_seeds:
            new_sig = TargetSignal(
                user_id=current_user.id,
                asset=s["asset"], type=s["type"], signal=s["signal"],
                entry_price=s["entry"], target_price=s["target"],
                stop_loss=s["stop_loss"], confidence=s["confidence"],
                engine=s["engine"],
            )
            db.add(new_sig)
        db.commit()
        signals = db.query(TargetSignal).filter(
            TargetSignal.status == "Pending Approval",
            TargetSignal.user_id == current_user.id
        ).all()

    return {"queue": [
        {
            "id":          s.id,
            "asset":       s.asset,
            "type":        s.type,
            "signal":      s.signal,
            "entryPrice":  s.entry_price,   # ← now returned correctly
            "targetPrice": s.target_price,  # ← different from entryPrice
            "stopLoss":    s.stop_loss,
            "confidence":  s.confidence,
            "engine":      s.engine or "astra",
            "age":         "Just now"
        } for s in signals
    ]}

@router.get("/api/analyze/{symbol}")
def analyze_symbol(symbol: str, interval: str = "1d", period: str = "6mo", engine: str = "astra", current_user: User = Depends(get_current_user)):
    """
    Runs market analysis synchronously and returns a flat JSON result.
    The response contains chartData, signal, price info directly at the top level.
    """
    from app.services.ai_predictor import ai_engine
    
    try:
        result = ai_engine.analyze_market_data(symbol.upper(), interval=interval, period=period, engine=engine)
        return _sanitize(result)
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/financials/{symbol}")
def get_financials(symbol: str, current_user: User = Depends(get_current_user)):
    """
    Returns fundamental financial data for a symbol using yfinance Ticker.info.
    Powers the Financials, Forecasts, and Overview tabs.
    """
    import yfinance as yf
    import math
    
    def safe(v):
        """Sanitize NaN/Inf values for JSON serialization."""
        if v is None: return None
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        except: pass
        return v
    
    try:
        sym = symbol.upper()
        if "." not in sym and "^" not in sym and "=" not in sym:
            sym = sym + ".NS"

        # ── Cache check (5-min TTL) ───────────────────────────────────────────
        _cache_entry = _FINANCIALS_CACHE.get(sym)
        if _cache_entry:
            _cached_data, _cached_ts = _cache_entry
            if (datetime.now() - _cached_ts).total_seconds() < _FINANCIALS_CACHE_TTL:
                logger.debug(f"Financials cache hit for {sym}")
                return _cached_data

        ticker = yf.Ticker(sym)
        info = ticker.info or {}
        
        # Key facts
        key_facts = {
            "marketCap": safe(info.get("marketCap")),
            "dividendYield": safe(info.get("dividendYield")),
            "trailingPE": safe(info.get("trailingPE")),
            "forwardPE": safe(info.get("forwardPE")),
            "trailingEps": safe(info.get("trailingEps")),
            "forwardEps": safe(info.get("forwardEps")),
            "priceToBook": safe(info.get("priceToBook")),
            "priceToSales": safe(info.get("priceToSalesTrailing12Months")),
            "beta": safe(info.get("beta")),
            "fiftyTwoWeekHigh": safe(info.get("fiftyTwoWeekHigh")),
            "fiftyTwoWeekLow": safe(info.get("fiftyTwoWeekLow")),
            "averageVolume": safe(info.get("averageVolume")),
            "sharesOutstanding": safe(info.get("sharesOutstanding")),
            "floatShares": safe(info.get("floatShares")),
            "heldPercentInsiders": safe(info.get("heldPercentInsiders")),
            "heldPercentInstitutions": safe(info.get("heldPercentInstitutions")),
        }
        
        # Company profile
        profile = {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary"),
            "country": info.get("country"),
            "city": info.get("city"),
            "employees": safe(info.get("fullTimeEmployees")),
            "currency": info.get("currency", "INR"),
        }
        
        # Dividends
        dividends = {
            "dividendRate": safe(info.get("dividendRate")),
            "dividendYield": safe(info.get("dividendYield")),
            "payoutRatio": safe(info.get("payoutRatio")),
            "exDividendDate": info.get("exDividendDate"),
            "lastDividendValue": safe(info.get("lastDividendValue")),
        }
        
        # Growth & profitability
        growth = {
            "revenueGrowth": safe(info.get("revenueGrowth")),
            "earningsGrowth": safe(info.get("earningsGrowth")),
            "grossMargins": safe(info.get("grossMargins")),
            "operatingMargins": safe(info.get("operatingMargins")),
            "profitMargins": safe(info.get("profitMargins")),
            "returnOnEquity": safe(info.get("returnOnEquity")),
            "returnOnAssets": safe(info.get("returnOnAssets")),
            "totalRevenue": safe(info.get("totalRevenue")),
            "netIncome": safe(info.get("netIncomeToCommon")),
            "totalDebt": safe(info.get("totalDebt")),
            "totalCash": safe(info.get("totalCash")),
            "freeCashflow": safe(info.get("freeCashflow")),
            "operatingCashflow": safe(info.get("operatingCashflow")),
            "debtToEquity": safe(info.get("debtToEquity")),
            "currentRatio": safe(info.get("currentRatio")),
        }
        
        # Analyst targets
        forecasts = {
            "targetHighPrice": safe(info.get("targetHighPrice")),
            "targetLowPrice": safe(info.get("targetLowPrice")),
            "targetMeanPrice": safe(info.get("targetMeanPrice")),
            "targetMedianPrice": safe(info.get("targetMedianPrice")),
            "recommendationKey": info.get("recommendationKey"),
            "recommendationMean": safe(info.get("recommendationMean")),
            "numberOfAnalystOpinions": safe(info.get("numberOfAnalystOpinions")),
        }
        
        result = {
            "symbol": symbol.upper(),
            "profile": profile,
            "keyFacts": key_facts,
            "dividends": dividends,
            "growth": growth,
            "forecasts": forecasts,
        }
        _FINANCIALS_CACHE[sym] = (result, datetime.now())
        return result
    except Exception as e:
        logger.error(f"Financials fetch failed for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "profile": {"name": symbol.upper()},
            "keyFacts": {}, "dividends": {}, "growth": {}, "forecasts": {},
            "error": str(e)
        }

@router.get("/api/analyze/status/{task_id}")
def get_analysis_status(task_id: str):
    """
    Polls the status of a specific analysis task.
    """
    from app.services.tasks import celery_app
    from celery.result import AsyncResult
    
    # If the frontend gets "completed" immediately from the /analyze/ endpoint,
    # it might not even call this, but we handle it just in case.
    try:
        res = AsyncResult(task_id, app=celery_app)
        if res.ready():
            result = res.result
            if isinstance(result, dict) and "error" in result:
                 return {"status": "failed", "error": result["error"]}
            return {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        return {"status": "failed", "error": "Task engine unavailable"}
    
    return {"status": "processing"}

@router.get("/api/analyze/ensemble/{symbol}")
def analyze_ensemble(symbol: str, current_user: User = Depends(get_current_user)):
    """
    Ensemble signal: runs all 3 equity engines (ASTRA 1.0, RF, LSTM) and returns
    consensus signal + Kelly-sized position recommendation.
    Only signals where ≥2/3 engines agree are actionable (CONSENSUS).
    """
    from app.services.ai_predictor import ai_engine
    from collections import Counter

    sym = symbol.upper()
    engines = ["astra", "astra_ai", "astra_ml"]
    results = {}
    signals = []

    for eng in engines:
        try:
            r = ai_engine.analyze_market_data(sym, engine=eng)
            results[eng] = r
            signals.append(r.get("signal", "HOLD"))
        except Exception as e:
            logger.warning(f"Ensemble: {eng} failed for {sym}: {e}")
            signals.append("HOLD")
            results[eng] = {"signal": "HOLD", "confidence": 0.0}

    # Tally votes
    vote_count = Counter(signals)
    top_signal, top_votes = vote_count.most_common(1)[0]
    consensus = top_votes >= 2

    # Average confidence across engines that agree
    agreeing = [results[e]["confidence"] for e, s in zip(engines, signals) if s == top_signal]
    avg_confidence = round(sum(agreeing) / len(agreeing), 1) if agreeing else 0.0

    # Use first agreeing engine's SL/TP
    ref = next((results[e] for e, s in zip(engines, signals) if s == top_signal), results["astra"])
    entry_price = ref.get("entry_price", 0.0)
    target = ref.get("target", 0.0)
    stop_loss = ref.get("stop_loss", 0.0)

    # Kelly position sizing (half-Kelly, capped at 15% of capital)
    kelly_pct = 0.0
    kelly_note = ""
    if consensus and top_signal != "HOLD" and entry_price > 0 and stop_loss > 0:
        avg_win_pct = abs(target - entry_price) / entry_price * 100 if target else 4.0
        avg_loss_pct = abs(entry_price - stop_loss) / entry_price * 100 if stop_loss else 2.0
        win_rate = avg_confidence / 100.0
        b = avg_win_pct / max(avg_loss_pct, 0.1)
        raw_kelly = win_rate - (1 - win_rate) / b
        half_kelly = max(0.0, raw_kelly * 0.5)  # Half-Kelly for safety
        kelly_pct = round(min(half_kelly, 0.15) * 100, 1)  # Cap at 15%
        kelly_note = f"Deploy {kelly_pct}% of capital (half-Kelly, capped 15%)"

    # Minimum confidence gate — don't signal unless confident
    MIN_CONSENSUS_CONFIDENCE = 60.0
    if avg_confidence < MIN_CONSENSUS_CONFIDENCE and top_signal != "HOLD":
        top_signal = "HOLD"
        kelly_note = f"Signal suppressed: avg confidence {avg_confidence}% < {MIN_CONSENSUS_CONFIDENCE}% threshold"

    return {
        "asset": sym,
        "consensus_signal": top_signal if consensus else "HOLD",
        "is_consensus": consensus,
        "votes": vote_count,
        "avg_confidence": avg_confidence,
        "kelly_pct": kelly_pct,
        "kelly_note": kelly_note,
        "entry_price": entry_price,
        "target": target,
        "stop_loss": stop_loss,
        "weekly_trend": ref.get("weekly_trend", "NEUTRAL"),
        "macro_trend": ref.get("macro_trend", "BULL"),
        "engine_breakdown": {
            e: {"signal": results[e].get("signal"), "confidence": results[e].get("confidence")}
            for e in engines
        },
    }

@router.get("/api/analyze/bulk")
def analyze_bulk(symbols: str, interval: str = "1d", period: str = "1mo", engine: str = "astra", current_user: User = Depends(get_current_user)):
    """
    Fast-path analysis for multiple symbols.
    Returns basic signal data without full OHLC to keep it fast.
    """
    from app.services.ai_predictor import ai_engine
    symbol_list = symbols.split(",")
    results = []
    
    for s in symbol_list:
        try:
            # We use a shorter period for bulk to keep it snappy
            res = ai_engine.analyze_market_data(s.strip().upper(), interval=interval, period="1mo", engine=engine)
            results.append({
                "symbol": s.strip().upper(),
                "name": res.get("company_name", s),
                "signal": res.get("signal", "HOLD"),
                "confidence": res.get("confidence", 50.0),
                "price": res.get("current_price", 0.0),
                "entry": res.get("entry_price", 0.0),
                "target": res.get("target", 0.0),
                "stop": res.get("stop_loss", 0.0),
                "currency": res.get("currency", "$")
            })
        except Exception as e:
            logger.error(f"Bulk analysis failed for {s}: {e}")
            results.append({"symbol": s, "error": str(e)})
            
    return {"results": results}

class SignalPayload(BaseModel):
    asset: str
    signal: str
    confidence: float
    entry_price: Optional[float] = None   # from scanner / AI engine
    target_price: Optional[float] = None  # snake_case (scanner sends this)
    stop_loss: Optional[float] = None
    # camelCase aliases (legacy / manual push)
    type: Optional[str] = "Equity"
    targetPrice: Optional[float] = None
    stopLoss: Optional[float] = None
    engine: Optional[str] = "astra"
    signal_type: Optional[str] = None    # ignored, kept for compat

@router.post("/api/signals")
def push_ai_signal(signal: SignalPayload, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Normalise camelCase vs snake_case fields from different callers
    target = signal.target_price or signal.targetPrice or 0.0
    sl     = signal.stop_loss    or signal.stopLoss    or 0.0
    new_sig = TargetSignal(
        user_id=current_user.id,
        asset=signal.asset,
        type=signal.type or "Equity",
        signal=signal.signal,
        entry_price=signal.entry_price or 0.0,
        target_price=target,
        stop_loss=sl,
        confidence=signal.confidence,
        engine=signal.engine or "astra",
    )
    db.add(new_sig)
    db.commit()
    db.refresh(new_sig)
    return {"status": "success", "id": new_sig.id}

@router.post("/api/execute")
def execute_trade(trade: TradeAction, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # ── Server-side risk guards ───────────────────────────────────────────────
    if _TRADING_HALTED:
        raise HTTPException(status_code=503, detail="⛔ Trading is halted. Use /api/emergency/resume to re-enable.")

    open_count = db.query(ActivePosition).filter(
        ActivePosition.user_id == current_user.id,
        ActivePosition.status == "OPEN"
    ).count()
    if open_count >= MAX_POSITIONS:
        raise HTTPException(status_code=429, detail=f"Max {MAX_POSITIONS} concurrent positions reached. Close a position first.")

    today_loss = db.query(func.sum(TradeRecord.pnl)).filter(
        TradeRecord.user_id == current_user.id,
        func.date(TradeRecord.timestamp) == date.today(),
        TradeRecord.pnl < 0,
    ).scalar() or 0.0
    if today_loss < -MAX_DAILY_LOSS:
        raise HTTPException(status_code=429, detail=f"Daily loss limit (₹{MAX_DAILY_LOSS:,.0f}) reached. Trading paused for today.")

    if trade.quantity < 1:
        raise HTTPException(status_code=400, detail="Quantity must be at least 1.")
    # ─────────────────────────────────────────────────────────────────────────

    signal = db.query(TargetSignal).filter(TargetSignal.id == trade.id, TargetSignal.user_id == current_user.id).first()
    if signal:
        signal.status = "Executed"

    try:
        result = broker_service.execute_trade(
            asset=trade.asset,
            action=trade.action,
            quantity=trade.quantity,
            price=trade.price
        )
        
        # 1. Create permanent trade record
        new_trade = TradeRecord(
            user_id=current_user.id,
            trade_id=result["order_id"],
            asset=trade.asset,
            action=trade.action,
            price=trade.price,
            quantity=trade.quantity,
            pnl=0.0,
            status="Executed (AI Signal)"
        )
        db.add(new_trade)

        # 2. Create interactive active position for P&L tracking
        new_pos = ActivePosition(
            user_id=current_user.id,
            asset=trade.asset,
            direction=trade.action,
            entry_price=trade.price,
            quantity=trade.quantity,
            target_price=trade.target_price,
            stop_loss=trade.stop_loss,
            status="OPEN"
        )
        db.add(new_pos)
        
        db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

class ManualTradeAction(BaseModel):
    asset: str
    action: str
    quantity: int
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

@router.post("/api/execute/manual")
def execute_manual_trade(trade: ManualTradeAction, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        result = broker_service.execute_trade(
            asset=trade.asset,
            action=trade.action,
            quantity=trade.quantity,
            price=trade.price
        )
        
        new_trade = TradeRecord(
            user_id=current_user.id,
            trade_id=result["order_id"],
            asset=trade.asset,
            action=trade.action,
            price=trade.price,
            quantity=trade.quantity,
            pnl=0.0,
            status="Executed (Manual)"
        )
        db.add(new_trade)

        new_pos = ActivePosition(
            user_id=current_user.id,
            asset=trade.asset,
            direction=trade.action,
            entry_price=trade.price,
            quantity=trade.quantity,
            target_price=trade.target_price,
            stop_loss=trade.stop_loss,
            status="OPEN"
        )
        db.add(new_pos)

        db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/positions")
def get_active_positions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    positions = db.query(ActivePosition).filter(
        ActivePosition.user_id == current_user.id,
        ActivePosition.status == "OPEN"
    ).all()
    
    resp = []
    for p in positions:
        # Get live price from the engine
        current_price = ai_engine.get_realtime_price(p.asset)
        if current_price == 0.0:
            current_price = p.entry_price

        # AUTO-EXIT CHECK
        triggered = False
        exit_reason = ""
        if p.direction == "BUY":
            if p.target_price and current_price >= p.target_price:
                triggered = True; exit_reason = "Target Hit"
            elif p.stop_loss and current_price <= p.stop_loss:
                triggered = True; exit_reason = "Stop Loss Hit"
        else: # SELL
            if p.target_price and current_price <= p.target_price:
                triggered = True; exit_reason = "Target Hit"
            elif p.stop_loss and current_price >= p.stop_loss:
                triggered = True; exit_reason = "Stop Loss Hit"

        if triggered:
            # Auto-square off
            p.status = "CLOSED"
            p.exit_price = current_price
            p.exit_time = datetime.utcnow()
            
            # Record exit in history
            exit_action = "SELL" if p.direction == "BUY" else "BUY"
            pnl = (current_price - p.entry_price) * p.quantity if p.direction == "BUY" else (p.entry_price - current_price) * p.quantity
            
            new_history = TradeRecord(
                user_id=current_user.id,
                trade_id=f"AUTO-{p.id}",
                asset=p.asset,
                action=f"{exit_action} ({exit_reason})",
                price=current_price,
                quantity=p.quantity,
                pnl=round(pnl, 2),
                status="Closed (Auto)"
            )
            db.add(new_history)
            db.commit()
            continue # Don't add to active resp

        unrealized_pnl = (current_price - p.entry_price) * p.quantity if p.direction == "BUY" else (p.entry_price - current_price) * p.quantity
        
        resp.append({
            "id": p.id,
            "asset": p.asset,
            "direction": p.direction,
            "entry_price": p.entry_price,
            "quantity": p.quantity,
            "target_price": p.target_price,
            "stop_loss": p.stop_loss,
            "current_price": round(current_price, 2),
            "pnl": round(unrealized_pnl, 2),
            "pnl_pct": round((unrealized_pnl / (p.entry_price * p.quantity)) * 100, 2) if p.entry_price and p.quantity else 0,
            "time": p.entry_time.strftime("%H:%M:%S"),
            "status": p.status
        })
    return {"positions": resp}

@router.post("/api/positions/squareoff/{position_id}")
def square_off_position(position_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    pos = db.query(ActivePosition).filter(ActivePosition.id == position_id, ActivePosition.user_id == current_user.id).first()
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")
    
    # Execute opposite trade
    exit_action = "SELL" if pos.direction == "BUY" else "BUY"
    
    try:
        # Use real LTP from the paper trading engine (not random)
        exit_price = broker_service.get_exit_price(pos.asset, pos.direction)
        if not exit_price or exit_price <= 0:
            # Fallback: use last known current_price from AI engine
            exit_price = ai_engine.get_realtime_price(pos.asset) or pos.entry_price

        result = broker_service.execute_trade(
            asset=pos.asset,
            action=exit_action,
            quantity=pos.quantity,
            price=exit_price
        )
        exit_price = result.get("executed_price", exit_price)

        pnl = (exit_price - pos.entry_price) * pos.quantity if pos.direction == "BUY" else (pos.entry_price - exit_price) * pos.quantity
        
        # Update position
        pos.status = "CLOSED"
        pos.exit_price = exit_price
        pos.exit_time = datetime.utcnow()
        
        # Add to history
        new_history = TradeRecord(
            user_id=current_user.id,
            trade_id=result["order_id"],
            asset=pos.asset,
            action=exit_action + " (Square Off)",
            price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            status="Closed"
        )
        db.add(new_history)
        db.commit()
        
        return {"status": "success", "pnl": pnl, "exit_price": exit_price}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/history")
def get_trade_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    history = db.query(TradeRecord).filter(TradeRecord.user_id == current_user.id).order_by(TradeRecord.timestamp.desc()).all()
    return {"history": [
        {
            "id": h.trade_id,
            "asset": h.asset,
            "action": h.action,
            "price": h.price,
            "qty": h.quantity,
            "pnl": h.pnl,
            "status": h.status,
            "time": h.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(h.timestamp, "strftime") else "Unknown"
        } for h in history
    ]}


# ══════════════════════════════════════════════════════════════
#  CRYPTO ENDPOINTS — ASTRA.CRYPTO ENGINE
# ══════════════════════════════════════════════════════════════

@router.get("/api/crypto/watchlist")
def get_crypto_watchlist(
    market: str = "international",
    current_user: User = Depends(get_current_user)
):
    """
    Returns live prices for the crypto watchlist ticker strip.
    market: 'international' (USD) | 'indian' (INR)
    """
    from app.services.crypto_engine import crypto_engine
    try:
        prices = crypto_engine.get_watchlist_prices(market=market)
        return {"market": market, "assets": prices}
    except Exception as e:
        logger.error(f"Crypto watchlist failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/crypto/fear-greed")
def get_fear_greed(current_user: User = Depends(get_current_user)):
    """Returns the current Fear & Greed Index from Alternative.me."""
    from app.services.crypto_engine import crypto_engine
    try:
        fg = crypto_engine.get_fear_greed()
        dom = crypto_engine.get_btc_dominance()
        return {"fear_greed": fg, "btc_dominance": dom}
    except Exception as e:
        logger.error(f"Fear & Greed fetch failed: {e}")
        return {"fear_greed": {"value": 50, "label": "Neutral", "updated": ""}, "btc_dominance": 50.0}


@router.get("/api/scan/universe")
def scan_universe(
    engine: str = "astra_ai",
    top_k: int = 20,
    min_confidence: float = 55.0,
    signal_filter: Optional[str] = None,        # "BUY" | "SELL" | None (all)
    sector: Optional[str] = None,               # filter by sector
    current_user: User = Depends(get_current_user),
):
    """
    Scan the full NSE universe (NIFTY 500 ~141 symbols) for live signals.

    Returns top_k signals ranked by confidence.
    - engine: 'astra_ai' (fast, default) | 'astra' (rules) | 'astra_ml' (slow)
    - min_confidence: minimum confidence % to include (default 55)
    - signal_filter: 'BUY' or 'SELL' to show only one direction
    - sector: e.g. 'IT', 'Banking', 'Pharma' — scans only that sector
    """
    from app.services.universe_scanner import universe_scanner
    try:
        if sector:
            results = universe_scanner.scan_sector(
                sector=sector, engines=[engine],
                top_k=top_k, min_confidence=min_confidence
            )
        else:
            results = universe_scanner.scan(
                engines=[engine], top_k=top_k, min_confidence=min_confidence
            )
        if signal_filter:
            results = [r for r in results if r.get("signal") == signal_filter.upper()]
        return {
            "status": "ok",
            "engine": engine,
            "total_signals": len(results),
            "universe_size": len(universe_scanner.symbols),
            "signals": results,
        }
    except Exception as e:
        logger.error(f"Universe scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/scan/full")
def scan_full_consensus(
    top_k: int = 15,
    current_user: User = Depends(get_current_user),
):
    """
    Two-stage consensus scan:
    Stage 1: ASTRA.AI on all ~141 symbols → top 30 candidates
    Stage 2: ASTRA.ML confirmation on those 30
    Returns signals where both engines agree.
    Slower (~60–90s) but highest conviction.
    """
    from app.services.universe_scanner import universe_scanner
    try:
        results = universe_scanner.full_scan(top_k=top_k)
        return {"status": "ok", "signals": results, "pipeline": "astra_ai→astra_ml"}
    except Exception as e:
        logger.error(f"Full consensus scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/scan/sectors")
def get_available_sectors(current_user: User = Depends(get_current_user)):
    """List all available sectors in the universe."""
    from app.services.universe_scanner import universe_scanner, _SECTOR_MAP
    sectors = sorted(set(_SECTOR_MAP.values()))
    sector_counts = {}
    for s in sectors:
        sector_counts[s] = sum(1 for sym in universe_scanner.symbols
                               if universe_scanner.get_sector(sym) == s)
    sector_counts["Other"] = sum(1 for sym in universe_scanner.symbols
                                 if universe_scanner.get_sector(sym) == "Other")
    return {"sectors": sectors, "counts": sector_counts,
            "universe_size": len(universe_scanner.symbols)}


@router.get("/api/intraday/{symbol}")
def analyze_intraday(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """
    Intraday signal for a single NSE stock.
    Uses 15-minute OHLCV data.
    Strategies: Opening Range Breakout (ORB) + VWAP Mean Reversion.

    Returns: signal, strategy, entry_price, sl, tp, confidence, time_remaining_min
    Only generates signals during market hours (09:15 – 14:45 IST).
    """
    from app.services.intraday_engine import intraday_engine
    sym = symbol.upper()
    if not sym.endswith(".NS"):
        sym = sym + ".NS"
    try:
        result = intraday_engine.analyze_intraday(sym)
        return result
    except Exception as e:
        logger.error(f"Intraday analysis failed for {sym}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/intraday/scan/top")
def intraday_universe_scan(
    top_k: int = 10,
    current_user: User = Depends(get_current_user),
):
    """
    Scans NIFTY 50 for intraday signals.
    Returns top_k signals ranked by confidence.
    Best called between 09:45 – 14:30 IST.
    """
    from app.services.intraday_engine import intraday_engine
    from app.services.universe_scanner import _NIFTY_50
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = [s + ".NS" for s in _NIFTY_50]
    results = []

    def _scan_one(sym):
        try:
            r = intraday_engine.analyze_intraday(sym)
            if r.get("signal") not in ("BUY", "SELL"):
                return None
            return r
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_scan_one, s): s for s in symbols}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return {
        "status": "ok",
        "signals": results[:top_k],
        "scanned": len(symbols),
        "active_signals": len(results),
    }


@router.get("/api/commodities/list")
def list_commodities(current_user: User = Depends(get_current_user)):
    """List all supported commodity symbols with metadata."""
    from app.services.commodities_engine import COMMODITIES
    return {"commodities": [
        {"symbol": sym, **meta} for sym, meta in COMMODITIES.items()
    ]}


@router.get("/api/commodities/analyze/{symbol}")
def analyze_commodity(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """
    Full commodity analysis for a single futures contract.
    Symbols: GC=F (Gold), SI=F (Silver), CL=F (Crude), NG=F (NatGas),
             HG=F (Copper), ALI=F (Aluminium), ZC=F (Corn), ZW=F (Wheat)

    Returns: signal, confidence, entry_price, stop_loss, target,
             seasonal_bias, macro context (VIX, DXY), chart_data
    """
    from app.services.commodities_engine import commodities_engine
    try:
        return commodities_engine.analyze(symbol.upper())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Commodity analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/commodities/scan")
def scan_commodities(
    signal_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Scan all commodity futures for signals.
    signal_filter: 'BUY' | 'SELL' | None (return all, sorted by confidence)
    """
    from app.services.commodities_engine import commodities_engine
    results = commodities_engine.scan_all(signal_filter=signal_filter)
    return {"status": "ok", "signals": results, "total": len(results)}


@router.get("/api/learning/buffer/summary")
def replay_buffer_summary(current_user: User = Depends(get_current_user)):
    """
    Replay buffer health check: win rate of recorded experiences,
    per-engine stats, and drift detection result.
    """
    from app.services.replay_buffer import replay_buffer
    return replay_buffer.summary()


@router.get("/api/learning/buffer/drift")
def check_drift(
    window: int = 20,
    current_user: User = Depends(get_current_user),
):
    """
    Drift detection: compares recent {window} trades vs overall win rate.
    If recent WR drops >15pp, flags drift and recommends retraining.
    """
    from app.services.replay_buffer import replay_buffer
    return replay_buffer.drift_check(window=window)


@router.get("/api/crypto/analyze/{symbol}")
def analyze_crypto(
    symbol: str,
    market: str = "international",
    timeframe: str = "1d",
    engine: str = "astra_crypto",
    current_user: User = Depends(get_current_user)
):
    """
    Full ASTRA.CRYPTO analysis for a single symbol.
    symbol:    e.g. BTC-USD, ETH-USD, BTC-INR, ETH-INR
    market:    'international' | 'indian'
    timeframe: '1d' | '4h' | '1h'
    engine:    'astra_crypto' (rule-based) | 'astra_crypto_ml' (LSTM)
    """
    from app.services.crypto_engine import crypto_engine
    try:
        result = crypto_engine.analyze(
            symbol=symbol.upper(),
            market=market,
            timeframe=timeframe,
            engine=engine
        )
        return result
    except Exception as e:
        logger.error(f"Crypto analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
#  SETTINGS ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

# Settings keys that contain API keys (masked in GET responses)
_SECRET_KEYS = {"anthropic_api_key", "groq_api_key"}

# All valid setting keys with their defaults
_SETTING_DEFAULTS = {
    "llm_enabled":              "false",
    "anthropic_api_key":        "",
    "groq_api_key":             "",
    "auto_execute_mode":        "advisory",   # advisory | auto
    "auto_execute_threshold":   "GREEN",      # GREEN | AMBER_GREEN
    "always_debate":            "false",
    "max_debate_rounds":        "1",
    "max_risk_rounds":          "1",
}


def _mask(key: str, value: str) -> str:
    """Mask API keys — show only last 4 chars."""
    if not value:
        return ""
    if key in _SECRET_KEYS and len(value) > 4:
        return "•" * (len(value) - 4) + value[-4:]
    return value


@router.get("/api/settings")
def get_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all app settings. API keys are masked."""
    rows = db.query(AppSettings).all()
    stored = {r.key: r.value for r in rows}

    result = {}
    for key, default in _SETTING_DEFAULTS.items():
        raw_value = stored.get(key, default) or default
        result[key] = _mask(key, raw_value)

    # Append provider status from LLM router
    try:
        from app.services.llm_router import LLMRouter
        status = LLMRouter().provider_status()
        result["provider_status"] = status
    except Exception:
        result["provider_status"] = {"active_provider": "none", "llm_available": False}

    return result


class SettingsUpdate(BaseModel):
    settings: dict   # {key: value} pairs to update


@router.put("/api/settings")
def update_settings(
    payload: SettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update one or more settings. API keys are stored as-is (not hashed)."""
    updated = []
    for key, value in payload.settings.items():
        if key not in _SETTING_DEFAULTS:
            continue   # ignore unknown keys
        # Don't overwrite a real key with a masked placeholder
        if key in _SECRET_KEYS and value and all(c == "•" for c in value[:-4]):
            continue
        row = db.query(AppSettings).filter(AppSettings.key == key).first()
        if row:
            row.value = str(value) if value is not None else ""
            row.updated_at = datetime.utcnow()
        else:
            row = AppSettings(key=key, value=str(value) if value is not None else "")
            db.add(row)
        updated.append(key)
    db.commit()

    # Invalidate LLM router singletons so new keys take effect immediately
    try:
        import app.services.llm_router as lr
        lr._fast_router  = None
        lr._smart_router = None
    except Exception:
        pass

    return {"updated": updated, "message": f"{len(updated)} setting(s) saved."}


@router.get("/api/settings/provider-status")
def provider_status(current_user: User = Depends(get_current_user)):
    """Quick check: which LLM provider is active."""
    try:
        from app.services.llm_router import LLMRouter
        return LLMRouter().provider_status()
    except Exception as e:
        return {"active_provider": "none", "llm_available": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  AGENT ENRICHMENT ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

class EnrichRequest(BaseModel):
    llm_override:          Optional[bool] = None   # None = use global setting
    auto_execute_override: Optional[str]  = None   # "advisory" / "auto" / None


@router.post("/api/analyze/{symbol}/enrich")
def enrich_signal(
    symbol: str,
    payload: EnrichRequest,
    engine: str = "astra",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run the full multi-agent pipeline on a symbol.
    First runs the ML engine, then enriches with LLM agents.
    Returns the complete enriched signal (may take 15-30 seconds with LLM).
    """
    import json as _json
    from app.services.agent_pipeline import agent_pipeline

    # Step 1: ML analysis
    try:
        base = ai_engine.analyze_market_data(symbol.upper(), engine=engine)
        base = _sanitize(base)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {e}")

    # Step 2: Agent enrichment
    try:
        enriched = agent_pipeline.run(
            symbol=symbol.upper(),
            base_signal=base,
            db=db,
            llm_override=payload.llm_override,
            auto_execute_override=payload.auto_execute_override,
        )
    except Exception as e:
        logger.error(f"Agent pipeline failed for {symbol}: {e}")
        enriched = base   # degrade gracefully

    return enriched


@router.get("/api/agent/memories/{symbol}")
def get_agent_memories(
    symbol: str,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return ASTRA's learned lessons for a symbol."""
    rows = (
        db.query(AgentMemory)
        .filter(AgentMemory.symbol == symbol.upper())
        .order_by(AgentMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "symbol":  symbol.upper(),
        "lessons": [
            {
                "id":         r.id,
                "content":    r.content,
                "type":       r.memory_type,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/api/agent/memories")
def get_all_agent_memories(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all recent lessons across all symbols."""
    rows = (
        db.query(AgentMemory)
        .order_by(AgentMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "lessons": [
            {
                "id":         r.id,
                "symbol":     r.symbol,
                "content":    r.content,
                "type":       r.memory_type,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }

