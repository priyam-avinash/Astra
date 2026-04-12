from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.database import get_db, TradeRecord, TargetSignal, User, ActivePosition
from app.services.ai_predictor import ai_engine
from app.services.broker import broker_service
from app.services.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from jose import JWTError, jwt
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # --- AUTH BYPASS INJECTED AS REQUESTED ---
    # Automatically creates and returns a mock user to unblock UI testing
    admin_user = db.query(User).filter(User.username == "admin_bypass").first()
    if not admin_user:
        admin_user = User(username="admin_bypass", hashed_password="bypass_hash")
        db.add(admin_user)
        db.commit()
    return admin_user

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
        # Seed mock AI signals from static data (no blocking ML training)
        mock_seeds = [
            {"asset": "RELIANCE.NS", "type": "Equity", "signal": "BUY",  "target": 1480.0, "stop_loss": 1350.0, "confidence": 74.2},
            {"asset": "TCS.NS",      "type": "Equity", "signal": "BUY",  "target": 4050.0, "stop_loss": 3800.0, "confidence": 68.5},
            {"asset": "HDFCBANK.NS", "type": "Equity", "signal": "SELL", "target": 1580.0, "stop_loss": 1720.0, "confidence": 61.0},
            {"asset": "GC=F",        "type": "Commodity", "signal": "BUY","target": 2720.0, "stop_loss": 2580.0, "confidence": 71.3},
            {"asset": "^NSEI",       "type": "F&O",    "signal": "HOLD", "target": 23500.0,"stop_loss": 22000.0,"confidence": 55.0},
        ]
        for s in mock_seeds:
            new_sig = TargetSignal(
                user_id=current_user.id,
                asset=s["asset"], type=s["type"], signal=s["signal"],
                target_price=s["target"], stop_loss=s["stop_loss"], confidence=s["confidence"]
            )
            db.add(new_sig)
        db.commit()
        signals = db.query(TargetSignal).filter(
            TargetSignal.status == "Pending Approval",
            TargetSignal.user_id == current_user.id
        ).all()

    return {"queue": [
        {
            "id": s.id,
            "asset": s.asset,
            "type": s.type,
            "signal": s.signal,
            "targetPrice": s.target_price,
            "stopLoss": s.stop_loss,
            "confidence": s.confidence,
            "age": "Just now"
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
        return result
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
        
        return {
            "symbol": symbol.upper(),
            "profile": profile,
            "keyFacts": key_facts,
            "dividends": dividends,
            "growth": growth,
            "forecasts": forecasts,
        }
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
    type: str
    signal: str
    targetPrice: float
    stopLoss: float
    confidence: float

@router.post("/api/signals")
def push_ai_signal(signal: SignalPayload, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_sig = TargetSignal(
        user_id=current_user.id,
        asset=signal.asset,
        type=signal.type,
        signal=signal.signal,
        target_price=signal.targetPrice,
        stop_loss=signal.stopLoss,
        confidence=signal.confidence
    )
    db.add(new_sig)
    db.commit()
    db.refresh(new_sig)
    return {"status": "success", "id": new_sig.id}

@router.post("/api/execute")
def execute_trade(trade: TradeAction, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
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
        # Mock logic to get "exit price"
        import random
        exit_price = pos.entry_price * (1 + random.uniform(-0.02, 0.05)) 
        
        result = broker_service.execute_trade(
            asset=pos.asset,
            action=exit_action,
            quantity=pos.quantity,
            price=exit_price
        )
        
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

