from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL")
# Default to sqlite for local development if no DATABASE_URL is provided
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./astra_trading.db"
elif not DATABASE_URL.startswith("postgresql"):
    # If a URL is provided but it's not postgres, still check for sqlite pattern
    if "sqlite" not in DATABASE_URL:
         DATABASE_URL = "sqlite:///./astra_trading.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    trades = relationship("TradeRecord", back_populates="owner")
    signals = relationship("TargetSignal", back_populates="owner")
    positions = relationship("ActivePosition", back_populates="owner")

class TradeRecord(Base):
    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trade_id = Column(String, unique=True, index=True)
    asset = Column(String, index=True)
    action = Column(String) 
    price = Column(Float)
    quantity = Column(Integer)
    pnl = Column(Float, default=0.0)
    status = Column(String, default="Executed")
    timestamp = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="trades")

class TargetSignal(Base):
    __tablename__ = "active_signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    asset = Column(String, index=True)
    type = Column(String)
    signal = Column(String)
    target_price = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float)
    status = Column(String, default="Pending Approval")
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="signals")

class ActivePosition(Base):
    __tablename__ = "active_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    asset = Column(String, index=True)
    direction = Column(String) # BUY or SELL
    entry_price = Column(Float)
    quantity = Column(Integer)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    status = Column(String, default="OPEN") # OPEN, CLOSED
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)

    owner = relationship("User", back_populates="positions")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
