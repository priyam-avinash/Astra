from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
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
    # ── Continuous learning fields ──────────────────────────────────────
    reflection       = Column(Text,    nullable=True)   # LLM reflection text
    reflection_at    = Column(DateTime, nullable=True)  # when reflection was generated
    outcome_return_pct = Column(Float, nullable=True)   # % return for this trade

    owner = relationship("User", back_populates="trades")

class TargetSignal(Base):
    __tablename__ = "active_signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    asset = Column(String, index=True)
    type = Column(String)
    signal = Column(String)
    entry_price = Column(Float, nullable=True, default=0.0)
    target_price = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float)
    engine = Column(String, nullable=True, default="astra")
    status = Column(String, default="Pending Approval")
    created_at = Column(DateTime, default=datetime.utcnow)
    # ── Multi-agent enrichment fields ───────────────────────────────────
    llm_enriched         = Column(Boolean,  default=False, nullable=True)
    news_sentiment       = Column(String,   nullable=True)   # POSITIVE/NEGATIVE/NEUTRAL/MIXED
    news_summary         = Column(Text,     nullable=True)
    fundamental_score    = Column(Float,    nullable=True)
    fundamental_grade    = Column(String,   nullable=True)   # A/B/C/D/F
    debate_verdict       = Column(String,   nullable=True)   # CONFIRMED/DOWNGRADED/REJECTED
    debate_text          = Column(Text,     nullable=True)   # JSON of full debate
    risk_verdict         = Column(String,   nullable=True)   # GREEN/AMBER/RED
    risk_text            = Column(Text,     nullable=True)   # JSON of portfolio decision
    recommended_position_pct = Column(Float, nullable=True)
    gated_by             = Column(String,   nullable=True)   # news/fundamentals/debate/portfolio_manager
    gate_reason          = Column(Text,     nullable=True)

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

class AppSettings(Base):
    """Key-value store for app configuration (API keys, feature toggles, etc.)"""
    __tablename__ = "app_settings"

    id         = Column(Integer,  primary_key=True, index=True)
    key        = Column(String,   unique=True, nullable=False, index=True)
    value      = Column(Text,     nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentMemory(Base):
    """Persistent lessons extracted by the ReflectorAgent for continuous learning."""
    __tablename__ = "agent_memory"

    id              = Column(Integer,  primary_key=True, index=True)
    symbol          = Column(String,   nullable=False, index=True)
    memory_type     = Column(String,   default="ticker_lesson")  # ticker_lesson / cross_ticker
    content         = Column(Text,     nullable=False)
    source_trade_id = Column(Integer,  ForeignKey("trade_history.id"), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ── SQLite column-migration helper (dev only) ─────────────────────────
# create_all() won't add new columns to existing tables, so we do it manually.
def _run_migrations():
    """Add new columns to existing tables without wiping data."""
    migrations = [
        # Original columns
        ("active_signals",   "entry_price",       "REAL DEFAULT 0.0"),
        ("active_signals",   "engine",             "TEXT DEFAULT 'astra'"),
        ("active_positions", "entry_features_json","TEXT"),
        ("active_positions", "engine",             "TEXT DEFAULT 'unknown'"),
        # v1.8 — multi-agent enrichment on active_signals
        ("active_signals",   "llm_enriched",            "INTEGER DEFAULT 0"),
        ("active_signals",   "news_sentiment",           "TEXT"),
        ("active_signals",   "news_summary",             "TEXT"),
        ("active_signals",   "fundamental_score",        "REAL"),
        ("active_signals",   "fundamental_grade",        "TEXT"),
        ("active_signals",   "debate_verdict",           "TEXT"),
        ("active_signals",   "debate_text",              "TEXT"),
        ("active_signals",   "risk_verdict",             "TEXT"),
        ("active_signals",   "risk_text",                "TEXT"),
        ("active_signals",   "recommended_position_pct", "REAL"),
        ("active_signals",   "gated_by",                 "TEXT"),
        ("active_signals",   "gate_reason",              "TEXT"),
        # v1.8 — continuous learning on trade_history
        ("trade_history",    "reflection",          "TEXT"),
        ("trade_history",    "reflection_at",       "TEXT"),
        ("trade_history",    "outcome_return_pct",  "REAL"),
    ]
    with engine.connect() as conn:
        for table, col, col_type in migrations:
            try:
                conn.execute(__import__('sqlalchemy').text(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                ))
                conn.commit()
            except Exception:
                pass  # Column already exists — safe to ignore

_run_migrations()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
