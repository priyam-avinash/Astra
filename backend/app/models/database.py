from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

from app.core.config import BACKEND_DIR, SERVERLESS, DATA_DIR

# SQLite path is anchored to backend/ (it used to be relative to the current
# working directory, so starting the server from the repo root silently
# created a second, empty database). On serverless hosts the code dir is
# read-only, so the fallback SQLite file lives in /tmp (ephemeral — attach a
# Postgres database for anything you want to keep).
_SQLITE_PATH = (DATA_DIR / "astra_trading.db") if SERVERLESS else (BACKEND_DIR / "astra_trading.db")
_DEFAULT_SQLITE = f"sqlite:///{_SQLITE_PATH}"


def _resolve_database_url() -> str:
    # Vercel's Postgres/Neon integration exposes POSTGRES_URL / DATABASE_URL.
    for key in ("DATABASE_URL", "POSTGRES_URL", "POSTGRES_PRISMA_URL"):
        url = (os.getenv(key) or "").strip().strip('"')
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql") or url.startswith("sqlite"):
            return url
    if SERVERLESS:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_SQLITE


DATABASE_URL = _resolve_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
elif SERVERLESS:
    # Each serverless instance is short-lived; don't hold pooled connections.
    from sqlalchemy.pool import NullPool
    engine = create_engine(DATABASE_URL, poolclass=NullPool, pool_pre_ping=True)
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
    reflection          = Column(Text,    nullable=True)   # LLM reflection text
    reflection_at       = Column(DateTime, nullable=True)  # when reflection was generated
    outcome_return_pct  = Column(Float,   nullable=True)   # % return for this trade
    entry_features_json = Column(Text,    nullable=True)   # JSON of 20-feature vector at entry
    signal_label        = Column(String,  nullable=True)   # "CORRECT" | "INCORRECT" | "NEUTRAL"

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
    # ── Self-learning: feature snapshot at entry time ────────────────────
    entry_features_json = Column(Text, nullable=True)  # JSON dict of 20 indicator values
    engine              = Column(String, nullable=True, default="unknown")  # which engine fired

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


class BrokerCredential(Base):
    """
    Per-user broker credentials, encrypted at rest with Fernet (app.services.crypto).
    One row per (user_id, broker) pair — UNIQUE constraint enforces this.
    Tokens are SEBI-mandated to expire daily; UI shows expires_at to prompt re-login.
    """
    __tablename__ = "broker_credentials"

    id          = Column(Integer,  primary_key=True, index=True)
    user_id     = Column(Integer,  ForeignKey("users.id"), nullable=False, index=True)
    broker      = Column(String,   nullable=False)  # "Upstox" | "Dhan" | "Zerodha" | ...
    # All sensitive strings stored as Fernet ciphertext via app.services.crypto.encrypt/decrypt
    access_token_enc   = Column(Text, nullable=True)
    refresh_token_enc  = Column(Text, nullable=True)
    client_id_enc      = Column(Text, nullable=True)
    api_key_enc        = Column(Text, nullable=True)
    api_secret_enc     = Column(Text, nullable=True)
    expires_at         = Column(DateTime, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)
    updated_at         = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
        # v1.12 — self-learning feature capture
        ("trade_history",    "entry_features_json", "TEXT"),
        ("trade_history",    "signal_label",        "TEXT"),
    ]
    from sqlalchemy import text as _sql_text
    if_not_exists = "" if IS_SQLITE else "IF NOT EXISTS "
    with engine.connect() as conn:
        for table, col, col_type in migrations:
            if not IS_SQLITE:
                col_type = col_type.replace("REAL", "DOUBLE PRECISION")
            try:
                conn.execute(_sql_text(f"ALTER TABLE {table} ADD COLUMN {if_not_exists}{col} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()  # column already exists (SQLite) — safe to ignore

_run_migrations()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
