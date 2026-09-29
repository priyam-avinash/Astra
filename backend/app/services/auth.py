from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

import hashlib
import logging
import secrets

_log = logging.getLogger(__name__)


def _resolve_secret_key() -> str:
    """JWT signing key.

    1. JWT_SECRET_KEY if set (and not a template placeholder).
    2. Otherwise derived from the database URL — a value only the server
       knows — so hosted deploys (Vercel + Postgres) get a stable, private
       key without anyone having to paste one in.
    3. Local dev falls back to a fixed dev key; a serverless host with no
       database gets a random per-instance key (logins won't survive restarts).
    """
    from app.core.config import is_placeholder, SERVERLESS
    key = (os.getenv("JWT_SECRET_KEY") or "").strip()
    if key and not is_placeholder(key):
        return key
    for name in ("DATABASE_URL", "POSTGRES_URL", "POSTGRES_PRISMA_URL"):
        db = (os.getenv(name) or "").strip()
        if db.startswith(("postgres://", "postgresql")):
            return hashlib.sha256(f"astra-jwt-v1|{db}".encode()).hexdigest()
    if SERVERLESS:
        _log.warning("No JWT_SECRET_KEY or database configured — using a random per-instance key")
        return secrets.token_hex(32)
    return "astra_local_dev_only_secret"


SECRET_KEY = _resolve_secret_key()
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60 * 24 * 7)) # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
