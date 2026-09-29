"""
ASTRA runtime configuration
===========================
Single place that (1) loads .env files and (2) resolves secrets.

Why this exists
---------------
Before v1.13 only broker.py called load_dotenv(), so whether a module saw
TWELVE_DATA_KEY / DHAN_* depended on import order. Keys could also be entered
in the Settings UI (stored in the app_settings table), but data providers only
read os.environ — so UI-entered keys were silently ignored.

Import this module first (main.py does). Everything else calls get_secret().
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]      # …/backend
REPO_DIR = BACKEND_DIR.parent                           # …/ (repo root)

# Serverless (Vercel): the code directory is read-only and there is no
# long-running process, so writable state goes to /tmp and background
# loops (Dhan feed, position monitor) are replaced by on-request work.
SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("ASTRA_SERVERLESS"))
DATA_DIR = Path(os.getenv("ASTRA_DATA_DIR") or ("/tmp/astra-data" if SERVERLESS else BACKEND_DIR / "data"))


def _load_env_files() -> list[str]:
    loaded = []
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return loaded
    # backend/.env wins over repo-root .env; real environment wins over both.
    for path in (BACKEND_DIR / ".env", REPO_DIR / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(str(path))
    return loaded


ENV_FILES_LOADED = _load_env_files()

# Values people leave in .env templates. Treated as "not configured".
_PLACEHOLDER_MARKERS = ("enter_your", "your_", "changeme", "change_me", "xxxx", "<", "placeholder")


def is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip().strip('"').strip("'")
    if not v or v.lower() in {"none", "null", "false", "0"}:
        return True
    low = v.lower()
    return any(m in low for m in _PLACEHOLDER_MARKERS)


# Settings-UI key  →  env var names (first match wins)
SECRET_ENV_ALIASES = {
    "twelve_data_key":    ("TWELVE_DATA_KEY", "TWELVE_DATA_API_KEY", "TWELVEDATA_API_KEY"),
    "alpha_vantage_key":  ("ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_KEY"),
    "dhan_client_id":     ("DHAN_CLIENT_ID",),
    "dhan_access_token":  ("DHAN_ACCESS_TOKEN",),
    "anthropic_api_key":  ("ANTHROPIC_API_KEY",),
    "groq_api_key":       ("GROQ_API_KEY",),
}

# The old code shipped a shared public Alpha Vantage key as a default. It is
# permanently over quota, so it is never used.
_BANNED_VALUES = {"XV1FMHS5UHPIIPAZ"}


def _db_setting(key: str) -> Optional[str]:
    try:
        from app.models.database import SessionLocal, AppSettings
        db = SessionLocal()
        try:
            row = db.query(AppSettings).filter(AppSettings.key == key).first()
            return row.value if row and row.value else None
        finally:
            db.close()
    except Exception:
        return None


def get_secret(name: str) -> Optional[str]:
    """
    Resolve a secret by its settings key (e.g. "twelve_data_key").
    Order: Settings UI (DB) → environment / .env. Placeholders count as unset.
    """
    candidates = [_db_setting(name)]
    for env_name in SECRET_ENV_ALIASES.get(name, (name.upper(),)):
        candidates.append(os.getenv(env_name))
    for value in candidates:
        if value is None:
            continue
        v = str(value).strip().strip('"').strip("'")
        if v in _BANNED_VALUES or is_placeholder(v):
            continue
        return v
    return None


def secret_source(name: str) -> str:
    """Where a secret comes from: 'settings', 'env' or 'missing' (for diagnostics)."""
    db_val = _db_setting(name)
    if db_val and not is_placeholder(db_val) and db_val.strip() not in _BANNED_VALUES:
        return "settings"
    return "env" if get_secret(name) else "missing"


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
