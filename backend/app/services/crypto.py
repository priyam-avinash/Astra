"""
ASTRA Symmetric Crypto — encrypts broker credentials at rest.

We use Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` library.
Key is derived from CREDENTIAL_ENCRYPTION_KEY env (must be a urlsafe-base64-encoded
32-byte key). Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If the env var is missing:
  - In test/dev (PAPER_TRADING=true), we use a deterministic key derived from JWT_SECRET_KEY
    so existing tests/dev work without manual setup.
  - In production (PAPER_TRADING=false), we REFUSE to start.

The deterministic dev key is intentionally weak — it just means we don't crash in dev
when the env isn't fully configured. Production MUST set CREDENTIAL_ENCRYPTION_KEY.
"""

import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _resolve_key() -> bytes:
    """Return the Fernet key (32-byte url-safe-base64-encoded)."""
    explicit = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
    if explicit:
        return explicit.encode()

    # Dev/test fallback — deterministic from JWT_SECRET_KEY
    paper = os.getenv("PAPER_TRADING", "true").lower() != "false"
    if not paper:
        raise RuntimeError(
            "REFUSING TO START: CREDENTIAL_ENCRYPTION_KEY is not set and PAPER_TRADING=false. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"  and set it in env."
        )
    seed = os.getenv("JWT_SECRET_KEY", "dev_fallback_seed_change_me")
    derived = hashlib.sha256(seed.encode() + b"::astra-cred-key").digest()
    return base64.urlsafe_b64encode(derived)


_KEY = _resolve_key()
_fernet = Fernet(_KEY)


def encrypt(plaintext: str) -> str:
    """Encrypt a string. Returns url-safe-base64 ciphertext."""
    if plaintext is None:
        return ""
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a Fernet token. Returns the original plaintext, or "" on failure."""
    if not ciphertext:
        return ""
    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.warning("crypto.decrypt: invalid token (wrong key or tampered ciphertext)")
        return ""
