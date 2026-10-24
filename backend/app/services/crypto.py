"""
ASTRA Symmetric Crypto — encrypts broker credentials at rest.

We use Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` library.
Key is taken from CREDENTIAL_ENCRYPTION_KEY env (must be a urlsafe-base64-encoded
32-byte key). Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If the env var is missing, the key is derived from the server's JWT signing
key (app.services.auth.SECRET_KEY — JWT_SECRET_KEY, else derived from the
Postgres URL), which is private to the deployment. Earlier versions derived it
from JWT_SECRET_KEY with a public constant fallback seed; ciphertext written
with that legacy key is still decrypted (new writes always use the new key).
"""

import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger(__name__)

_SALT = b"::astra-cred-key"


def _derive(seed: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(seed.encode() + _SALT).digest())


def _legacy_key() -> bytes:
    """Key used before v1.14 (public constant seed when JWT_SECRET_KEY was unset)."""
    return _derive(os.getenv("JWT_SECRET_KEY", "dev_fallback_seed_change_me"))


def _resolve_key() -> bytes:
    """Return the primary Fernet key (32-byte url-safe-base64-encoded)."""
    explicit = (os.getenv("CREDENTIAL_ENCRYPTION_KEY") or "").strip()
    if explicit:
        return explicit.encode()
    from app.services.auth import SECRET_KEY
    return _derive(SECRET_KEY)


def _build() -> MultiFernet:
    primary = _resolve_key()
    keys = [primary]
    legacy = _legacy_key()
    if legacy != primary:
        keys.append(legacy)          # decrypt-only: values stored by older versions
    return MultiFernet([Fernet(k) for k in keys])


_fernet = _build()


def encrypt(plaintext: str) -> str:
    """Encrypt a string with the primary key. Returns url-safe-base64 ciphertext."""
    if plaintext is None:
        return ""
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a Fernet token (primary key, then legacy). Returns "" on failure."""
    if not ciphertext:
        return ""
    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.warning("crypto.decrypt: invalid token (wrong key or tampered ciphertext)")
        return ""
