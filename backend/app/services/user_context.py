"""
Request-scoped current-user context.

In FastAPI's async model, simple module-level globals don't work for per-request
state — a single process serves many concurrent requests. We use `contextvars`
which is the stdlib mechanism for this: each task gets its own view of the var.

Setting:
  - `app.api.endpoints.get_current_user` sets it when JWT validates.
  - Tests can set it directly via `set_current_user_id(int)`.

Reading:
  - Broker plugin adapters read it to know which user's tokens to fetch.
  - Returns `None` when called outside a request (backtest scripts, etc.) —
    callers fall back to env-var credentials in that case.
"""

from contextvars import ContextVar
from typing import Optional

# The actual context variable. None = "no user context" (script / unauthenticated).
_current_user_id: ContextVar[Optional[int]] = ContextVar(
    "astra_current_user_id", default=None
)


def set_current_user_id(user_id: Optional[int]) -> None:
    """Set the current user id for the active request/task."""
    _current_user_id.set(user_id)


def get_current_user_id() -> Optional[int]:
    """Read the current user id. Returns None if not set."""
    return _current_user_id.get()


class user_scope:
    """
    Context manager — temporarily sets a user id. Useful in tests.

    Usage:
        with user_scope(42):
            ... code that calls broker plugins ...
    """
    def __init__(self, user_id: Optional[int]):
        self._uid = user_id
        self._token = None

    def __enter__(self):
        self._token = _current_user_id.set(self._uid)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _current_user_id.reset(self._token)
