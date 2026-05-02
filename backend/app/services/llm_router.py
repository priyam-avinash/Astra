"""
ASTRA LLM Router
================
Provider chain: Anthropic (Claude) → Groq (free, llama-3.3-70b) → Rule-based fallback.

Keys are loaded from AppSettings DB at runtime (entered via Settings UI) with .env as fallback.
When neither key is present, LLM features degrade gracefully — callers receive a LLMUnavailableError
which the agent pipeline catches and handles with rule-based substitutes.
"""

import json
import logging
import os
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when no LLM provider is configured and rule-based fallback is needed."""
    pass


def _get_setting(key: str) -> Optional[str]:
    """Read a setting from AppSettings DB. Returns None if not found."""
    try:
        from app.models.database import SessionLocal, AppSettings
        db = SessionLocal()
        try:
            row = db.query(AppSettings).filter(AppSettings.key == key).first()
            return row.value if row and row.value and row.value.strip() else None
        finally:
            db.close()
    except Exception:
        return None


class LLMRouter:
    """
    Single-entry-point LLM abstraction.

    Usage:
        router = LLMRouter()
        text   = router.complete(system="You are...", user="Analyse...")
        obj    = router.complete_json(system="...", user="...", schema_hint="Return JSON: {field: ...}")
    """

    # ── Model selection ──────────────────────────────────────────────
    ANTHROPIC_FAST   = "claude-haiku-4-5"          # cheap/fast — analysts
    ANTHROPIC_SMART  = "claude-sonnet-4-5"          # powerful  — portfolio manager
    GROQ_FAST        = "llama-3.1-8b-instant"       # Groq free tier, very fast
    GROQ_SMART       = "llama-3.3-70b-versatile"    # Groq free tier, more capable

    def __init__(self, prefer_smart: bool = False):
        """
        prefer_smart=True  → use the more capable model (Portfolio Manager, Debate arbitrators)
        prefer_smart=False → use the faster/cheaper model (Analysts, individual debate turns)
        """
        self.prefer_smart = prefer_smart

    # ── Provider detection ────────────────────────────────────────────
    def _anthropic_key(self) -> Optional[str]:
        return (
            _get_setting("anthropic_api_key")
            or os.getenv("ANTHROPIC_API_KEY", "")
            or None
        )

    def _groq_key(self) -> Optional[str]:
        return (
            _get_setting("groq_api_key")
            or os.getenv("GROQ_API_KEY", "")
            or None
        )

    def provider_status(self) -> dict:
        """Returns which providers are configured (for Settings UI)."""
        ak = self._anthropic_key()
        gk = self._groq_key()
        active = "anthropic" if ak else ("groq" if gk else "none")
        return {
            "anthropic_configured": bool(ak),
            "groq_configured":      bool(gk),
            "active_provider":      active,
            "llm_available":        active != "none",
        }

    # ── Core completion ───────────────────────────────────────────────
    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        """
        Text completion. Tries Anthropic first, then Groq, then raises LLMUnavailableError.
        """
        ak = self._anthropic_key()
        if ak:
            return self._anthropic_complete(ak, system, user, max_tokens)

        gk = self._groq_key()
        if gk:
            return self._groq_complete(gk, system, user, max_tokens)

        raise LLMUnavailableError("No LLM provider configured. Add Anthropic or Groq key in Settings.")

    def complete_json(self, system: str, user: str, schema_hint: str = "",
                      max_tokens: int = 600) -> dict:
        """
        Structured JSON completion. Appends schema hint to system prompt.
        Parses the response and returns a Python dict.
        Falls back to empty dict on parse failure (never raises).
        """
        json_system = system + (
            f"\n\nYou MUST respond with valid JSON only. No markdown, no explanation outside JSON. "
            f"Schema: {schema_hint}"
        )
        try:
            raw = self.complete(json_system, user, max_tokens=max_tokens)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except LLMUnavailableError:
            raise
        except Exception as e:
            logger.warning(f"LLM JSON parse failed: {e}")
            return {}

    # ── Anthropic backend ──────────────────────────────────────────────
    def _anthropic_complete(self, key: str, system: str, user: str, max_tokens: int) -> str:
        model = self.ANTHROPIC_SMART if self.prefer_smart else self.ANTHROPIC_FAST
        headers = {
            "x-api-key":         key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    # ── Groq backend (OpenAI-compatible) ──────────────────────────────
    def _groq_complete(self, key: str, system: str, user: str, max_tokens: int) -> str:
        model = self.GROQ_SMART if self.prefer_smart else self.GROQ_FAST
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":      model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── Convenience singletons ─────────────────────────────────────────────────
_fast_router  = None
_smart_router = None

def get_fast_router() -> LLMRouter:
    global _fast_router
    if _fast_router is None:
        _fast_router = LLMRouter(prefer_smart=False)
    return _fast_router

def get_smart_router() -> LLMRouter:
    global _smart_router
    if _smart_router is None:
        _smart_router = LLMRouter(prefer_smart=True)
    return _smart_router
