"""
ASTRA Strategy Registry
=========================
Auto-discovers all Strategy subclasses in the app.strategies package and exposes
them via a stable API. Used by:
  - The backtest/evaluate endpoints (list strategies, run by name)
  - The frontend strategy picker
  - The marketplace/listings page

Strategies register themselves simply by existing as a subclass of Strategy in
this package. No manual registration needed.

Convention:
  - First-party engines:   app/strategies/<lowercase_name>.py with one Strategy subclass
  - User strategies:        app/strategies/user/<user_id>/<strategy_name>.py
                            (loaded on-demand by user_id, not at startup)
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Optional, Type

from app.strategies.base import Strategy, StrategyMeta

logger = logging.getLogger(__name__)

# Cached: name → Strategy class
_REGISTRY: dict[str, Type[Strategy]] = {}
_DISCOVERED = False


def _discover_first_party() -> None:
    """Walk app.strategies.* and register every Strategy subclass."""
    global _REGISTRY, _DISCOVERED
    if _DISCOVERED:
        return
    import app.strategies as pkg
    skip_modules = {"base", "registry", "user"}     # don't import these as strategies
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if modinfo.name in skip_modules or modinfo.name.startswith("_"):
            continue
        full_name = f"app.strategies.{modinfo.name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as e:
            logger.warning(f"[registry] failed to import {full_name}: {e}")
            continue

        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if cls is Strategy:
                continue
            if not issubclass(cls, Strategy):
                continue
            # Skip if defined in a different module (e.g. re-imported helpers)
            if cls.__module__ != full_name:
                continue
            try:
                meta: StrategyMeta = cls().meta()
            except Exception as e:
                logger.warning(f"[registry] {cls.__name__}.meta() raised: {e}")
                continue
            _REGISTRY[meta.name] = cls
            logger.info(f"[registry] {meta.name} v{meta.version}  ({cls.__module__}.{cls.__name__})")

    _DISCOVERED = True


def list_strategies() -> list[dict]:
    """Public list — used by frontend strategy picker."""
    _discover_first_party()
    out = []
    for name, cls in sorted(_REGISTRY.items()):
        try:
            meta = cls().meta()
            out.append({
                "name":        meta.name,
                "version":     meta.version,
                "description": meta.description,
                "timeframe":   meta.timeframe.value,
                "asset_class": meta.asset_class,
                "long_only":   meta.long_only,
                "owner":       meta.owner,
                "tags":        meta.tags,
            })
        except Exception as e:
            logger.warning(f"[registry] meta() failed for {name}: {e}")
    return out


def get_strategy(name: str) -> Optional[Strategy]:
    """Return a fresh instance of `name` strategy, or None if not registered."""
    _discover_first_party()
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    return cls()


def get_all() -> list[Strategy]:
    """Instantiate one of every registered strategy. Used by batch evaluators."""
    _discover_first_party()
    return [cls() for cls in _REGISTRY.values()]
