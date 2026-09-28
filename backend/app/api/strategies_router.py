"""
Strategy Marketplace API
==========================
Exposes the strategy registry over HTTP so the frontend can:
  - List available strategies (free / pro / pro-plus tiers)
  - Run backtests on demand
  - Get strategy metadata for marketplace cards

Endpoints:
  GET  /strategies                  list all registered strategies (metadata only)
  GET  /strategies/{name}           detailed metadata for one strategy
  POST /strategies/{name}/backtest  run the full evaluator on this strategy
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("")
async def list_all_strategies():
    """Return metadata for every registered strategy."""
    from app.strategies.registry import list_strategies
    return {"strategies": list_strategies()}


@router.get("/{name}")
async def get_strategy_detail(name: str):
    """Return detailed metadata for one strategy."""
    from app.strategies.registry import get_strategy
    eng = get_strategy(name)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    meta = eng.meta()
    return {
        "name":        meta.name,
        "version":     meta.version,
        "description": meta.description,
        "timeframe":   meta.timeframe.value,
        "asset_class": meta.asset_class,
        "long_only":   meta.long_only,
        "owner":       meta.owner,
        "tags":        meta.tags,
        "universe_size": len(eng.universe()),
    }


@router.post("/{name}/backtest")
async def backtest_strategy(name: str):
    """
    Run the full evaluator on `name`.
    NOTE: this is a synchronous call — backtest takes 30-90 seconds.
    Frontend should show a loading state. Future: queue via Celery for long runs.
    """
    from app.strategies.registry import get_strategy
    from app.services.strategy_evaluator import evaluate

    eng = get_strategy(name)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    try:
        report = evaluate(eng, max_workers=4)
    except Exception as e:
        logger.exception(f"Backtest of {name} failed")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    return {
        "strategy":         report.strategy_name,
        "version":          report.strategy_version,
        "verdict":          report.verdict,
        "reasons":          report.reasons,
        "elapsed_sec":      report.elapsed_sec,
        "base_stats":       report.base_stats.__dict__,
        "stress_stats":     report.stress_stats.__dict__,
        "extreme_stats":    report.extreme_stats.__dict__,
        "regime_pass_rate": report.regime_pass_rate,
        "regime_buckets":   report.regime_buckets,
    }
