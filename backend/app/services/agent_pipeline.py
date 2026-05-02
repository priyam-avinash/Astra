"""
ASTRA Multi-Agent Pipeline Orchestrator
========================================
Single entry-point that runs the tiered agent pipeline.

Tier activation is controlled by settings + signal properties:

  Tier 0 — always: ML engines (ASTRA 1.0 / AI / ML)
  Tier 1 — LLM enabled: NewsAnalystAgent + FundamentalsScorer
  Tier 2 — LLM enabled + confidence in grey zone (or forced): BullBearDebateAgent
  Tier 3 — LLM enabled + Tier 2 passed: RiskPortfolioAgent

Each tier writes results into a flat dict that accumulates throughout the pipeline
and is returned as the enriched signal payload.

Settings are read from AppSettings DB (entered via UI) with sane defaults.
Per-signal overrides are accepted as kwargs.

Usage (from endpoint):
    from app.services.agent_pipeline import agent_pipeline
    enriched = agent_pipeline.run(symbol, base_signal, db=db, llm_override=True)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _get_setting(key: str, default=None):
    """Read setting from AppSettings DB."""
    try:
        from app.models.database import SessionLocal, AppSettings
        db = SessionLocal()
        try:
            row = db.query(AppSettings).filter(AppSettings.key == key).first()
            if row and row.value is not None:
                # Coerce booleans
                if row.value.lower() in ("true", "1", "yes"):
                    return True
                if row.value.lower() in ("false", "0", "no"):
                    return False
                try:
                    return float(row.value) if "." in row.value else int(row.value)
                except Exception:
                    return row.value
            return default
        finally:
            db.close()
    except Exception:
        return default


class AgentPipeline:

    # Grey-zone: run debate when confidence is in this range
    GREY_ZONE_LOW  = 45.0
    GREY_ZONE_HIGH = 72.0

    def run(
        self,
        symbol:        str,
        base_signal:   dict,          # output from ai_engine.analyze_market_data()
        db=None,                      # SQLAlchemy session (for lessons + positions)
        # Per-signal overrides (None = use global setting)
        llm_override:  Optional[bool] = None,
        auto_execute_override: Optional[str] = None,  # "advisory" / "auto" / None
    ) -> dict:
        """
        Enriches base_signal with agent analysis.

        base_signal must contain at minimum:
          signal, confidence, entry_price, target, stop_loss, current_price

        Returns the base_signal dict augmented with:
          llm_enriched, news, fundamentals, debate, portfolio (all optional — only if LLM runs)
        """
        t0 = time.time()
        result = dict(base_signal)  # copy so we don't mutate caller's dict
        result["llm_enriched"] = False
        result["agent_pipeline_ms"] = 0

        # ── Resolve settings ──────────────────────────────────────────
        global_llm_enabled = _get_setting("llm_enabled", False)
        llm_active = llm_override if llm_override is not None else global_llm_enabled

        if not llm_active:
            return result   # ML-only mode — return as-is

        # ── Extract base signal fields ────────────────────────────────
        signal     = result.get("signal", "HOLD")
        confidence = float(result.get("confidence", 50.0))
        entry      = float(result.get("entry_price", 0))
        sl         = float(result.get("stop_loss",   0))
        tp         = float(result.get("target",      0))
        indicators = self._extract_indicators(result)

        # ── Fetch context data ────────────────────────────────────────
        open_positions = []
        if db is not None:
            try:
                from app.models.database import ActivePosition
                rows = db.query(ActivePosition).filter(
                    ActivePosition.status == "OPEN"
                ).all()
                open_positions = [
                    {"asset": p.asset, "direction": p.direction, "pnl": 0.0}
                    for p in rows
                ]
            except Exception:
                pass

        past_lessons = []
        if db is not None:
            try:
                from app.services.reflector_agent import reflector_agent
                past_lessons = reflector_agent.get_lessons(symbol, db, limit=3)
            except Exception:
                pass

        # ── TIER 1: News + Fundamentals ───────────────────────────────
        news_result = self._run_tier1_news(symbol, signal, confidence)
        fund_result = self._run_tier1_fundamentals(symbol)

        result["news"]         = news_result.to_dict() if news_result else None
        result["fundamentals"] = fund_result.to_dict() if fund_result else None

        news_sentiment = news_result.sentiment if news_result else "NEUTRAL"
        news_summary   = news_result.summary   if news_result else ""
        fund_grade     = fund_result.grade      if fund_result else "C"
        fund_score     = fund_result.score      if fund_result else 50.0

        # Apply Tier 1 gates
        gated = False
        if news_result and news_result.should_gate:
            result["signal"]     = "HOLD"
            result["gated_by"]   = "news"
            result["gate_reason"] = f"Negative news suppressed {signal} signal."
            gated = True
            logger.info(f"[{symbol}] Signal gated by negative news.")

        if not gated and fund_result and fund_result.gate_long and signal == "BUY":
            result["signal"]     = "HOLD"
            result["gated_by"]   = "fundamentals"
            result["gate_reason"] = f"Fundamentals grade {fund_grade} too weak for multi-day BUY."
            gated = True
            logger.info(f"[{symbol}] Signal gated by weak fundamentals ({fund_grade}).")

        result["llm_enriched"] = True

        # If gated, skip Tier 2+3
        if gated:
            result["agent_pipeline_ms"] = round((time.time() - t0) * 1000)
            return result

        # ── TIER 2: Bull/Bear Debate ──────────────────────────────────
        run_debate = (
            self.GREY_ZONE_LOW <= confidence <= self.GREY_ZONE_HIGH
            or _get_setting("always_debate", False)
        )

        debate_result = None
        if run_debate:
            debate_result = self._run_tier2_debate(
                symbol, signal, confidence, indicators,
                news_summary, news_sentiment, fund_grade, past_lessons
            )
            result["debate"] = debate_result.to_dict() if debate_result else None

            if debate_result and debate_result.verdict == "REJECTED":
                result["signal"]     = "HOLD"
                result["gated_by"]   = "debate"
                result["gate_reason"] = f"Bull/Bear debate rejected signal: {debate_result.verdict_reasoning[:80]}"
                result["agent_pipeline_ms"] = round((time.time() - t0) * 1000)
                return result

        debate_verdict   = debate_result.verdict          if debate_result else "CONFIRMED"
        debate_reasoning = debate_result.verdict_reasoning if debate_result else ""
        position_mult    = debate_result.suggested_position_multiplier if debate_result else 1.0
        result["position_multiplier"] = position_mult

        # ── TIER 3: Risk & Portfolio Manager ─────────────────────────
        auto_mode = auto_execute_override or _get_setting("auto_execute_mode", "advisory")
        auto_threshold = _get_setting("auto_execute_threshold", "GREEN")

        portfolio_result = self._run_tier3_risk(
            symbol, signal, confidence, entry, sl, tp,
            debate_verdict, debate_reasoning,
            news_sentiment, news_summary, fund_grade, fund_score,
            open_positions, past_lessons,
            auto_mode, str(auto_threshold),
        )
        result["portfolio"] = portfolio_result.to_dict() if portfolio_result else None

        if portfolio_result:
            result["risk_verdict"]             = portfolio_result.risk_verdict
            result["recommended_position_pct"] = portfolio_result.recommended_position_pct
            result["auto_execute"]             = portfolio_result.auto_execute
            # If portfolio manager updated SL/TP, use those
            if portfolio_result.price_target:
                result["target"] = portfolio_result.price_target
            if portfolio_result.stop_loss:
                result["stop_loss"] = portfolio_result.stop_loss

            if portfolio_result.risk_verdict == "RED":
                result["signal"]     = "HOLD"
                result["gated_by"]   = "portfolio_manager"
                result["gate_reason"] = f"Portfolio Manager vetoed trade (RED): {portfolio_result.executive_summary[:80]}"

        result["agent_pipeline_ms"] = round((time.time() - t0) * 1000)
        logger.info(
            f"[{symbol}] Pipeline complete in {result['agent_pipeline_ms']}ms | "
            f"Signal: {result['signal']} | Debate: {debate_verdict} | "
            f"Risk: {result.get('risk_verdict','—')}"
        )
        return result

    # ── Tier runners ───────────────────────────────────────────────────

    def _run_tier1_news(self, symbol, signal, confidence):
        try:
            from app.services.news_agent import news_agent
            return news_agent.analyze(symbol, signal=signal, confidence=confidence)
        except Exception as e:
            logger.warning(f"Tier1 News failed for {symbol}: {e}")
            return None

    def _run_tier1_fundamentals(self, symbol):
        try:
            from app.services.fundamentals_scorer import fundamentals_scorer
            # Fetch financials inline
            import yfinance as yf, math, warnings
            warnings.filterwarnings("ignore")
            sym = symbol if "." in symbol else symbol + ".NS"
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            def safe(v):
                if v is None: return None
                try:
                    f = float(v)
                    return None if math.isnan(f) or math.isinf(f) else f
                except Exception:
                    return None

            financials = {
                "keyFacts": {
                    "trailingPE":       safe(info.get("trailingPE")),
                    "priceToBook":      safe(info.get("priceToBook")),
                    "returnOnEquity":   safe(info.get("returnOnEquity")),
                    "beta":             safe(info.get("beta")),
                },
                "growth": {
                    "profitMargins":    safe(info.get("profitMargins")),
                    "operatingMargins": safe(info.get("operatingMargins")),
                    "revenueGrowth":    safe(info.get("revenueGrowth")),
                    "earningsGrowth":   safe(info.get("earningsGrowth")),
                    "currentRatio":     safe(info.get("currentRatio")),
                    "debtToEquity":     safe(info.get("debtToEquity")),
                },
            }
            return fundamentals_scorer.score(financials)
        except Exception as e:
            logger.warning(f"Tier1 Fundamentals failed for {symbol}: {e}")
            return None

    def _run_tier2_debate(
        self, symbol, signal, confidence, indicators,
        news_summary, news_sentiment, fund_grade, past_lessons
    ):
        try:
            from app.services.debate_agent import debate_agent
            return debate_agent.debate(
                symbol=symbol, signal=signal, confidence=confidence,
                indicators=indicators, news_summary=news_summary,
                news_sentiment=news_sentiment, fund_grade=fund_grade,
                past_lessons=past_lessons,
            )
        except Exception as e:
            logger.warning(f"Tier2 Debate failed for {symbol}: {e}")
            return None

    def _run_tier3_risk(
        self, symbol, signal, confidence, entry, sl, tp,
        debate_verdict, debate_reasoning, news_sentiment, news_summary,
        fund_grade, fund_score, open_positions, past_lessons,
        auto_mode, auto_threshold,
    ):
        try:
            from app.services.risk_agent import risk_portfolio_agent
            return risk_portfolio_agent.assess(
                symbol=symbol, signal=signal, confidence=confidence,
                entry_price=entry, stop_loss=sl, target_price=tp,
                debate_verdict=debate_verdict, debate_reasoning=debate_reasoning,
                news_sentiment=news_sentiment, news_summary=news_summary,
                fund_grade=fund_grade, fund_score=fund_score,
                open_positions=open_positions, past_lessons=past_lessons,
                auto_execute_mode=auto_mode,
                auto_execute_threshold=auto_threshold,
            )
        except Exception as e:
            logger.warning(f"Tier3 Risk failed for {symbol}: {e}")
            return None

    # ── Indicator extractor ────────────────────────────────────────────
    def _extract_indicators(self, signal_dict: dict) -> dict:
        """Pull named indicator values from the signal dict for debate context."""
        keys = ["RSI", "MACD", "ADX", "Stoch_K", "BB_PctB",
                "Dist_SMA200", "Volume_Ratio", "OBV_Slope"]
        result = {}
        # Also check chartData last row
        chart = signal_dict.get("chartData", [])
        last  = chart[-1] if chart else {}
        for k in keys:
            v = signal_dict.get(k) or last.get(k)
            if v is not None:
                result[k] = v
        return result


# ── Module singleton ────────────────────────────────────────────────────────
agent_pipeline = AgentPipeline()
