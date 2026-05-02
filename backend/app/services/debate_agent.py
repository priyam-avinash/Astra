"""
ASTRA Bull/Bear Debate Agent
=============================
Tier 2 — Signal Validation.

Runs a 3-call LLM debate pipeline:
  1. Bull Researcher  — argues the case FOR the trade
  2. Bear Researcher  — argues the case AGAINST the trade (reads Bull's argument)
  3. Research Manager — reads both arguments and delivers a structured verdict

Returns DebateResult with:
  - bull_argument:               str
  - bear_argument:               str
  - verdict:                     CONFIRMED / DOWNGRADED / REJECTED
  - verdict_reasoning:           str
  - suggested_position_multiplier: float (1.0 / 0.5 / 0.0)

Triggered only for grey-zone confidence (45-70%) or when explicitly requested.
Falls back gracefully if LLM is unavailable.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    bull_argument:                str
    bear_argument:                str
    verdict:                      str    # CONFIRMED / DOWNGRADED / REJECTED
    verdict_reasoning:            str
    suggested_position_multiplier: float  # 1.0 / 0.5 / 0.0
    source:                       str    # "llm" or "rule_based"

    def to_dict(self) -> dict:
        return {
            "bull_argument":                self.bull_argument,
            "bear_argument":                self.bear_argument,
            "verdict":                      self.verdict,
            "verdict_reasoning":            self.verdict_reasoning,
            "suggested_position_multiplier": self.suggested_position_multiplier,
            "source":                       self.source,
        }


class BullBearDebateAgent:

    def debate(
        self,
        symbol:           str,
        signal:           str,
        confidence:       float,
        indicators:       dict,
        news_summary:     str  = "",
        news_sentiment:   str  = "NEUTRAL",
        fund_grade:       str  = "C",
        past_lessons:     list = None,
    ) -> DebateResult:
        """
        symbol       — ticker (e.g. RELIANCE.NS)
        signal       — BUY / SELL / HOLD
        confidence   — ML confidence 0-100
        indicators   — dict of key indicators (RSI, MACD, ADX, etc.)
        news_summary — 2-3 sentence news context
        news_sentiment — POSITIVE/NEGATIVE/NEUTRAL/MIXED
        fund_grade   — A/B/C/D/F fundamentals grade
        past_lessons — list of reflection strings from ReflectorAgent
        """
        past_lessons = past_lessons or []

        try:
            from app.services.llm_router import get_fast_router, get_smart_router, LLMUnavailableError
            fast_router  = get_fast_router()
            smart_router = get_smart_router()

            context = self._build_context(
                symbol, signal, confidence, indicators,
                news_summary, news_sentiment, fund_grade, past_lessons
            )

            # ── Round 1: Bull ──────────────────────────────────────────
            bull_sys = (
                "You are a bullish equity researcher. Your job is to build the strongest "
                "possible case FOR this trade based on the data provided. Be specific, "
                "cite the indicators, and limit your response to 150 words."
            )
            bull_arg = fast_router.complete(bull_sys, context, max_tokens=250)

            # ── Round 2: Bear (reads Bull's argument) ─────────────────
            bear_sys = (
                "You are a bearish equity researcher. Your job is to build the strongest "
                "possible case AGAINST this trade. Directly rebut the bull's argument "
                "where possible. Be specific, cite risks, and limit to 150 words."
            )
            bear_user = f"{context}\n\n--- Bull Researcher's Argument ---\n{bull_arg}"
            bear_arg  = fast_router.complete(bear_sys, bear_user, max_tokens=250)

            # ── Round 3: Research Manager (synthesis + verdict) ────────
            mgr_sys = (
                "You are a senior research manager at an institutional equity desk. "
                "You have read arguments from both a bull and a bear researcher. "
                "Your job is to deliver a clear, unbiased verdict on whether to proceed "
                "with this trade. You must respond in valid JSON only."
            )
            mgr_user = (
                f"{context}\n\n"
                f"--- Bull Researcher ---\n{bull_arg}\n\n"
                f"--- Bear Researcher ---\n{bear_arg}\n\n"
                f"Deliver your verdict as JSON:\n"
                f'{{"verdict": "CONFIRMED|DOWNGRADED|REJECTED", '
                f'"reasoning": "2-3 sentences explaining your decision", '
                f'"key_risk": "single biggest risk to this trade", '
                f'"key_catalyst": "single strongest argument for the trade"}}'
            )
            verdict_dict = smart_router.complete_json(
                mgr_sys, mgr_user,
                schema_hint='{"verdict":"str","reasoning":"str","key_risk":"str","key_catalyst":"str"}',
                max_tokens=300
            )

            verdict   = verdict_dict.get("verdict", "CONFIRMED").upper()
            reasoning = verdict_dict.get("reasoning", "")
            if verdict not in ("CONFIRMED", "DOWNGRADED", "REJECTED"):
                verdict = "CONFIRMED"

            multiplier = {"CONFIRMED": 1.0, "DOWNGRADED": 0.5, "REJECTED": 0.0}.get(verdict, 1.0)

            return DebateResult(
                bull_argument=bull_arg,
                bear_argument=bear_arg,
                verdict=verdict,
                verdict_reasoning=reasoning,
                suggested_position_multiplier=multiplier,
                source="llm",
            )

        except Exception as e:
            logger.warning(f"Debate agent failed for {symbol}: {e}")
            return self._rule_based_fallback(signal, confidence, news_sentiment, fund_grade)

    # ── Context builder ────────────────────────────────────────────────
    def _build_context(self, symbol, signal, confidence, indicators,
                       news_summary, news_sentiment, fund_grade, past_lessons) -> str:
        ind_str = ""
        for k, v in (indicators or {}).items():
            if v is not None:
                try:
                    ind_str += f"  {k}: {float(v):.2f}\n"
                except Exception:
                    ind_str += f"  {k}: {v}\n"

        lessons_str = ""
        if past_lessons:
            lessons_str = "\n\nPast lessons from previous trades on this ticker:\n"
            for i, lesson in enumerate(past_lessons[:3], 1):
                lessons_str += f"  {i}. {lesson}\n"

        return (
            f"Symbol: {symbol}\n"
            f"ML Signal: {signal} (Confidence: {confidence:.1f}%)\n"
            f"News Sentiment: {news_sentiment}\n"
            f"News Summary: {news_summary or 'No recent news.'}\n"
            f"Fundamentals Grade: {fund_grade}\n\n"
            f"Technical Indicators:\n{ind_str}"
            f"{lessons_str}"
        )

    # ── Rule-based fallback (no LLM) ──────────────────────────────────
    def _rule_based_fallback(
        self, signal: str, confidence: float,
        news_sentiment: str, fund_grade: str
    ) -> DebateResult:
        """Simple heuristic verdict when LLM is unavailable."""

        neg_count = sum([
            news_sentiment == "NEGATIVE",
            fund_grade in ("D", "F"),
            confidence < 50,
        ])

        if neg_count >= 2:
            verdict    = "REJECTED"
            multiplier = 0.0
            reasoning  = "Rule-based: negative news + weak fundamentals + low confidence."
        elif neg_count == 1 or confidence < 60:
            verdict    = "DOWNGRADED"
            multiplier = 0.5
            reasoning  = "Rule-based: one or more risk factors present; position size reduced."
        else:
            verdict    = "CONFIRMED"
            multiplier = 1.0
            reasoning  = "Rule-based: no major risk flags detected."

        return DebateResult(
            bull_argument="LLM unavailable — rule-based assessment used.",
            bear_argument="LLM unavailable — rule-based assessment used.",
            verdict=verdict,
            verdict_reasoning=reasoning,
            suggested_position_multiplier=multiplier,
            source="rule_based",
        )


# ── Module singleton ────────────────────────────────────────────────────────
debate_agent = BullBearDebateAgent()
