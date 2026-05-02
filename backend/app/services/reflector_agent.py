"""
ASTRA Reflector Agent — Continuous Learning Loop
=================================================
Tier 4 — Deferred, async (runs after position close via Celery).

After each position closes, the Reflector:
  1. Reviews the original signal, entry/exit prices, and P&L
  2. Uses the LLM to generate a 2-4 sentence reflection
  3. Extracts 1-2 actionable lessons and stores them in AgentMemory
  4. These lessons are injected into future debates/news-analysis for the same ticker

Falls back to a rule-based reflection if LLM is unavailable.
Lessons persist in the DB — they accumulate over time, making ASTRA progressively smarter.
"""

import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class ReflectorAgent:

    def reflect(
        self,
        symbol:         str,
        original_signal: str,
        confidence:     float,
        entry_price:    float,
        exit_price:     float,
        pnl:            float,
        pnl_pct:        float,
        exit_reason:    str,
        news_sentiment: str = "NEUTRAL",
        fund_grade:     str = "C",
        debate_verdict: str = "CONFIRMED",
        risk_verdict:   str = "GREEN",
        trade_id:       int = None,
        db = None,
    ) -> str:
        """
        Generate a reflection and store lessons.
        Returns the reflection text.

        db — SQLAlchemy session (if provided, stores to AgentMemory)
        """
        outcome = "profitable" if pnl >= 0 else "losing"

        # ── LLM reflection ────────────────────────────────────────────
        reflection_text = self._llm_reflect(
            symbol, original_signal, confidence,
            entry_price, exit_price, pnl, pnl_pct,
            exit_reason, news_sentiment, fund_grade,
            debate_verdict, risk_verdict, outcome,
        )

        # ── Store lessons in DB ───────────────────────────────────────
        if db is not None:
            lessons = self._extract_lessons(
                symbol, original_signal, pnl_pct, exit_reason,
                news_sentiment, fund_grade, reflection_text
            )
            for lesson in lessons:
                self._store_lesson(db, symbol, lesson, trade_id)

        return reflection_text

    def get_lessons(self, symbol: str, db, limit: int = 3) -> List[str]:
        """
        Fetch the most recent lessons for this ticker to inject into agent prompts.
        Returns an empty list if no lessons exist yet.
        """
        try:
            from app.models.database import AgentMemory
            rows = (
                db.query(AgentMemory)
                .filter(AgentMemory.symbol == symbol)
                .order_by(AgentMemory.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.content for r in rows]
        except Exception as e:
            logger.warning(f"Failed to fetch lessons for {symbol}: {e}")
            return []

    def get_all_lessons(self, db, limit: int = 3) -> List[str]:
        """
        Fetch the most recent cross-ticker lessons (for Portfolio Manager context).
        """
        try:
            from app.models.database import AgentMemory
            rows = (
                db.query(AgentMemory)
                .filter(AgentMemory.memory_type == "cross_ticker")
                .order_by(AgentMemory.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.content for r in rows]
        except Exception as e:
            logger.warning(f"Failed to fetch cross-ticker lessons: {e}")
            return []

    # ── Private helpers ────────────────────────────────────────────────

    def _llm_reflect(
        self,
        symbol, original_signal, confidence,
        entry_price, exit_price, pnl, pnl_pct,
        exit_reason, news_sentiment, fund_grade,
        debate_verdict, risk_verdict, outcome,
    ) -> str:
        try:
            from app.services.llm_router import get_fast_router, LLMUnavailableError
            router = get_fast_router()

            direction = "gain" if pnl >= 0 else "loss"
            system = (
                "You are ASTRA's self-learning trading engine. You review closed trades and "
                "generate brief, actionable reflections that improve future signal quality. "
                "Be specific about what indicators or conditions were correct or misleading. "
                "Write exactly 2-4 sentences."
            )
            user = (
                f"Symbol: {symbol}\n"
                f"Signal: {original_signal} (confidence {confidence:.0f}%)\n"
                f"Entry: ₹{entry_price:.2f} → Exit: ₹{exit_price:.2f}\n"
                f"Result: {direction} of ₹{abs(pnl):.2f} ({pnl_pct:+.1f}%)\n"
                f"Exit reason: {exit_reason}\n"
                f"News at entry: {news_sentiment} | Fundamentals: {fund_grade}\n"
                f"Debate verdict: {debate_verdict} | Risk verdict: {risk_verdict}\n\n"
                f"Write a brief trading reflection (2-4 sentences) on what went right or wrong "
                f"and what should be checked differently next time for {symbol}."
            )
            return router.complete(system, user, max_tokens=200)

        except Exception as e:
            logger.warning(f"LLM reflection failed for {symbol}: {e}")
            return self._rule_based_reflection(
                symbol, original_signal, pnl_pct, exit_reason,
                news_sentiment, fund_grade
            )

    def _rule_based_reflection(
        self, symbol, signal, pnl_pct, exit_reason,
        news_sentiment, fund_grade
    ) -> str:
        """Simple template-based reflection when LLM unavailable."""
        if pnl_pct >= 3:
            quality = "strong"
            verdict = "The signal was well-calibrated."
        elif pnl_pct >= 0:
            quality = "marginal"
            verdict = "The signal was directionally correct but produced limited returns."
        elif pnl_pct >= -3:
            quality = "weak"
            verdict = "The signal underperformed; consider tightening the confidence threshold."
        else:
            quality = "failed"
            verdict = "The signal produced a significant loss; review the indicator confluence for this pattern."

        news_note = ""
        if news_sentiment == "NEGATIVE":
            news_note = f" Negative news sentiment at entry may have contributed to the {quality} outcome."

        fund_note = ""
        if fund_grade in ("D", "F"):
            fund_note = " Weak fundamentals — consider filtering out D/F-grade stocks."

        return (
            f"{verdict}{news_note}{fund_note} "
            f"Exit triggered by: {exit_reason}. "
            f"Overall trade return: {pnl_pct:+.1f}%."
        )

    def _extract_lessons(
        self, symbol, signal, pnl_pct, exit_reason,
        news_sentiment, fund_grade, reflection_text
    ) -> List[str]:
        """
        Extract 1-2 structured lessons from the reflection text.
        These are stored separately from the reflection for easy injection.
        """
        lessons = []

        # Rule-based lesson extraction (supplemented by LLM reflection text)
        if pnl_pct < -5:
            lessons.append(
                f"[{symbol}] {signal} signal at this confidence level resulted in "
                f"{pnl_pct:.1f}% loss. Double-check ADX and MACD alignment before next {signal}."
            )
        elif news_sentiment == "NEGATIVE" and pnl_pct < 0:
            lessons.append(
                f"[{symbol}] Negative news at entry correlated with a losing trade ({pnl_pct:.1f}%). "
                f"Suppress {signal} signals when news sentiment is negative for this ticker."
            )
        elif pnl_pct > 5:
            lessons.append(
                f"[{symbol}] {signal} with {news_sentiment} news and {fund_grade} fundamentals "
                f"yielded +{pnl_pct:.1f}%. This setup has historically performed well."
            )

        if fund_grade in ("D", "F") and pnl_pct < 0:
            lessons.append(
                f"[{symbol}] D/F fundamentals + {signal} signal = losing trade. "
                f"Consider blocking {signal} signals for weak-fundamental stocks."
            )

        # Always store the LLM reflection itself as a lesson if it's meaningful
        if len(reflection_text) > 40 and "LLM unavailable" not in reflection_text:
            lessons.append(f"[{symbol}] ASTRA reflection: {reflection_text[:200]}")

        return lessons[:2]  # max 2 lessons per trade

    def _store_lesson(self, db, symbol: str, content: str, trade_id: Optional[int]) -> None:
        try:
            from app.models.database import AgentMemory
            row = AgentMemory(
                symbol=symbol,
                memory_type="ticker_lesson",
                content=content,
                source_trade_id=trade_id,
                created_at=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            logger.info(f"Lesson stored for {symbol}: {content[:60]}…")
        except Exception as e:
            logger.warning(f"Failed to store lesson: {e}")
            try:
                db.rollback()
            except Exception:
                pass


# ── Module singleton ────────────────────────────────────────────────────────
reflector_agent = ReflectorAgent()
