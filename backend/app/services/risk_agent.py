"""
ASTRA Risk & Portfolio Manager Agents
======================================
Tier 3 — Institutional Risk Assessment (the most powerful tier).

Runs a 4-call LLM pipeline:
  1. Aggressive Debater    — champions the upside; argues for taking the trade
  2. Conservative Debater  — champions capital preservation; argues against / de-risks
  3. Neutral Debater       — balances both views; proposes a middle path
  4. Portfolio Manager     — reads all three views + past context, delivers the final
                             authoritative PortfolioDecision

PortfolioDecision contains:
  - risk_verdict:              GREEN / AMBER / RED
  - executive_summary:         str
  - investment_thesis:         str
  - recommended_position_pct:  float  (0–100 — % of per-trade capital to deploy)
  - price_target:              float
  - time_horizon:              str    (e.g. "3–7 trading days")
  - auto_execute:              bool   (True when GREEN and auto_execute mode is on)

The Portfolio Manager is the ONLY agent with final authority to produce the
PortfolioDecision. It uses the "smart" LLM model (Claude Sonnet or Groq 70B).

Falls back gracefully to rule-based assessment if LLM is unavailable.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioDecision:
    # Risk verdict
    risk_verdict:             str     # GREEN / AMBER / RED
    executive_summary:        str
    investment_thesis:        str
    key_risks:                str
    # Position guidance
    recommended_position_pct: float   # 0–100
    price_target:             Optional[float]
    stop_loss:                Optional[float]
    time_horizon:             str
    # Debate transcripts
    aggressive_view:          str
    conservative_view:        str
    neutral_view:             str
    # Meta
    auto_execute:             bool
    source:                   str     # "llm" or "rule_based"

    def to_dict(self) -> dict:
        return {
            "risk_verdict":             self.risk_verdict,
            "executive_summary":        self.executive_summary,
            "investment_thesis":        self.investment_thesis,
            "key_risks":                self.key_risks,
            "recommended_position_pct": round(self.recommended_position_pct, 1),
            "price_target":             self.price_target,
            "stop_loss":                self.stop_loss,
            "time_horizon":             self.time_horizon,
            "aggressive_view":          self.aggressive_view,
            "conservative_view":        self.conservative_view,
            "neutral_view":             self.neutral_view,
            "auto_execute":             self.auto_execute,
            "source":                   self.source,
        }


class RiskPortfolioAgent:

    def assess(
        self,
        symbol:             str,
        signal:             str,
        confidence:         float,
        entry_price:        float,
        stop_loss:          float,
        target_price:       float,
        debate_verdict:     str          = "CONFIRMED",
        debate_reasoning:   str          = "",
        news_sentiment:     str          = "NEUTRAL",
        news_summary:       str          = "",
        fund_grade:         str          = "C",
        fund_score:         float        = 50.0,
        open_positions:     List[dict]   = None,
        past_lessons:       List[str]    = None,
        auto_execute_mode:  str          = "advisory",   # "advisory" or "auto"
        auto_execute_threshold: str      = "GREEN",      # "GREEN" or "AMBER_GREEN"
    ) -> PortfolioDecision:
        """
        Full risk debate pipeline.

        open_positions  — list of dicts from /api/positions (for concentration check)
        past_lessons    — list of reflection strings from ReflectorAgent
        auto_execute_mode — from per-trade or global settings
        """
        open_positions = open_positions or []
        past_lessons   = past_lessons   or []

        try:
            from app.services.llm_router import get_fast_router, get_smart_router, LLMUnavailableError
            fast_router  = get_fast_router()
            smart_router = get_smart_router()

            context = self._build_context(
                symbol, signal, confidence, entry_price, stop_loss, target_price,
                debate_verdict, debate_reasoning, news_sentiment, news_summary,
                fund_grade, fund_score, open_positions, past_lessons,
            )

            # ── Round 1: Aggressive Debater ─────────────────────────────
            agg_sys = (
                "You are an aggressive equity portfolio manager. You believe in taking "
                "calculated risks for superior returns. Build the strongest case FOR "
                "executing this trade at the proposed size. Focus on upside potential, "
                "strong signals, and why the risk/reward is attractive. 120 words max."
            )
            agg_view = fast_router.complete(agg_sys, context, max_tokens=200)

            # ── Round 2: Conservative Debater (reads Aggressive) ────────
            con_sys = (
                "You are a conservative risk manager at an equity fund. Your primary duty "
                "is capital preservation. Respond to the aggressive manager's argument and "
                "highlight all downside risks, concentration risks, and why position sizing "
                "should be reduced or the trade avoided. 120 words max."
            )
            con_user = (
                f"{context}\n\n"
                f"--- Aggressive Manager's View ---\n{agg_view}"
            )
            con_view = fast_router.complete(con_sys, con_user, max_tokens=200)

            # ── Round 3: Neutral Debater (reads both) ───────────────────
            neu_sys = (
                "You are a neutral risk committee member. You have heard both the aggressive "
                "and conservative perspectives. Propose a balanced course of action: how much "
                "position size, any conditions, and what monitoring is needed. 120 words max."
            )
            neu_user = (
                f"{context}\n\n"
                f"--- Aggressive Manager ---\n{agg_view}\n\n"
                f"--- Conservative Manager ---\n{con_view}"
            )
            neu_view = fast_router.complete(neu_sys, neu_user, max_tokens=200)

            # ── Round 4: Portfolio Manager (final authority) ─────────────
            pm_sys = (
                "You are the Chief Portfolio Manager. You have heard the full risk committee "
                "debate. You have final authority. Deliver an authoritative PortfolioDecision "
                "in valid JSON. Consider ALL perspectives before deciding. "
                "Your decision directly controls trade execution — be precise."
            )

            rr_ratio = 0.0
            if entry_price and stop_loss and target_price and entry_price > 0:
                risk   = abs(entry_price - stop_loss)
                reward = abs(target_price - entry_price)
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

            pm_user = (
                f"{context}\n\n"
                f"Risk/Reward Ratio: {rr_ratio:.2f}:1\n\n"
                f"--- Aggressive Manager ---\n{agg_view}\n\n"
                f"--- Conservative Manager ---\n{con_view}\n\n"
                f"--- Neutral Member ---\n{neu_view}\n\n"
                f"Deliver your PortfolioDecision as JSON:\n"
                f'{{'
                f'"risk_verdict": "GREEN|AMBER|RED", '
                f'"executive_summary": "2-3 sentence executive summary", '
                f'"investment_thesis": "why this trade is worth taking (or not)", '
                f'"key_risks": "top 2 risks in one sentence", '
                f'"recommended_position_pct": <number 0-100>, '
                f'"price_target": <number or null>, '
                f'"stop_loss": <number or null>, '
                f'"time_horizon": "<string like 3-7 trading days>"'
                f'}}\n\n'
                f'GREEN = proceed with full/recommended size. '
                f'AMBER = proceed with reduced size (max 50%). '
                f'RED = do not execute this trade.'
            )

            pm_dict = smart_router.complete_json(
                pm_sys, pm_user,
                schema_hint=(
                    '{"risk_verdict":"str","executive_summary":"str",'
                    '"investment_thesis":"str","key_risks":"str",'
                    '"recommended_position_pct":number,"price_target":number,'
                    '"stop_loss":number,"time_horizon":"str"}'
                ),
                max_tokens=400
            )

            risk_verdict = pm_dict.get("risk_verdict", "AMBER").upper()
            if risk_verdict not in ("GREEN", "AMBER", "RED"):
                risk_verdict = "AMBER"

            rec_pct = float(pm_dict.get("recommended_position_pct", 50))
            rec_pct = max(0.0, min(100.0, rec_pct))

            # Cap recommended_position_pct based on verdict
            if risk_verdict == "RED":
                rec_pct = 0.0
            elif risk_verdict == "AMBER":
                rec_pct = min(rec_pct, 50.0)

            # Determine auto_execute
            auto_exec = False
            if auto_execute_mode == "auto":
                if auto_execute_threshold == "GREEN":
                    auto_exec = (risk_verdict == "GREEN")
                else:  # AMBER_GREEN
                    auto_exec = (risk_verdict in ("GREEN", "AMBER"))

            # Override: never auto-execute RED
            if risk_verdict == "RED":
                auto_exec = False

            return PortfolioDecision(
                risk_verdict=risk_verdict,
                executive_summary=pm_dict.get("executive_summary", ""),
                investment_thesis=pm_dict.get("investment_thesis", ""),
                key_risks=pm_dict.get("key_risks", ""),
                recommended_position_pct=rec_pct,
                price_target=pm_dict.get("price_target"),
                stop_loss=pm_dict.get("stop_loss"),
                time_horizon=pm_dict.get("time_horizon", "3-7 trading days"),
                aggressive_view=agg_view,
                conservative_view=con_view,
                neutral_view=neu_view,
                auto_execute=auto_exec,
                source="llm",
            )

        except Exception as e:
            logger.warning(f"Risk agent failed for {symbol}: {e}")
            return self._rule_based_fallback(
                signal, confidence, debate_verdict,
                news_sentiment, fund_grade, open_positions,
                auto_execute_mode, auto_execute_threshold,
            )

    # ── Context builder ────────────────────────────────────────────────
    def _build_context(
        self, symbol, signal, confidence, entry, sl, tp,
        debate_verdict, debate_reasoning,
        news_sentiment, news_summary,
        fund_grade, fund_score,
        open_positions, past_lessons,
    ) -> str:

        # Open positions context
        op_str = ""
        if open_positions:
            op_str = f"\n\nCurrently Open Positions ({len(open_positions)}):\n"
            for p in open_positions[:5]:
                op_str += f"  • {p.get('asset','?')} {p.get('direction','?')} — PnL: {p.get('pnl',0):.0f}\n"

        # Past lessons
        lessons_str = ""
        if past_lessons:
            lessons_str = "\n\nPast lessons from ASTRA's memory on this ticker:\n"
            for i, l in enumerate(past_lessons[:3], 1):
                lessons_str += f"  {i}. {l}\n"

        rr = 0.0
        if entry and sl and tp and entry > 0:
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            rr = round(reward / risk, 2) if risk > 0 else 0.0

        return (
            f"Symbol: {symbol}\n"
            f"ML Signal: {signal} | Confidence: {confidence:.1f}%\n"
            f"Entry: ₹{entry:.2f} | Stop Loss: ₹{sl:.2f} | Target: ₹{tp:.2f}\n"
            f"Risk/Reward: {rr:.2f}:1\n"
            f"Debate Verdict: {debate_verdict} — {debate_reasoning[:80]}\n"
            f"News: {news_sentiment} — {(news_summary or 'None')[:100]}\n"
            f"Fundamentals Grade: {fund_grade} ({fund_score:.0f}/100)"
            f"{op_str}"
            f"{lessons_str}"
        )

    # ── Rule-based fallback ────────────────────────────────────────────
    def _rule_based_fallback(
        self, signal, confidence, debate_verdict,
        news_sentiment, fund_grade, open_positions,
        auto_execute_mode, auto_execute_threshold,
    ) -> PortfolioDecision:

        red_flags = sum([
            debate_verdict == "REJECTED",
            news_sentiment == "NEGATIVE",
            fund_grade in ("D", "F"),
            confidence < 45,
            len(open_positions) >= 4,
        ])
        amber_flags = sum([
            debate_verdict == "DOWNGRADED",
            news_sentiment in ("MIXED", "NEGATIVE"),
            fund_grade == "C",
            confidence < 60,
        ])

        if red_flags >= 2 or debate_verdict == "REJECTED":
            risk_verdict = "RED"
            rec_pct      = 0.0
        elif red_flags >= 1 or amber_flags >= 2:
            risk_verdict = "AMBER"
            rec_pct      = 40.0
        else:
            risk_verdict = "GREEN"
            rec_pct      = 100.0

        auto_exec = False
        if auto_execute_mode == "auto" and risk_verdict != "RED":
            if auto_execute_threshold == "GREEN":
                auto_exec = risk_verdict == "GREEN"
            else:
                auto_exec = risk_verdict in ("GREEN", "AMBER")

        return PortfolioDecision(
            risk_verdict=risk_verdict,
            executive_summary=f"Rule-based assessment: {risk_verdict}. {red_flags} red flags detected.",
            investment_thesis=f"Signal: {signal} at {confidence:.0f}% confidence.",
            key_risks=f"News: {news_sentiment}. Fundamentals: {fund_grade}.",
            recommended_position_pct=rec_pct,
            price_target=None,
            stop_loss=None,
            time_horizon="3-7 trading days",
            aggressive_view="LLM unavailable.",
            conservative_view="LLM unavailable.",
            neutral_view="LLM unavailable.",
            auto_execute=auto_exec,
            source="rule_based",
        )


# ── Module singleton ────────────────────────────────────────────────────────
risk_portfolio_agent = RiskPortfolioAgent()
