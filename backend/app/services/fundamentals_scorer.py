"""
ASTRA Fundamentals Scorer
=========================
Tier 1 — Rules-based (no LLM needed).

Converts the financials dict (from /api/financials endpoint) into:
  - score:  0–100 composite health score
  - grade:  A (80+) / B (60-79) / C (40-59) / D (20-39) / F (<20)
  - breakdown: dict of individual component scores for display
  - gate_long: bool — True when fundamentals are too weak for a multi-day BUY

Used as a fast, free complement to the LLM agents.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class FundamentalsScore:
    def __init__(self, score: float, grade: str, breakdown: dict, gate_long: bool, message: str):
        self.score     = round(score, 1)
        self.grade     = grade
        self.breakdown = breakdown
        self.gate_long = gate_long   # suppress multi-day BUY when True
        self.message   = message

    def to_dict(self) -> dict:
        return {
            "score":     self.score,
            "grade":     self.grade,
            "breakdown": self.breakdown,
            "gate_long": self.gate_long,
            "message":   self.message,
        }


class FundamentalsScorer:

    def score(self, financials: dict) -> FundamentalsScore:
        """
        financials — the dict returned by /api/financials/{symbol}
                     expected sub-keys: keyFacts, growth
        """
        if not financials:
            return FundamentalsScore(50.0, "C", {}, False, "No fundamental data available.")

        kf = financials.get("keyFacts", {})
        gr = financials.get("growth",   {})

        breakdown = {}
        total_weight = 0
        total_score  = 0

        def _safe(v) -> Optional[float]:
            if v is None:
                return None
            try:
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else f
            except Exception:
                return None

        def _add(label: str, value: Optional[float], weight: float,
                 score_fn) -> None:
            nonlocal total_weight, total_score
            if value is None:
                breakdown[label] = {"value": None, "score": None, "weight": weight}
                return
            s = max(0.0, min(100.0, float(score_fn(value))))
            breakdown[label] = {"value": round(value, 4), "score": round(s, 1), "weight": weight}
            total_weight += weight
            total_score  += s * weight

        # ── Valuation ──────────────────────────────────────────────────
        # P/E ratio: below 15 = great (100), 15-25 = good (70), 25-40 = fair (40), >40 = poor (0)
        pe = _safe(kf.get("trailingPE"))
        _add("P/E Ratio", pe, 15, lambda v:
             100 if v < 0 else          # negative PE (loss) = 0
             (100 if v < 15 else
              (80  if v < 25 else
               (50  if v < 35 else
                (20  if v < 50 else 0)))))

        # P/B ratio: <1 = great (100), 1-3 = good (75), 3-6 = fair (40), >6 = poor (10)
        pb = _safe(kf.get("priceToBook"))
        _add("P/B Ratio", pb, 10, lambda v:
             100 if v < 1  else
             (75  if v < 3  else
              (40  if v < 6  else 10)))

        # ── Profitability ─────────────────────────────────────────────
        # ROE: >25% = excellent, 15-25 = good, 8-15 = fair, <8 = poor
        roe = _safe(kf.get("returnOnEquity"))
        _add("ROE %", roe, 20, lambda v:
             100 if v > 0.25 else
             (80  if v > 0.15 else
              (55  if v > 0.08 else
               (20  if v > 0    else 0))))

        # Profit margin: >20% = 100, 10-20 = 75, 5-10 = 50, 2-5 = 25, <2 = 0
        margin = _safe(gr.get("profitMargins"))
        _add("Profit Margin %", margin, 20, lambda v:
             100 if v > 0.20 else
             (75  if v > 0.10 else
              (50  if v > 0.05 else
               (25  if v > 0.02 else 0))))

        # Operating margin: >15% = 100
        op_margin = _safe(gr.get("operatingMargins"))
        _add("Operating Margin %", op_margin, 10, lambda v:
             100 if v > 0.15 else
             (75  if v > 0.10 else
              (50  if v > 0.05 else
               (20  if v > 0    else 0))))

        # ── Growth ────────────────────────────────────────────────────
        # Revenue growth: >20% = 100, 10-20 = 75, 0-10 = 50, <0 = 10
        rev_growth = _safe(gr.get("revenueGrowth"))
        _add("Revenue Growth %", rev_growth, 10, lambda v:
             100 if v > 0.20 else
             (75  if v > 0.10 else
              (50  if v > 0    else 10)))

        # Earnings growth: same scale
        earn_growth = _safe(gr.get("earningsGrowth"))
        _add("Earnings Growth %", earn_growth, 10, lambda v:
             100 if v > 0.25 else
             (75  if v > 0.10 else
              (50  if v > 0    else 10)))

        # ── Financial health ──────────────────────────────────────────
        # Current ratio: >2 = 100, 1-2 = 70, 0.5-1 = 30, <0.5 = 0
        cr = _safe(gr.get("currentRatio"))
        _add("Current Ratio", cr, 10, lambda v:
             100 if v > 2.0 else
             (70  if v > 1.0 else
              (30  if v > 0.5 else 0)))

        # Debt to equity: <0.5 = 100, 0.5-1 = 75, 1-2 = 50, 2-3 = 20, >3 = 0
        dte = _safe(gr.get("debtToEquity"))
        _add("Debt/Equity", dte, 10, lambda v:
             100 if v < 0.5  else
             (75  if v < 1.0  else
              (50  if v < 2.0  else
               (20  if v < 3.0  else 0))))

        # Beta: 0.8-1.2 = ideal (low systemic risk), outside = deducted
        beta = _safe(kf.get("beta"))
        _add("Beta", beta, 5, lambda v:
             100 if 0.8 <= v <= 1.2 else
             (70  if 0.5 <= v <= 1.5 else
              (40  if 0.2 <= v <= 2.0 else 10)))

        # ── Compute composite ─────────────────────────────────────────
        if total_weight > 0:
            raw = total_score / total_weight
        else:
            raw = 50.0  # insufficient data → neutral

        # Penalty for missing critical data (>60% fields missing → –10)
        filled = sum(1 for v in breakdown.values() if v["value"] is not None)
        if filled < len(breakdown) * 0.4:
            raw = max(0, raw - 10)

        score = round(raw, 1)

        if score >= 80: grade = "A"
        elif score >= 60: grade = "B"
        elif score >= 40: grade = "C"
        elif score >= 20: grade = "D"
        else: grade = "F"

        # Gate multi-day BUY if D or F
        gate_long = grade in ("D", "F")

        # Human-readable message
        if grade == "A":
            msg = "Fundamentally strong — excellent profitability, low debt, healthy growth."
        elif grade == "B":
            msg = "Solid fundamentals — good financial health with minor concerns."
        elif grade == "C":
            msg = "Average fundamentals — some weaknesses present; proceed with caution."
        elif grade == "D":
            msg = "Weak fundamentals — high risk; consider shorter holding period."
        else:
            msg = "Poor fundamentals — significant financial stress detected."

        return FundamentalsScore(
            score=score, grade=grade,
            breakdown=breakdown, gate_long=gate_long,
            message=msg,
        )


# ── Module singleton ────────────────────────────────────────────────────────
fundamentals_scorer = FundamentalsScorer()
