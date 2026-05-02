"""
ASTRA News Analyst Agent
========================
Tier 1 — Signal Intelligence.

Fetches recent news for a symbol from Alpha Vantage News Sentiment API,
then uses the LLM Router to produce:
  - sentiment:    POSITIVE / NEGATIVE / NEUTRAL / MIXED
  - summary:      2-3 sentence human-readable summary
  - key_events:   list of up to 3 bullet points
  - should_gate:  bool — True when negative news should suppress a BUY signal

Falls back to keyword-based sentiment if LLM is unavailable.
Results are cached per (symbol, date) in memory to avoid redundant API calls.
"""

import logging
import os
import re
import requests
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── In-process daily cache ──────────────────────────────────────────────────
_NEWS_CACHE: dict = {}   # key = "SYMBOL-YYYY-MM-DD"


NEGATIVE_KEYWORDS = [
    "fraud", "scam", "scandal", "investigation", "probe", "lawsuit", "default",
    "bankruptcy", "insolvency", "downgrade", "miss", "loss", "decline", "crash",
    "penalty", "fine", "ban", "suspension", "recall", "accident", "death",
    "layoff", "retrench", "resign", "arrested", "raid", "seized", "block",
    "regulatory action", "sebi", "ed probe", "cbi", "nclt", "delisted",
]

POSITIVE_KEYWORDS = [
    "profit", "growth", "record", "upgrade", "beat", "acquisition", "merger",
    "expansion", "contract", "deal", "order", "launch", "approval", "dividend",
    "buyback", "partnership", "collaboration", "award", "win", "outperform",
]


class NewsAnalysis:
    def __init__(
        self,
        sentiment: str,
        summary: str,
        key_events: list,
        should_gate: bool,
        source: str = "llm",
        raw_headlines: list = None,
    ):
        self.sentiment    = sentiment      # POSITIVE / NEGATIVE / NEUTRAL / MIXED
        self.summary      = summary
        self.key_events   = key_events     # list[str]
        self.should_gate  = should_gate    # True = suppress BUY below confidence threshold
        self.source       = source         # "llm" or "keyword"
        self.raw_headlines = raw_headlines or []

    def to_dict(self) -> dict:
        return {
            "sentiment":    self.sentiment,
            "summary":      self.summary,
            "key_events":   self.key_events,
            "should_gate":  self.should_gate,
            "source":       self.source,
        }


class NewsAnalystAgent:

    AV_BASE = "https://www.alphavantage.co/query"

    def __init__(self):
        self.av_key = (
            os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHA_VANTAGE_KEY", "XV1FMHS5UHPIIPAZ")
        )

    # ── Public entry point ────────────────────────────────────────────
    def analyze(
        self,
        symbol: str,
        signal: str = "HOLD",
        confidence: float = 50.0,
    ) -> NewsAnalysis:
        """
        Fetch and analyse news for `symbol`.

        signal      — current ML signal (BUY/SELL/HOLD) — used for gating logic
        confidence  — ML confidence % — gating only suppresses below 60%
        """
        cache_key = f"{symbol}-{date.today().isoformat()}"
        if cache_key in _NEWS_CACHE:
            logger.debug(f"News cache hit for {symbol}")
            return _NEWS_CACHE[cache_key]

        headlines = self._fetch_news(symbol)
        if not headlines:
            result = NewsAnalysis(
                sentiment="NEUTRAL",
                summary="No recent news found for this symbol.",
                key_events=[],
                should_gate=False,
                source="no_data",
            )
            _NEWS_CACHE[cache_key] = result
            return result

        # Try LLM analysis first
        result = self._llm_analyze(symbol, headlines, signal, confidence)
        if result is None:
            # Fall back to keyword-based
            result = self._keyword_analyze(symbol, headlines, signal, confidence)

        _NEWS_CACHE[cache_key] = result
        return result

    # ── Alpha Vantage news fetch ──────────────────────────────────────
    def _fetch_news(self, symbol: str) -> list[str]:
        """Returns list of headline strings (up to 15 most recent)."""
        try:
            bare = symbol.replace(".NS", "").replace(".BSE", "").replace(".BO", "")
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers":  bare,
                "apikey":   self.av_key,
                "limit":    "15",
                "sort":     "LATEST",
            }
            resp = requests.get(self.AV_BASE, params=params, timeout=10)
            data = resp.json()

            if "feed" not in data:
                # Try global macro news as fallback
                params2 = {"function": "NEWS_SENTIMENT", "topics": "economy_macro",
                           "apikey": self.av_key, "limit": "5"}
                resp2 = requests.get(self.AV_BASE, params=params2, timeout=8)
                data2 = resp2.json()
                feed = data2.get("feed", [])
            else:
                feed = data["feed"]

            headlines = []
            for item in feed[:15]:
                title = item.get("title", "")
                summary = item.get("summary", "")
                ticker_sentiment = ""
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() in [bare.upper(), symbol.upper()]:
                        score = float(ts.get("ticker_sentiment_score", 0))
                        label = ts.get("ticker_sentiment_label", "")
                        ticker_sentiment = f" [Score: {score:+.2f}, {label}]"
                headlines.append(f"{title}{ticker_sentiment}")

            return headlines
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return []

    # ── LLM-based analysis ────────────────────────────────────────────
    def _llm_analyze(
        self,
        symbol: str,
        headlines: list[str],
        signal: str,
        confidence: float,
    ) -> Optional[NewsAnalysis]:
        try:
            from app.services.llm_router import get_fast_router, LLMUnavailableError
            router = get_fast_router()

            bare = symbol.replace(".NS", "").replace(".BO", "")
            news_block = "\n".join(f"• {h}" for h in headlines[:10])

            system = (
                f"You are a professional financial news analyst specialising in Indian equities. "
                f"Your task: analyse recent news headlines for {bare} and determine their impact "
                f"on a potential trading signal. Be concise, factual, and trading-focused."
            )
            user = (
                f"Symbol: {bare} | Current ML Signal: {signal} | Confidence: {confidence:.1f}%\n\n"
                f"Recent headlines:\n{news_block}\n\n"
                f"Respond in JSON with exactly these fields:\n"
                f'{{"sentiment": "POSITIVE|NEGATIVE|NEUTRAL|MIXED", '
                f'"summary": "2-3 sentence summary of key news impact", '
                f'"key_events": ["event 1", "event 2", "event 3"], '
                f'"should_gate": true/false, '
                f'"gate_reason": "reason if gating, else empty string"}}\n\n'
                f'Set should_gate=true ONLY if sentiment is NEGATIVE AND there is material '
                f'company-specific bad news (fraud, earnings miss, regulatory action, default). '
                f'Macro/sector headwinds alone should NOT gate a signal.'
            )

            result_dict = router.complete_json(system, user,
                schema_hint='{"sentiment":"str","summary":"str","key_events":["str"],"should_gate":bool,"gate_reason":"str"}')

            if not result_dict or "sentiment" not in result_dict:
                return None

            sentiment   = result_dict.get("sentiment", "NEUTRAL")
            summary     = result_dict.get("summary", "")
            key_events  = result_dict.get("key_events", [])[:3]
            should_gate = bool(result_dict.get("should_gate", False))
            # Only gate BUY signals, and only when confidence is below threshold
            if signal != "BUY" or confidence >= 65.0:
                should_gate = False

            return NewsAnalysis(
                sentiment=sentiment,
                summary=summary,
                key_events=key_events,
                should_gate=should_gate,
                source="llm",
                raw_headlines=headlines,
            )

        except Exception as e:
            logger.warning(f"LLM news analysis failed for {symbol}: {e}")
            return None

    # ── Rule-based keyword fallback ───────────────────────────────────
    def _keyword_analyze(
        self,
        symbol: str,
        headlines: list[str],
        signal: str,
        confidence: float,
    ) -> NewsAnalysis:
        combined = " ".join(headlines).lower()

        neg_hits = [kw for kw in NEGATIVE_KEYWORDS if kw in combined]
        pos_hits = [kw for kw in POSITIVE_KEYWORDS if kw in combined]

        if len(neg_hits) > len(pos_hits) + 1:
            sentiment = "NEGATIVE"
        elif len(pos_hits) > len(neg_hits) + 1:
            sentiment = "POSITIVE"
        elif neg_hits and pos_hits:
            sentiment = "MIXED"
        else:
            sentiment = "NEUTRAL"

        # Build key events from first 3 headlines
        key_events = [h.split("[Score")[0].strip() for h in headlines[:3]]

        strong_neg = any(kw in combined for kw in [
            "fraud", "scam", "investigation", "default", "bankruptcy",
            "arrested", "raid", "sebi", "delisted",
        ])
        should_gate = (
            signal == "BUY"
            and confidence < 65.0
            and sentiment == "NEGATIVE"
            and strong_neg
        )

        summary = (
            f"Keyword analysis found {len(pos_hits)} positive and {len(neg_hits)} negative "
            f"signals in recent news. Overall sentiment: {sentiment}."
        )

        return NewsAnalysis(
            sentiment=sentiment,
            summary=summary,
            key_events=key_events,
            should_gate=should_gate,
            source="keyword",
            raw_headlines=headlines,
        )


# ── Module singleton ────────────────────────────────────────────────────────
news_agent = NewsAnalystAgent()
