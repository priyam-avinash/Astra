"""
ASTRA Universe Scanner
======================
Scans the full NSE equity universe (NIFTY 500 + additional stocks)
for trading signals using ASTRA.AI (fastest) as primary screener,
then confirms top candidates with ASTRA.ML.

Usage:
    from app.services.universe_scanner import universe_scanner
    results = universe_scanner.scan(engines=["astra_ai"], top_k=10)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

logger = logging.getLogger(__name__)

# ── NIFTY 500 Universe ────────────────────────────────────────────────────────

# NIFTY 50
_NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BAJFINANCE", "KOTAKBANK", "BHARTIARTL", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ADANIENT", "POWERGRID", "TECHM", "INDUSINDBK", "DIVISLAB", "JSWSTEEL",
    "BAJAJFINSV", "COALINDIA", "HCLTECH", "ONGC", "NTPC", "M&M", "TATAMOTORS",
    "TATASTEEL", "CIPLA", "BRITANNIA", "DRREDDY", "BPCL", "HEROMOTOCO",
    "GRASIM", "EICHERMOT", "TATACONSUM", "SBILIFE", "HDFCLIFE", "UPL",
    "APOLLOHOSP", "HINDALCO", "ADANIPORTS", "BAJAJ-AUTO", "VEDL", "ITC",
]

# NIFTY NEXT 50
_NIFTY_NEXT_50 = [
    "SIEMENS", "ABB", "PIDILITIND", "SRF", "HAVELLS", "MUTHOOTFIN", "BERGEPAINT",
    "COLPAL", "DABUR", "MARICO", "GODREJCP", "TORNTPHARM", "BOSCHLTD",
    "MCDOWELL-N", "RECLTD", "PFC", "IRCTC", "CHOLAFIN", "TATAPOWER",
    "ADANIGREEN", "ADANITRANS", "DMART", "NAUKRI", "INDIGO", "BANDHANBNK",
    "IDFCFIRSTB", "FEDERALBNK", "CANBK", "PNBHOUSING", "PNB", "UNIONBANK",
    "BANKBARODA", "GAIL", "IOC", "PETRONET", "CONCOR", "NHPC", "SJVN",
    "TRENTLTD", "AUROPHARMA", "LUPIN", "BIOCON", "CADILAHC", "ALKEM",
    "TORNTPOWER", "TATACOMM", "MPHASIS", "LTIM", "PERSISTENT",
]

# MIDCAP additions
_MIDCAP = [
    "NYKAA", "PAYTM", "ZOMATO", "POLICYBZR", "IRFC", "RVNL", "HUDCO",
    "MOTHERSON", "ASHOKLEY", "ESCORTS", "SONACOMS", "KPITTECH", "TIINDIA",
    "LALPATHLAB", "METROPOLIS", "AAVAS", "HOMEFIRST", "SBFC", "FIVESTAR",
    "CREDITACC", "JMFINANCIL", "CHOICEIN", "ANGELONE", "CDSL", "BSE", "MCX",
    "MSCI", "NAVIN", "DEEPAKNTR", "INDUSTOWER", "FLUOROCHEM", "ALKYLAMINE",
    "GALAXYSURF", "GRINDWELL", "KAJARIACER", "CERA", "SOMANY", "ORIENTELEC",
    "DIXON", "AMBER", "WHIRLPOOL", "VOLTAS",
]

# Full universe with .NS suffix
NIFTY_500 = [s + ".NS" for s in (_NIFTY_50 + _NIFTY_NEXT_50 + _MIDCAP)]

# ── Sector map (top 50 stocks) ────────────────────────────────────────────────

_SECTOR_MAP = {
    "RELIANCE.NS":    "Energy",
    "TCS.NS":         "IT",
    "HDFCBANK.NS":    "Banking",
    "INFY.NS":        "IT",
    "ICICIBANK.NS":   "Banking",
    "HINDUNILVR.NS":  "FMCG",
    "SBIN.NS":        "Banking",
    "BAJFINANCE.NS":  "NBFC",
    "KOTAKBANK.NS":   "Banking",
    "BHARTIARTL.NS":  "Telecom",
    "LT.NS":          "Engineering",
    "AXISBANK.NS":    "Banking",
    "ASIANPAINT.NS":  "Paints",
    "MARUTI.NS":      "Auto",
    "SUNPHARMA.NS":   "Pharma",
    "TITAN.NS":       "Consumer",
    "ULTRACEMCO.NS":  "Cement",
    "WIPRO.NS":       "IT",
    "NESTLEIND.NS":   "FMCG",
    "ADANIENT.NS":    "Conglomerate",
    "POWERGRID.NS":   "Power",
    "TECHM.NS":       "IT",
    "INDUSINDBK.NS":  "Banking",
    "DIVISLAB.NS":    "Pharma",
    "JSWSTEEL.NS":    "Steel",
    "BAJAJFINSV.NS":  "NBFC",
    "COALINDIA.NS":   "Mining",
    "HCLTECH.NS":     "IT",
    "ONGC.NS":        "Energy",
    "NTPC.NS":        "Power",
    "M&M.NS":         "Auto",
    "TATAMOTORS.NS":  "Auto",
    "TATASTEEL.NS":   "Steel",
    "CIPLA.NS":       "Pharma",
    "BRITANNIA.NS":   "FMCG",
    "DRREDDY.NS":     "Pharma",
    "BPCL.NS":        "Energy",
    "HEROMOTOCO.NS":  "Auto",
    "GRASIM.NS":      "Cement",
    "EICHERMOT.NS":   "Auto",
    "TATACONSUM.NS":  "FMCG",
    "SBILIFE.NS":     "Insurance",
    "HDFCLIFE.NS":    "Insurance",
    "UPL.NS":         "Agrochem",
    "APOLLOHOSP.NS":  "Healthcare",
    "HINDALCO.NS":    "Metals",
    "ADANIPORTS.NS":  "Infrastructure",
    "BAJAJ-AUTO.NS":  "Auto",
    "VEDL.NS":        "Metals",
    "ITC.NS":         "FMCG",
    # NIFTY NEXT 50
    "SIEMENS.NS":     "Engineering",
    "HAVELLS.NS":     "Engineering",
    "VOLTAS.NS":      "Engineering",
    "PIDILITIND.NS":  "Consumer",
    "BERGEPAINT.NS":  "Paints",
    "GODREJCP.NS":    "Consumer",
    "MUTHOOTFIN.NS":  "NBFC",
    "COLPAL.NS":      "FMCG",
    "DABUR.NS":       "FMCG",
    "MARICO.NS":      "FMCG",
    "TORNTPHARM.NS":  "Pharma",
    "LUPIN.NS":       "Pharma",
    "BIOCON.NS":      "Pharma",
    "BOSCHLTD.NS":    "Auto",
    "MOTHERSON.NS":   "Auto",
    "BALKRISIND.NS":  "Auto",
    "MCDOWELL-N.NS":  "Consumer",
    "PAGEIND.NS":     "Consumer",
    "IRCTC.NS":       "Consumer",
    "ADANIGREEN.NS":  "Energy",
    "ADANITRANS.NS":  "Energy",
    "ADANIPOWER.NS":  "Energy",
    "TATAPOWER.NS":   "Power",
    "RECLTD.NS":      "NBFC",
    "PFC.NS":         "NBFC",
    "CHOLAFIN.NS":    "NBFC",
    "IDFCFIRSTB.NS":  "Banking",
    "FEDERALBNK.NS":  "Banking",
    "BANDHANBNK.NS":  "Banking",
    "CANBK.NS":       "Banking",
    "PNB.NS":         "Banking",
    "UNIONBANK.NS":   "Banking",
    "BANKBARODA.NS":  "Banking",
    "GAIL.NS":        "Energy",
    "IOC.NS":         "Energy",
    "PETRONET.NS":    "Energy",
    "TATACOMM.NS":    "Telecom",
    "INDUSTOWER.NS":  "Telecom",
    "MPHASIS.NS":     "IT",
    "LTIM.NS":        "IT",
    "PERSISTENT.NS":  "IT",
    "TATAELXSI.NS":   "IT",
    "LTTS.NS":        "IT",
    "COFORGE.NS":     "IT",
    "INDIAMART.NS":   "IT",
    # MIDCAP
    "NYKAA.NS":       "Consumer",
    "ZOMATO.NS":      "Consumer",
    "PAYTM.NS":       "NBFC",
    "POLICYBZR.NS":   "Insurance",
    "DELHIVERY.NS":   "Logistics",
    "ANGELONE.NS":    "NBFC",
    "CDSL.NS":        "NBFC",
    "BSE.NS":         "NBFC",
    "MCX.NS":         "NBFC",
    "DIXON.NS":       "Engineering",
    "AMBER.NS":       "Engineering",
}


class UniverseScanner:
    """
    Scans the NSE equity universe for actionable signals using
    ASTRA.AI as primary screener and ASTRA.ML for confirmation.
    """

    def __init__(self):
        # Import lazily to avoid circular dependency at module load time
        from app.services.ai_predictor import ai_engine as _ai_engine
        self.ai_engine = _ai_engine
        self.symbols = NIFTY_500
        logger.info(
            f"UniverseScanner initialized — {len(self.symbols)} symbols in universe"
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _scan_one(self, symbol: str, engine_name: str) -> dict | None:
        """
        Analyze a single symbol. Returns result dict or None on failure.
        Includes a small sleep to avoid hammering yfinance.
        """
        try:
            time.sleep(0.1)
            result = self.ai_engine.analyze_market_data(symbol, engine=engine_name)
            if result.get("signal") in ("BUY", "SELL"):
                return {
                    "symbol":       symbol,
                    "signal":       result.get("signal"),
                    "confidence":   result.get("confidence", 0.0),
                    "entry_price":  result.get("entry_price", 0.0),
                    "target":       result.get("target", 0.0),
                    "stop_loss":    result.get("stop_loss", 0.0),
                    "engine":       engine_name,
                    "weekly_trend": result.get("weekly_trend", "NEUTRAL"),
                    "sector":       self.get_sector(symbol),
                }
        except Exception as exc:
            logger.debug(f"[{engine_name}] {symbol} failed: {exc}")
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        engines: list = None,
        top_k: int = 20,
        min_confidence: float = 55.0,
        max_workers: int = 8,
    ) -> list:
        """
        Parallel scan of the full universe using the specified engine(s).

        Parameters
        ----------
        engines : list of str, default ["astra_ai"]
            One or more of "astra", "astra_ai", "astra_ml".
        top_k : int
            Maximum number of results to return (sorted by confidence desc).
        min_confidence : float
            Minimum confidence threshold to include a signal.
        max_workers : int
            Thread pool size. Keep <= 8 to avoid yfinance rate limits.

        Returns
        -------
        list[dict]
            Each dict has keys: symbol, signal, confidence, entry_price,
            target, stop_loss, engine, weekly_trend, sector.
            The list also carries a ``scan_duration_sec`` attribute on the
            first element (stored as a regular key for JSON-serialisability).
        """
        if engines is None:
            engines = ["astra_ai"]

        start_ts = time.time()
        signals: list[dict] = []

        for engine_name in engines:
            logger.info(
                f"UniverseScanner: starting {engine_name} scan "
                f"({len(self.symbols)} symbols, {max_workers} workers)"
            )
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for sym in self.symbols:
                    fut = pool.submit(self._scan_one, sym, engine_name)
                    futures[fut] = sym

                for fut in as_completed(futures):
                    result = fut.result()
                    if result is not None and result["confidence"] >= min_confidence:
                        signals.append(result)

        # Sort by confidence descending, take top_k
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        top_signals = signals[:top_k]

        duration = round(time.time() - start_ts, 1)
        logger.info(
            f"UniverseScanner: scan complete — "
            f"{len(top_signals)} actionable signals in {duration}s"
        )

        # Attach metadata as a plain key on each result (JSON-friendly)
        for r in top_signals:
            r["scan_duration_sec"] = duration

        return top_signals

    def full_scan(self, top_k: int = 15) -> dict:
        """
        Two-stage scan:
          Stage 1 — ASTRA.AI across full universe (fast), take top 30 candidates.
          Stage 2 — ASTRA.ML re-runs on those 30 for confirmation.
          Returns merged results with consensus signal.

        Returns
        -------
        dict with keys:
          "astra_ai_results" : list of raw AI signals
          "astra_ml_results" : list of ML confirmation signals
          "consensus"        : list of stocks where both engines agree
          "top_k"            : consensus list truncated to top_k
        """
        logger.info("UniverseScanner: starting full_scan (AI → ML pipeline)")

        # Stage 1: fast ASTRA.AI scan, get top 30
        ai_results = self.scan(engines=["astra_ai"], top_k=30, min_confidence=50.0)
        candidate_symbols = [r["symbol"] for r in ai_results]
        logger.info(
            f"UniverseScanner: Stage 1 done — "
            f"{len(candidate_symbols)} candidates for ML confirmation"
        )

        # Stage 2: ASTRA.ML on the 30 candidates
        ml_results = []
        for sym in candidate_symbols:
            result = self._scan_one(sym, "astra_ml")
            if result is not None:
                ml_results.append(result)

        # Build lookup maps
        ai_map = {r["symbol"]: r for r in ai_results}
        ml_map = {r["symbol"]: r for r in ml_results}

        # Consensus: both engines agree on direction
        consensus = []
        for sym in candidate_symbols:
            ai_sig = ai_map.get(sym, {}).get("signal")
            ml_sig = ml_map.get(sym, {}).get("signal")
            if ai_sig and ml_sig and ai_sig == ml_sig:
                avg_conf = round(
                    (ai_map[sym]["confidence"] + ml_map[sym]["confidence"]) / 2, 1
                )
                consensus.append({
                    "symbol":             sym,
                    "astra_ai_signal":    ai_sig,
                    "astra_ml_signal":    ml_sig,
                    "consensus":          ai_sig,
                    "avg_confidence":     avg_conf,
                    "entry_price":        ai_map[sym]["entry_price"],
                    "target":             ai_map[sym]["target"],
                    "stop_loss":          ai_map[sym]["stop_loss"],
                    "weekly_trend":       ai_map[sym]["weekly_trend"],
                    "sector":             ai_map[sym]["sector"],
                })

        consensus.sort(key=lambda x: x["avg_confidence"], reverse=True)

        return {
            "astra_ai_results": ai_results,
            "astra_ml_results": ml_results,
            "consensus":        consensus,
            "top_k":            consensus[:top_k],
        }

    def get_sector(self, symbol: str) -> str:
        """Return sector for a symbol, or 'Other' if not mapped."""
        return _SECTOR_MAP.get(symbol, "Other")

    def scan_sector(self, sector: str, **kwargs) -> list:
        """
        Filter universe to a single sector, then run scan().
        Thread-safe: builds a local symbol list instead of mutating self.symbols.

        Parameters
        ----------
        sector : str  e.g. "Banking", "IT", "Pharma"
        **kwargs      forwarded to scan()
        """
        # Build local list — never mutates self.symbols, safe under concurrent calls
        symbols_to_scan = [s for s in self.symbols if self.get_sector(s) == sector]
        if not symbols_to_scan:
            logger.warning(f"UniverseScanner: no symbols found for sector '{sector}'")
            return []

        logger.info(
            f"UniverseScanner: scanning {len(symbols_to_scan)} "
            f"symbols in sector '{sector}'"
        )

        engines        = kwargs.pop("engines",        ["astra_ai"])
        top_k          = kwargs.pop("top_k",          20)
        min_confidence = kwargs.pop("min_confidence", 55.0)
        max_workers    = kwargs.pop("max_workers",    4)   # conservative for sector scans

        start_ts = time.time()
        signals: list[dict] = []

        for engine_name in engines:
            futures: dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for sym in symbols_to_scan:
                    fut = pool.submit(self._scan_one, sym, engine_name)
                    futures[fut] = sym
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is not None and result["confidence"] >= min_confidence:
                        signals.append(result)

        signals.sort(key=lambda x: x["confidence"], reverse=True)
        top_signals = signals[:top_k]
        duration = round(time.time() - start_ts, 1)
        for r in top_signals:
            r["scan_duration_sec"] = duration
        return top_signals


# ── Module-level singleton ────────────────────────────────────────────────────

universe_scanner = UniverseScanner()
