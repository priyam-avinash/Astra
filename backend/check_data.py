#!/usr/bin/env python3
"""
ASTRA data-provider check — run this first when "no data" shows up.

    cd backend && source .venv/bin/activate
    python check_data.py                    # tests every provider
    python check_data.py TCS ETH-USD        # custom symbols

Prints, per provider: OK + bars/last price, or the exact failure reason
(bad key, quota exhausted, blocked network, plan restriction…). Nothing is
traded; no keys are printed.
"""
import logging
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import ENV_FILES_LOADED, get_secret, secret_source  # noqa: E402
from app.services.market_data import HAVE_CURL_CFFI, market_data  # noqa: E402

G, R, Y, D, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def main():
    symbols = sys.argv[1:] or ["RELIANCE.NS", "^NSEI", "BTC-USD"]
    print(f"\n{B}ASTRA data check{X}  (paper trading only)\n")
    print(f"  .env files loaded : {', '.join(ENV_FILES_LOADED) or R + 'none found' + X}")
    print(f"  curl_cffi         : {G + 'installed' + X if HAVE_CURL_CFFI else R + 'MISSING — pip install curl_cffi (Yahoo will 429)' + X}")
    for key in ("twelve_data_key", "alpha_vantage_key", "dhan_client_id", "dhan_access_token",
                "anthropic_api_key", "groq_api_key"):
        src = secret_source(key)
        mark = f"{G}set ({src}){X}" if get_secret(key) else f"{D}not set{X}"
        print(f"  {key:<18}: {mark}")

    print(f"\n{B}Provider probe{X}  symbols: {', '.join(symbols)}\n")
    t0 = time.time()
    results = market_data.probe(symbols)
    for key, r in results.items():
        name, sym = key.split(":", 1)
        if r.get("ok"):
            print(f"  {G}✓{X} {name:<14} {sym:<13} {r['bars']:>4} bars  last {r['last_close']} ({r['last_date']})  {D}{r['secs']}s{X}")
        elif r.get("detail") == "not configured":
            print(f"  {D}-{X} {name:<14} {sym:<13} {D}not configured (optional){X}")
        else:
            colour = Y if r.get("kind") in ("unsupported", "no_data") else R
            print(f"  {colour}✗{X} {name:<14} {sym:<13} {colour}[{r.get('kind')}] {r.get('detail')}{X}")

    print(f"\n{B}End-to-end{X}")
    for sym in symbols:
        df = market_data.get_ohlcv(sym, period="1y", interval="1d")
        q = market_data.get_quote_info(sym)
        if df.empty:
            print(f"  {R}✗ {sym}: no daily data from any provider{X}")
        else:
            print(f"  {G}✓ {sym}{X}: {len(df)} daily bars via {df.attrs.get('source')}; "
                  f"quote {q['price']} via {q['source']}")
    intraday = market_data.get_ohlcv("RELIANCE.NS", period="5d", interval="15m")
    print(f"  {'✓' if not intraday.empty else '✗'} RELIANCE 15m: {len(intraday)} bars"
          f"{' via ' + str(intraday.attrs.get('source')) if not intraday.empty else ''}")
    print(f"\n{D}done in {time.time() - t0:.1f}s{X}\n")


if __name__ == "__main__":
    main()
