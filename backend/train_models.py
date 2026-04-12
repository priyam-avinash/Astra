"""
ASTRA Model Training Pipeline v3.0
====================================
Trains 4 separate models:
  1. astra_rf.joblib          — Random Forest regressor for equities (ASTRA.AI)
  2. astra_lstm_equity.keras  — Bidirectional LSTM for equities (ASTRA.ML)
  3. astra_lstm_crypto.keras  — Bidirectional LSTM for crypto (ASTRA.CRYPTO.ML)
  4. astra_lstm_equity_scaler.joblib / astra_lstm_crypto_scaler.joblib

Run from backend/ directory:
  source .venv/bin/activate
  python train_models.py
"""
import os
import sys
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Training symbols — broad NIFTY 50 sample for robustness, not overfitting to one stock
EQUITY_TRAIN_SYMBOLS = [
    "^NSEI",       # NIFTY 50 Index — best primary
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]

CRYPTO_TRAIN_SYMBOLS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
]


def av_fallback_download(symbol: str) -> "pd.DataFrame":
    """
    Direct Alpha Vantage download for training — uses full outputsize (20 years).
    Used as fallback when yfinance is rate-limited or returns empty data.
    """
    import pandas as pd
    import requests
    import os
    av_key = os.getenv("ALPHA_VANTAGE_KEY", "XV1FMHS5UHPIIPAZ")
    # Map .NS symbols to AV format
    av_sym = symbol.replace(".NS", ".BSE").replace("^NSEI", "NSEI.BSE")
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={av_sym}&apikey={av_key}&outputsize=full"
    )
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        if "Time Series (Daily)" in data:
            ts = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts, orient="index")
            df = df.rename(columns={
                "1. open": "Open", "2. high": "High",
                "3. low": "Low",  "4. close": "Close", "5. volume": "Volume"
            })
            df.index = pd.to_datetime(df.index)
            df = df.astype(float).sort_index()
            logger.info(f"  AlphaVantage fallback: {len(df)} bars for {symbol}")
            return df
        # AV returned an error/limit message
        info = data.get("Information", data.get("Note", "Unknown AV error"))
        logger.warning(f"  AlphaVantage limit/error for {symbol}: {info[:80]}")
    except Exception as e:
        logger.warning(f"  AlphaVantage fallback failed for {symbol}: {e}")
    return pd.DataFrame()


def aggregate_equity_data(ai_engine, symbols: list, period: str = "5y", interval: str = "1d"):
    """Fetch and combine training data from multiple equity symbols."""
    import pandas as pd
    all_dfs = []
    for sym in symbols:
        logger.info(f"Fetching equity data: {sym}")
        df = ai_engine._fetch_data(sym, period=period, interval=interval)
        if df is not None and not df.empty and len(df) > 60:
            all_dfs.append(df)
            logger.info(f"  → {len(df)} bars for {sym} (via primary source)")
        else:
            # Try Alpha Vantage direct as last resort
            logger.info(f"  Primary failed — trying Alpha Vantage direct for {sym}")
            df_av = av_fallback_download(sym)
            if not df_av.empty and len(df_av) > 60:
                all_dfs.append(df_av)
                logger.info(f"  → {len(df_av)} bars for {sym} (via AV fallback)")
            else:
                logger.warning(f"  → Skipped {sym} (insufficient data from all sources)")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=False)


def train_rf(ai_engine, df):
    """Train and save Random Forest model."""
    logger.info("=" * 50)
    logger.info("Training ASTRA.AI (RandomForestRegressor) — 20-feature equity model")
    logger.info("=" * 50)
    model = ai_engine.train_model_rf(df)
    if model:
        path = os.path.join(MODELS_DIR, "astra_rf.joblib")
        joblib.dump(model, path)
        logger.info(f"✅ Saved Random Forest → {path}")
        return True
    else:
        logger.error("❌ RF training failed")
        return False


def train_lstm_equity(ai_engine, df):
    """Train and save Bidirectional LSTM equity model."""
    logger.info("=" * 50)
    logger.info("Training ASTRA.ML Equity (Bidirectional LSTM + Attention)")
    logger.info("=" * 50)
    model, scaler = ai_engine.train_model_lstm(df, lookback=30)
    if model and scaler:
        model_path = os.path.join(MODELS_DIR, "astra_lstm_equity.keras")
        scaler_path = os.path.join(MODELS_DIR, "astra_lstm_equity_scaler.joblib")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ Saved Equity LSTM → {model_path}")
        logger.info(f"✅ Saved Equity Scaler → {scaler_path}")
        return True
    else:
        logger.error("❌ Equity LSTM training failed (TensorFlow may not be installed)")
        return False


def train_lstm_crypto(crypto_engine, symbols: list):
    """Train and save Bidirectional LSTM crypto model."""
    logger.info("=" * 50)
    logger.info("Training ASTRA.CRYPTO.ML (Bidirectional LSTM — multi-coin)")
    logger.info("=" * 50)
    model, scaler = crypto_engine.train_lstm_crypto(symbols=symbols, lookback=30)
    if model and scaler:
        model_path = os.path.join(MODELS_DIR, "astra_lstm_crypto.keras")
        scaler_path = os.path.join(MODELS_DIR, "astra_lstm_crypto_scaler.joblib")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ Saved Crypto LSTM → {model_path}")
        logger.info(f"✅ Saved Crypto Scaler → {scaler_path}")
        return True
    else:
        logger.error("❌ Crypto LSTM training failed (TensorFlow may not be installed)")
        return False


def main():
    import pandas as pd

    logger.info("🚀 ASTRA Training Pipeline v3.0 Starting...")
    logger.info(f"   Models directory: {MODELS_DIR}")

    # Import engines
    try:
        from app.services.ai_predictor import ai_engine
        from app.services.crypto_engine import crypto_engine
    except ImportError as e:
        logger.error(f"Failed to import engines: {e}")
        sys.exit(1)

    results = {}

    # ── 1. Fetch combined equity data ──
    logger.info("\n📥 Fetching equity training data (multi-symbol)...")
    equity_df = aggregate_equity_data(ai_engine, EQUITY_TRAIN_SYMBOLS, period="5y", interval="1d")
    if equity_df.empty:
        logger.warning("⚠️  No equity data fetched. Trying with reduced symbol list...")
        equity_df = aggregate_equity_data(ai_engine, ["^NSEI"], period="2y", interval="1d")

    if not equity_df.empty:
        logger.info(f"   Total equity rows: {len(equity_df)}")

        # ── 2. Train Random Forest ──
        results["rf"] = train_rf(ai_engine, equity_df)

        # ── 3. Train Equity LSTM ──
        results["lstm_equity"] = train_lstm_equity(ai_engine, equity_df)
    else:
        logger.error("❌ Cannot train equity models — no data available")
        results["rf"] = False
        results["lstm_equity"] = False

    # ── 4. Train Crypto LSTM ──
    results["lstm_crypto"] = train_lstm_crypto(crypto_engine, CRYPTO_TRAIN_SYMBOLS)

    # ── Summary ──
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {name:20s}: {status}")

    any_failed = not all(results.values())
    if any_failed:
        logger.warning("\n⚠️  Some models failed. Check logs above.")
        logger.warning("   The app will still run using rule-based ASTRA 1.0 as fallback.")
    else:
        logger.info("\n🎉 All models trained successfully!")
    logger.info("   Restart the backend server to load new models.")


if __name__ == "__main__":
    main()
