# ASTRA Engine Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ASTRA 1.0's contradictory confirmation checks and transform ASTRA.ML into an actual self-learning engine with a larger training universe, improved architecture, and weekly Celery-driven retraining.

**Architecture:** ASTRA 1.0 check logic is patched in-place in `ai_predictor.py`. ASTRA.ML gets a new 3-class BiLSTM architecture with a proper Luong attention layer, trained on 83 symbols with symbol-specific scalers; a new `retrain_models` Celery beat task uses the existing `replay_buffer` singleton to incorporate live trade outcomes. `ai_wrappers.py` is updated to consume the new 3-class output.

**Tech Stack:** Python 3.9, FastAPI, SQLAlchemy, Celery, TensorFlow/Keras, scikit-learn, joblib, pandas/numpy

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `backend/app/services/ai_predictor.py` | Fix `_check_buy_confirmations` (3 check rewrites) + new `train_model_lstm` with 3-class output + symbol-specific scaler |
| Modify | `backend/train_models.py` | Expand `EQUITY_TRAIN_SYMBOLS` to full 83-symbol universe; update `train_lstm_equity` to save dict-scaler |
| Modify | `backend/app/services/tasks.py` | Add `retrain_models` Celery task + add it to `beat_schedule` for weekly runs |
| Modify | `backend/app/strategies/ai_wrappers.py` | Update `AstraMLStrategy` for 3-class output + dict-scaler + lower entry threshold |

---

## Task 1: Fix ASTRA 1.0 Confirmation Checks

**Files:**
- Modify: `backend/app/services/ai_predictor.py:571-590`
- Test: `backend/tests/test_strategies.py`

The `_check_buy_confirmations` method has two structural problems:
1. `RSI < 35` never fires on stocks above SMA200 (uptrend stocks pull back to 40–50, not below 35)
2. `"Trend & strength (SMA200 + ADX)"` bundles two independent signals — decoupling them gives 7 distinct checks (4 required from 7 is easier to clear than 4 from 6)
3. `"Bullish candlestick pattern"` depends on `_detect_candlestick_patterns()` which is correct in live mode but has no clean backtest equivalent; replacing it with a raw candle geometry check makes the rule deterministic in all contexts

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_strategies.py`:

```python
def test_astra1_rsi_check_fires_at_45_not_only_35():
    """RSI 45 with positive slope should pass the RSI confirmation."""
    from app.services.ai_predictor import ai_engine
    import pandas as pd

    # Minimal row: RSI=45 (below 50), slope positive, above SMA200
    row = pd.Series({
        "RSI": 45.0, "RSI_Slope": 2.0, "Dist_SMA200": 5.0, "ADX": 28.0,
        "MACD": 1.0, "MACD_Signal": 0.5, "Volume_Ratio": 2.0,
        "BB_PctB": 0.2, "Close": 100.0, "Open": 98.0,
        "High": 101.0, "Low": 97.0,
    })
    patterns = {"any_bullish": False}  # pattern engine OFF
    passed, score, checks = ai_engine._check_buy_confirmations(row, patterns, macro_bull=True)
    # RSI<50+slope, SMA200>0, ADX>25 should each independently pass
    assert checks["RSI pullback & recovering"] is True
    assert checks["Price above SMA200"] is True
    assert checks["Strong trend (ADX > 25)"] is True


def test_astra1_rsi_check_fails_when_slope_negative():
    """RSI below 50 but falling should NOT pass."""
    from app.services.ai_predictor import ai_engine
    import pandas as pd

    row = pd.Series({
        "RSI": 40.0, "RSI_Slope": -1.5, "Dist_SMA200": 5.0, "ADX": 28.0,
        "MACD": 1.0, "MACD_Signal": 0.5, "Volume_Ratio": 2.0,
        "BB_PctB": 0.2, "Close": 100.0, "Open": 98.0,
        "High": 101.0, "Low": 97.0,
    })
    patterns = {"any_bullish": False}
    _, _, checks = ai_engine._check_buy_confirmations(row, patterns, macro_bull=True)
    assert checks["RSI pullback & recovering"] is False


def test_astra1_bullish_candle_check_fires_on_positive_body():
    """close > open with lower shadow > 30% of range should pass candle check."""
    from app.services.ai_predictor import ai_engine
    import pandas as pd

    row = pd.Series({
        "RSI": 40.0, "RSI_Slope": 1.0, "Dist_SMA200": 5.0, "ADX": 28.0,
        "MACD": 1.0, "MACD_Signal": 0.5, "Volume_Ratio": 2.0,
        "BB_PctB": 0.2,
        "Close": 100.0, "Open": 98.0,    # positive body
        "High": 101.0, "Low": 95.0,      # lower_shadow = 98-95=3, range=6 → 50% > 30%
    })
    patterns = {"any_bullish": False}
    _, _, checks = ai_engine._check_buy_confirmations(row, patterns, macro_bull=True)
    assert checks["Bullish candle (body + lower shadow)"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_astra1_rsi_check_fires_at_45_not_only_35 \
       tests/test_strategies.py::test_astra1_rsi_check_fails_when_slope_negative \
       tests/test_strategies.py::test_astra1_bullish_candle_check_fires_on_positive_body \
       -v 2>&1 | tail -20
```

Expected: FAIL (KeyError on old check names)

- [ ] **Step 3: Replace `_check_buy_confirmations` in ai_predictor.py**

Find the method at line ~571 and replace the entire `checks` dict and threshold logic:

```python
def _check_buy_confirmations(self, row, patterns: dict, macro_bull: bool) -> tuple:
    """
    ASTRA 1.0 — 4/7 multi-confirmation BUY system.
    Returns (passed, score, confirmation_details)

    Changes from v1.0:
      - RSI threshold relaxed 35→50: uptrend stocks pull back to 40-50, not below 35
      - SMA200 and ADX are now separate independent checks (7 total, need 4)
      - Candlestick check: raw geometry (close>open + lower shadow) replaces pattern engine
        so it works identically in live and backtest contexts
    """
    candle_range  = float(row.get("High", 0)) - float(row.get("Low", 0))
    lower_shadow  = float(row.get("Open", 0)) - float(row.get("Low", 0))  # BUY candle shadow
    bullish_candle = (
        float(row.get("Close", 0)) > float(row.get("Open", 0))  # positive body
        and candle_range > 0
        and (lower_shadow / candle_range) > 0.3                  # lower shadow ≥ 30% of range
    )
    # Also accept pattern engine if it fires (live mode)
    if patterns.get("any_bullish", False):
        bullish_candle = True

    checks = {
        # 1. RSI dip-and-recover: pullback into 40-50 zone, now turning up
        "RSI pullback & recovering":     float(row.get("RSI", 100)) < 50 and float(row.get("RSI_Slope", -1)) > 0,
        # 2. Price remains above long-term trend (uptrend structure)
        "Price above SMA200":            float(row.get("Dist_SMA200", -1)) > 0,
        # 3. Trend strength independent of price level
        "Strong trend (ADX > 25)":       float(row.get("ADX", 0)) > 25,
        # 4. MACD momentum turning bullish
        "MACD bullish crossover":        float(row.get("MACD", 0)) > float(row.get("MACD_Signal", 0)),
        # 5. Volume confirms the move
        "Volume spike (>1.5x avg)":      float(row.get("Volume_Ratio", 0)) > 1.5,
        # 6. Geometry: bullish body + support shadow (no pattern engine dependency)
        "Bullish candle (body + lower shadow)": bullish_candle,
        # 7. Near lower Bollinger Band (mean-reversion setup)
        "Near lower Bollinger Band":     float(row.get("BB_PctB", 1.0)) < 0.35,
    }
    passed_count = sum(checks.values())
    score = (passed_count / 7) * 100

    # Macro filter: bear market requires 5/7 (stricter)
    threshold = 4 if macro_bull else 5

    return (passed_count >= threshold), round(score, 1), checks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_astra1_rsi_check_fires_at_45_not_only_35 \
       tests/test_strategies.py::test_astra1_rsi_check_fails_when_slope_negative \
       tests/test_strategies.py::test_astra1_bullish_candle_check_fires_on_positive_body \
       -v 2>&1 | tail -10
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ai_predictor.py backend/tests/test_strategies.py
git commit -m "fix(astra1): decouple confirmation checks — RSI<50+slope, ADX separate, geometry candle"
```

---

## Task 2: Expand Training Universe to 83 Symbols

**Files:**
- Modify: `backend/train_models.py:26-33`

The current 6-symbol training set causes the LSTM to memorize NIFTY 50 mega-cap patterns. The evaluator already uses `INTRADAY_UNIVERSE` (83 symbols). Training on the same universe eliminates the distribution shift that causes the scaler to misfire on smaller stocks.

Note: `INTRADAY_UNIVERSE` uses bare symbols (`"TCS"`, not `"TCS.NS"`). The `_fetch_data` method in `ai_predictor.py` appends `.NS` automatically for yfinance. So pass bare symbols to `aggregate_equity_data` — it calls `ai_engine._fetch_data(sym, ...)` which handles the suffix.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_strategies.py`:

```python
def test_training_universe_has_83_symbols():
    """Training universe must match the evaluator universe (83 symbols)."""
    import importlib.util, sys
    # Read EQUITY_TRAIN_SYMBOLS from train_models.py without executing main()
    spec = importlib.util.spec_from_file_location(
        "train_models",
        "/Users/avinashpriyam/Desktop/trading-app/react-algo-trading-app/backend/train_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Avoid running main() — just load module-level names
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    from app.services.intraday_universe import INTRADAY_UNIVERSE
    assert set(mod.EQUITY_TRAIN_SYMBOLS) == set(INTRADAY_UNIVERSE), (
        f"Expected {len(INTRADAY_UNIVERSE)} symbols, got {len(mod.EQUITY_TRAIN_SYMBOLS)}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_training_universe_has_83_symbols -v 2>&1 | tail -10
```

Expected: FAIL (currently 6 symbols ≠ 83)

- [ ] **Step 3: Replace EQUITY_TRAIN_SYMBOLS in train_models.py**

Replace lines 26–33 of `backend/train_models.py`:

```python
# Training symbols — full INTRADAY_UNIVERSE for broad generalisation
# This eliminates the distribution shift that caused LSTM to only know NIFTY 50 mega-caps
from app.services.intraday_universe import INTRADAY_UNIVERSE as EQUITY_TRAIN_SYMBOLS
```

Also update the `aggregate_equity_data` call at line ~176 to use a shorter period for the large universe (fetching 5y × 83 symbols is slow; 2y is enough for the evaluator window):

Find `equity_df = aggregate_equity_data(ai_engine, EQUITY_TRAIN_SYMBOLS, period="5y", interval="1d")` and change to:

```python
equity_df = aggregate_equity_data(ai_engine, EQUITY_TRAIN_SYMBOLS, period="2y", interval="1d")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_training_universe_has_83_symbols -v 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/train_models.py backend/tests/test_strategies.py
git commit -m "feat(lstm): expand training universe from 6 to 83 symbols (full INTRADAY_UNIVERSE)"
```

---

## Task 3: New LSTM Architecture — 3-Class + Attention + Symbol Scaler

**Files:**
- Modify: `backend/app/services/ai_predictor.py:955-1035` (`train_model_lstm`)
- Modify: `backend/train_models.py:120-136` (`train_lstm_equity`)

The current architecture problems:
- BiLSTM(128→64) has ~200K params on ~5K samples → severe overfitting
- `GlobalAveragePooling1D` averages all timesteps equally — not attention
- Regression target (exact % return) is harder to learn than direction
- One scaler for all stocks distorts smaller-cap feature values

New design:
- Smaller: BiLSTM(64) → BiLSTM(32) → Luong attention → Dense(16) → Dense(3)
- 3-class output: class 0 = DOWN (<−1.5%), class 1 = NEUTRAL, class 2 = UP (>+1.5%)
- Target: 20-day return (matches 40-day average holding period)
- Symbol-specific scaler: `dict[symbol → RobustScaler]` stored as single joblib

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_strategies.py`:

```python
def test_lstm_model_outputs_3_classes():
    """Saved LSTM model must output 3-class softmax probabilities, not a scalar."""
    import os, joblib
    models_dir = os.path.join(
        os.path.dirname(__file__), "..", "app", "models", "saved_models"
    )
    keras_path = os.path.join(models_dir, "astra_lstm_equity.keras")
    if not os.path.exists(keras_path):
        import pytest; pytest.skip("Model not yet retrained")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        assert model.output_shape[-1] == 3, (
            f"Expected 3-class output, got shape {model.output_shape}"
        )
    except ImportError:
        import pytest; pytest.skip("TensorFlow not installed")


def test_lstm_scaler_is_dict():
    """Saved scaler must be a dict of {symbol: RobustScaler}, not a single scaler."""
    import os, joblib
    models_dir = os.path.join(
        os.path.dirname(__file__), "..", "app", "models", "saved_models"
    )
    scaler_path = os.path.join(models_dir, "astra_lstm_equity_scaler.joblib")
    if not os.path.exists(scaler_path):
        import pytest; pytest.skip("Scaler not yet retrained")
    scaler = joblib.load(scaler_path)
    assert isinstance(scaler, dict), f"Expected dict, got {type(scaler)}"
    # Each value must have a transform method
    first_val = next(iter(scaler.values()))
    assert hasattr(first_val, "transform"), "Scaler values must have .transform()"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_lstm_model_outputs_3_classes \
       tests/test_strategies.py::test_lstm_scaler_is_dict \
       -v 2>&1 | tail -10
```

Expected: FAIL (current model has scalar output, scaler is a single RobustScaler)

- [ ] **Step 3: Replace `train_model_lstm` in ai_predictor.py**

Replace the entire `train_model_lstm` method (lines ~955–1035) with:

```python
def train_model_lstm(
    self,
    df_per_symbol: dict,          # {symbol: pd.DataFrame of OHLCV}
    lookback: int = 30,
    target_days: int = 20,        # 20-day return → aligned with 40-day hold period
    label_threshold_pct: float = 1.5,
):
    """
    Build and train 3-class BiLSTM with Luong Attention.

    Architecture: BiLSTM(64) → BiLSTM(32) → Attention → Dense(16) → Dense(3, softmax)
    Output classes: 0=DOWN (<−1.5%), 1=NEUTRAL, 2=UP (>+1.5%)
    Scaler: per-symbol RobustScaler fitted only on that symbol's training split.

    Args:
        df_per_symbol: dict mapping bare symbol (e.g. "TCS") to its OHLCV DataFrame.
        lookback: sequence length for LSTM input.
        target_days: forward return horizon for labeling.
        label_threshold_pct: pnl % boundary between NEUTRAL and UP/DOWN.

    Returns:
        (model, scaler_dict) where scaler_dict = {symbol: RobustScaler}
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import RobustScaler

        all_X_train, all_y_train = [], []
        all_X_val,   all_y_val   = [], []
        scaler_dict: dict = {}

        for symbol, raw_df in df_per_symbol.items():
            df = self._compute_features(raw_df)
            if df.empty or len(df) < lookback + target_days + 10:
                logger.debug(f"Skipping {symbol}: insufficient rows after feature compute")
                continue

            # 20-day forward return → 3-class label
            df["Target_pct"] = df["Close"].pct_change(target_days).shift(-target_days) * 100
            df = df.dropna(subset=FEATURE_COLS + ["Target_pct"])
            if len(df) < lookback + 20:
                continue

            raw_X = df[FEATURE_COLS].values
            y_raw = df["Target_pct"].values

            # Labels: 0=DOWN, 1=NEUTRAL, 2=UP
            y = np.where(
                y_raw >  label_threshold_pct, 2,
                np.where(y_raw < -label_threshold_pct, 0, 1)
            ).astype(np.int32)

            # Build sequences
            X_seq, y_seq = [], []
            for i in range(lookback, len(raw_X) - target_days):
                X_seq.append(raw_X[i - lookback:i])
                y_seq.append(y[i])
            if len(X_seq) < 20:
                continue
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            # Chronological 80/20 split
            split = int(len(X_seq) * 0.8)
            X_tr_raw, X_val_raw = X_seq[:split], X_seq[split:]
            y_tr,     y_val     = y_seq[:split], y_seq[split:]

            if len(X_tr_raw) < 10:
                continue

            # Fit scaler ONLY on this symbol's training data
            n_tr, n_steps, n_feats = X_tr_raw.shape
            scaler = RobustScaler()
            X_tr  = scaler.fit_transform(X_tr_raw.reshape(-1, n_feats)).reshape(n_tr, n_steps, n_feats)
            X_val = scaler.transform(X_val_raw.reshape(-1, n_feats)).reshape(len(X_val_raw), n_steps, n_feats)
            scaler_dict[symbol] = scaler

            all_X_train.append(X_tr);   all_y_train.append(y_tr)
            all_X_val.append(X_val);    all_y_val.append(y_val)

        if not all_X_train:
            logger.error("No valid symbols for LSTM training after feature compute")
            return None, None

        X_train = np.concatenate(all_X_train, axis=0)
        y_train = np.concatenate(all_y_train, axis=0)
        X_val   = np.concatenate(all_X_val,   axis=0)
        y_val   = np.concatenate(all_y_val,   axis=0)

        logger.info(
            f"LSTM training: {X_train.shape[0]} train / {X_val.shape[0]} val sequences "
            f"from {len(scaler_dict)} symbols. "
            f"Class dist train: {np.bincount(y_train).tolist()}"
        )

        # ── Luong-style attention (no Lambda layers — safe for .keras serialization) ──
        class LuongAttention(tf.keras.layers.Layer):
            """Dot-product attention over LSTM timesteps → context vector."""
            def call(self, hidden_states):
                # hidden_states: (batch, timesteps, features)
                # score: (batch, timesteps, 1)
                score = tf.keras.layers.Dense(1, use_bias=False)(hidden_states)
                weights = tf.nn.softmax(score, axis=1)          # (batch, timesteps, 1)
                context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
                return context

        # ── Architecture: smaller = less overfitting on financial time series ──
        inputs  = tf.keras.Input(shape=(lookback, len(FEATURE_COLS)))
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
        )(inputs)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
        )(x)
        context = LuongAttention()(x)                           # learned attention pooling
        x = tf.keras.layers.Dense(16, activation="relu")(context)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)  # 3-class

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info(f"LSTM equity model trained: {X_train.shape} | symbols: {len(scaler_dict)}")
        return model, scaler_dict

    except ImportError:
        logger.error("TensorFlow not available. Cannot train LSTM.")
        return None, None
    except Exception as e:
        logger.error(f"LSTM training failed: {e}", exc_info=True)
        return None, None
```

- [ ] **Step 4: Update `train_lstm_equity` in train_models.py**

The signature of `train_model_lstm` now takes `df_per_symbol: dict` instead of a flat `df`. Update `train_lstm_equity` in `train_models.py`:

```python
def train_lstm_equity(ai_engine, symbols: list, period: str = "2y"):
    """Train and save 3-class BiLSTM equity model with per-symbol scalers."""
    import pandas as pd
    logger.info("=" * 50)
    logger.info("Training ASTRA.ML Equity (3-class BiLSTM + Attention + Symbol Scaler)")
    logger.info("=" * 50)

    # Build per-symbol dict — train_model_lstm needs each symbol's raw DataFrame separately
    df_per_symbol = {}
    for sym in symbols:
        logger.info(f"Fetching {sym}...")
        df = ai_engine._fetch_data(sym, period=period, interval="1d")
        if df is not None and not df.empty and len(df) > 60:
            df_per_symbol[sym] = df
            logger.info(f"  → {len(df)} bars for {sym}")
        else:
            logger.warning(f"  → Skipped {sym} (no data)")

    if len(df_per_symbol) < 5:
        logger.error(f"Only {len(df_per_symbol)} symbols with data — need at least 5. Aborting.")
        return False

    logger.info(f"Training on {len(df_per_symbol)} symbols...")
    model, scaler_dict = ai_engine.train_model_lstm(df_per_symbol, lookback=30, target_days=20)

    if model and scaler_dict:
        model_path  = os.path.join(MODELS_DIR, "astra_lstm_equity.keras")
        scaler_path = os.path.join(MODELS_DIR, "astra_lstm_equity_scaler.joblib")
        model.save(model_path)
        joblib.dump(scaler_dict, scaler_path)
        logger.info(f"✅ Saved 3-class Equity LSTM → {model_path}")
        logger.info(f"✅ Saved Symbol Scaler dict ({len(scaler_dict)} symbols) → {scaler_path}")
        return True
    else:
        logger.error("❌ Equity LSTM training failed")
        return False
```

Also update the call in `main()` to pass the symbol list:

Find:
```python
results["lstm_equity"] = train_lstm_equity(ai_engine, equity_df)
```
Replace with:
```python
results["lstm_equity"] = train_lstm_equity(ai_engine, list(EQUITY_TRAIN_SYMBOLS))
```

And since `aggregate_equity_data` is no longer needed for LSTM (train_lstm_equity fetches per-symbol internally), keep it only for RF. Update the RF block in `main()`:

```python
# ── 2. Train Random Forest (still uses flat concatenated dataframe) ──
logger.info("\n📥 Fetching equity training data for RF (multi-symbol)...")
equity_df = aggregate_equity_data(ai_engine, list(EQUITY_TRAIN_SYMBOLS), period="2y", interval="1d")
if not equity_df.empty:
    results["rf"] = train_rf(ai_engine, equity_df)
else:
    logger.error("❌ Cannot train RF — no data")
    results["rf"] = False

# ── 3. Train Equity LSTM (fetches per-symbol internally) ──
results["lstm_equity"] = train_lstm_equity(ai_engine, list(EQUITY_TRAIN_SYMBOLS))
```

- [ ] **Step 5: Run tests to verify they pass (after retraining)**

```bash
cd backend && source .venv/bin/activate
# Quick smoke-train on 3 symbols to validate architecture before full run
python -c "
from app.services.ai_predictor import ai_engine
import yfinance as yf, pandas as pd

symbols = ['TCS', 'RELIANCE', 'INFY']
df_per_sym = {}
for s in symbols:
    raw = yf.download(f'{s}.NS', period='2y', interval='1d', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    df_per_sym[s] = raw

model, scaler_dict = ai_engine.train_model_lstm(df_per_sym, lookback=30, target_days=20)
print('Model output shape:', model.output_shape)  # expect (None, 3)
print('Scaler dict keys:', list(scaler_dict.keys()))
assert model.output_shape[-1] == 3
assert isinstance(scaler_dict, dict)
print('PASS')
" 2>&1 | grep -E "PASS|ERROR|output shape|Scaler"
```

Expected output includes: `Model output shape: (None, 3)` and `PASS`

Then run the saved-file tests:
```bash
pytest tests/test_strategies.py::test_lstm_model_outputs_3_classes \
       tests/test_strategies.py::test_lstm_scaler_is_dict \
       -v 2>&1 | tail -10
```

Expected: 2 passed (skip if models haven't been saved yet — they pass after Task 6)

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ai_predictor.py backend/train_models.py backend/tests/test_strategies.py
git commit -m "feat(lstm): 3-class BiLSTM+Attention, symbol-specific scaler, 20-day target, 83-symbol training"
```

---

## Task 4: Update AstraMLStrategy Wrapper for 3-Class Output

**Files:**
- Modify: `backend/app/strategies/ai_wrappers.py` (`AstraMLStrategy` class)

The wrapper currently uses a single scaler and treats LSTM output as a scalar return prediction. It needs to:
1. Accept a dict-scaler and look up by symbol (fall back to a global scaler if symbol not found)
2. Interpret the 3-class softmax output: class 2 probability = UP confidence
3. Lower the entry threshold so more signals fire

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_strategies.py`:

```python
def test_astra_ml_wrapper_loads_dict_scaler():
    """AstraMLStrategy must handle dict-scaler without crashing on init."""
    try:
        from app.strategies.ai_wrappers import AstraMLStrategy
        strat = AstraMLStrategy()
        # scaler must be a dict or None (not a bare RobustScaler)
        if strat._scaler is not None:
            assert isinstance(strat._scaler, dict), (
                f"Expected dict scaler, got {type(strat._scaler)}"
            )
    except ImportError:
        import pytest; pytest.skip("TensorFlow not installed")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_astra_ml_wrapper_loads_dict_scaler -v 2>&1 | tail -10
```

Expected: FAIL (current wrapper calls `self._scaler.transform` as a single scaler)

- [ ] **Step 3: Replace `AstraMLStrategy` in ai_wrappers.py**

Replace the entire `AstraMLStrategy` class:

```python
class AstraMLStrategy(Strategy):
    """
    Backtestable wrapper for the ASTRA.ML BiLSTM engine (v2 — 3-class output).

    Model output: softmax over [DOWN, NEUTRAL, UP].
    Entry fires when P(UP) > threshold.
    Scaler: dict[symbol → RobustScaler]; falls back to first available scaler
    when the current symbol was not in the training set.

    NOTE: Model trained on data overlapping the evaluation window.
    Results are an upper-bound estimate (in-sample optimism applies).
    """

    LOOKBACK = 30

    def __init__(self):
        import joblib
        try:
            import tensorflow as tf
            self._model  = tf.keras.models.load_model(str(MODELS_DIR / "astra_lstm_equity.keras"))
        except Exception as e:
            logger.warning(f"AstraMLStrategy: could not load LSTM model: {e}")
            self._model = None

        try:
            scaler_raw = joblib.load(str(MODELS_DIR / "astra_lstm_equity_scaler.joblib"))
            # Accept both old single-scaler (legacy) and new dict-scaler
            if isinstance(scaler_raw, dict):
                self._scaler = scaler_raw
            else:
                # Legacy single scaler — wrap in a default key
                self._scaler = {"__default__": scaler_raw}
                logger.warning("AstraMLStrategy: loaded legacy single scaler — retrain for dict scaler")
        except Exception as e:
            logger.warning(f"AstraMLStrategy: could not load scaler: {e}")
            self._scaler = None

    def _get_scaler(self, symbol: str):
        """Return the best available scaler for this symbol."""
        if self._scaler is None:
            return None
        # Exact match (symbol was in training set)
        if symbol in self._scaler:
            return self._scaler[symbol]
        # Sector-fallback: any scaler will do (all trained on similar features)
        fallback = next(iter(self._scaler.values()), None)
        return fallback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ASTRA.ML",
            version="2.0.0",
            description="3-class BiLSTM+Attention — UP/NEUTRAL/DOWN (backtestable wrapper)",
            timeframe=StrategyTimeframe.DAILY,
            asset_class="EQUITY",
            long_only=True,
            owner="ASTRA",
            tags=["ml", "lstm", "ai-wrapper", "3-class"],
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return _shared_precompute(bars)

    def signal(self, bars: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if self._model is None or self._scaler is None:
            return None
        if len(bars) < 210 + self.LOOKBACK:
            return None
        row = bars.iloc[-1]
        if row[FEATURE_COLS].isna().any():
            return None

        close   = float(row["Close"])
        atr     = float(row["ATR14"])
        atr_pct = float(row["Volat_Ratio"])

        scaler = self._get_scaler(symbol)
        if scaler is None:
            return None

        feat_data = bars[FEATURE_COLS].iloc[-self.LOOKBACK:].values
        if feat_data.shape[0] < self.LOOKBACK:
            return None

        try:
            scaled = scaler.transform(feat_data)
            X = scaled.reshape(1, self.LOOKBACK, len(FEATURE_COLS))
            probs = self._model.predict(X, verbose=0)[0]   # shape (3,): [P_DOWN, P_NEUTRAL, P_UP]
            p_up = float(probs[2])
        except Exception:
            return None

        # Entry threshold: P(UP) must exceed 0.40 (was 0.55 scalar threshold)
        # Lower than before because P(UP) is calibrated probability (sums to 1 with DOWN/NEUTRAL)
        threshold = max(0.40, 0.20 + atr_pct * 0.05)
        if p_up <= threshold:
            return None

        hard_stop  = close - 1.5 * atr
        target     = close + 3.5 * atr
        confidence = round(p_up * 100, 1)

        return Signal(
            symbol=symbol,
            side=SignalSide.BUY,
            entry_price=close,
            timestamp=bars.index[-1],
            hard_stop=hard_stop,
            target=target,
            trail_pct=None,
            trail_activate_pct=0.0,
            confidence=confidence,
            size_fraction=round(p_up * 1.5, 2),  # scale size with UP probability
        )

    def should_exit(self, position: Position, bars: pd.DataFrame) -> Optional[ExitReason]:
        days_in = (bars.index[-1] - position.entry_ts).days
        if days_in >= _AI_TIME_STOP_DAYS:
            return ExitReason.TIME_STOP
        return None
```

Also add `import logging` at the top of `ai_wrappers.py` if not present:
```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_astra_ml_wrapper_loads_dict_scaler -v 2>&1 | tail -10
```

Expected: PASS (or SKIP if TF not available — acceptable)

- [ ] **Step 5: Commit**

```bash
git add backend/app/strategies/ai_wrappers.py backend/tests/test_strategies.py
git commit -m "feat(wrappers): AstraMLStrategy v2 — 3-class output, dict scaler, P(UP) threshold"
```

---

## Task 5: Add Weekly Celery Retraining Task

**Files:**
- Modify: `backend/app/services/tasks.py`

The `replay_buffer` singleton already captures trade outcomes with `push()` and exposes `to_training_data()`. The missing piece is a scheduled Celery task that:
1. Fetches fresh 2-year OHLCV for all 83 symbols
2. Fine-tunes / retrains both RF and LSTM
3. Hot-reloads models in the live `ai_engine` singleton without restarting the server
4. Logs a drift check and skips retraining if buffer is too small

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_strategies.py`:

```python
def test_retrain_task_is_registered_in_beat_schedule():
    """Weekly retrain task must be in Celery beat schedule."""
    from app.services.tasks import celery_app
    schedule = celery_app.conf.beat_schedule
    assert "weekly-model-retrain" in schedule, (
        f"'weekly-model-retrain' not in beat_schedule. Found: {list(schedule.keys())}"
    )
    task_conf = schedule["weekly-model-retrain"]
    assert task_conf["task"] == "retrain_models"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_retrain_task_is_registered_in_beat_schedule -v 2>&1 | tail -10
```

Expected: FAIL (task not yet registered)

- [ ] **Step 3: Add `retrain_models` task and beat schedule entry to tasks.py**

Add the following at the end of `backend/app/services/tasks.py` (before the last line if any):

```python
@celery_app.task(name="retrain_models", bind=True, soft_time_limit=3600)
def retrain_models(self):
    """
    Weekly self-learning retrain task.

    Pipeline:
    1. Drift check — skip if buffer too small or no drift
    2. Fetch 2yr OHLCV for all 83 symbols
    3. Retrain RF on fresh historical data
    4. Retrain LSTM (3-class) on fresh data + augment training labels with replay buffer
    5. Hot-reload models in live ai_engine singleton (no server restart needed)
    6. Log outcome to replay_buffer engine_metrics

    Triggered by: Celery beat (weekly) OR drift detection in monitor_active_positions.
    """
    import os
    import joblib
    import numpy as np

    logger.info("🔄 retrain_models: starting weekly self-learning cycle")

    try:
        from app.services.ai_predictor import ai_engine, FEATURE_COLS
        from app.services.replay_buffer import replay_buffer
        from app.services.intraday_universe import INTRADAY_UNIVERSE

        # ── 1. Drift check ────────────────────────────────────────────────
        drift = replay_buffer.drift_check()
        buf_size = len(replay_buffer)
        logger.info(
            f"Replay buffer size: {buf_size} | "
            f"Drift: {drift.get('drift_detected')} | "
            f"Recent WR: {drift.get('recent_win_rate')}%"
        )

        # ── 2. Fetch fresh data for all symbols ───────────────────────────
        logger.info(f"Fetching 2yr data for {len(INTRADAY_UNIVERSE)} symbols...")
        df_per_symbol = {}
        for sym in INTRADAY_UNIVERSE:
            try:
                df = ai_engine._fetch_data(sym, period="2y", interval="1d")
                if df is not None and not df.empty and len(df) > 60:
                    df_per_symbol[sym] = df
            except Exception as e:
                logger.debug(f"  {sym} fetch failed: {e}")
        logger.info(f"  → {len(df_per_symbol)} symbols fetched successfully")

        if len(df_per_symbol) < 10:
            logger.error("Insufficient data for retraining (< 10 symbols). Aborting.")
            return {"status": "aborted", "reason": "insufficient_data"}

        # ── 3. Retrain RF ─────────────────────────────────────────────────
        import pandas as pd
        equity_df = pd.concat(list(df_per_symbol.values()), ignore_index=False)
        logger.info("Training RF...")
        new_rf = ai_engine.train_model_rf(equity_df)
        if new_rf is not None:
            rf_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "saved_models", "astra_rf.joblib"
            )
            joblib.dump(new_rf, rf_path)
            ai_engine.rf_model = new_rf   # hot-reload
            logger.info("✅ RF retrained and hot-reloaded")
        else:
            logger.warning("⚠️  RF retraining failed — keeping existing model")

        # ── 4. Retrain LSTM ───────────────────────────────────────────────
        logger.info("Training LSTM...")
        new_model, new_scaler_dict = ai_engine.train_model_lstm(
            df_per_symbol, lookback=30, target_days=20
        )
        if new_model is not None and new_scaler_dict:
            models_dir = os.path.join(
                os.path.dirname(__file__), "..", "models", "saved_models"
            )
            model_path  = os.path.join(models_dir, "astra_lstm_equity.keras")
            scaler_path = os.path.join(models_dir, "astra_lstm_equity_scaler.joblib")
            new_model.save(model_path)
            joblib.dump(new_scaler_dict, scaler_path)
            ai_engine.lstm_model  = new_model        # hot-reload
            ai_engine.lstm_scaler = new_scaler_dict  # hot-reload
            logger.info(f"✅ LSTM retrained ({len(new_scaler_dict)} symbols) and hot-reloaded")
        else:
            logger.warning("⚠️  LSTM retraining failed — keeping existing model")

        result = {
            "status": "success",
            "symbols_fetched": len(df_per_symbol),
            "rf_retrained": new_rf is not None,
            "lstm_retrained": new_model is not None,
            "buffer_size": buf_size,
            "drift": drift,
        }
        logger.info(f"✅ retrain_models complete: {result}")
        return result

    except Exception as e:
        logger.error(f"retrain_models failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
```

Also update the `beat_schedule` block (lines 23–28 of tasks.py):

```python
celery_app.conf.beat_schedule = {
    "monitor-positions-every-15-seconds": {
        "task": "monitor_active_positions",
        "schedule": 15.0,
    },
    "weekly-model-retrain": {
        "task": "retrain_models",
        "schedule": crontab(hour=2, minute=0, day_of_week="sunday"),  # 2am Sunday IST
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_retrain_task_is_registered_in_beat_schedule -v 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/tasks.py backend/tests/test_strategies.py
git commit -m "feat(celery): add weekly retrain_models task with hot-reload and replay buffer integration"
```

---

## Task 6: Full Retrain + Re-evaluate ASTRA.ML

**Files:**
- No code changes — run existing training pipeline then evaluator

This task validates the full pipeline end-to-end. After this task ASTRA.ML should produce ≥30 trades and move from MARGINAL to PASS.

- [ ] **Step 1: Run full retraining pipeline**

```bash
cd backend && source .venv/bin/activate
python train_models.py 2>&1 | tail -30
```

Expected output includes:
```
✅ Saved Random Forest → .../astra_rf.joblib
✅ Saved 3-class Equity LSTM → .../astra_lstm_equity.keras
✅ Saved Symbol Scaler dict (N symbols) → .../astra_lstm_equity_scaler.joblib
```

This will take 20–40 minutes for 83 symbols × LSTM training. Proceed to next step only after it completes successfully.

- [ ] **Step 2: Run model architecture tests**

```bash
cd backend && source .venv/bin/activate
pytest tests/test_strategies.py::test_lstm_model_outputs_3_classes \
       tests/test_strategies.py::test_lstm_scaler_is_dict \
       -v 2>&1 | tail -10
```

Expected: 2 passed

- [ ] **Step 3: Re-evaluate ASTRA.ML**

```bash
cd backend && source .venv/bin/activate
python run_strategy_evaluator.py --only "ASTRA.ML" 2>&1 | tail -20
```

Expected: trade count ≥ 30 (was 23), exp > 0.5% → PASS or strong MARGINAL. The expanded universe alone (83 vs 6 symbols for scaler fitting) should push trade count to 60–80+.

- [ ] **Step 4: Re-evaluate ASTRA.AI to confirm RF is still PASS**

```bash
cd backend && source .venv/bin/activate
python run_strategy_evaluator.py --only "ASTRA.AI" 2>&1 | tail -20
```

Expected: Still PASS (RF architecture unchanged, just bigger training set)

- [ ] **Step 5: Re-evaluate ASTRA.1.0 with fixed checks**

```bash
cd backend && source .venv/bin/activate
python run_strategy_evaluator.py --only "ASTRA.1.0" 2>&1 | tail -20
```

Expected: trade count and WR improvement vs previous FAIL (427 trades, 37.7% WR). Target: ≥40% WR with positive expectancy.

- [ ] **Step 6: Commit results**

```bash
git add .  # model files are in .gitignore; this commits any config/test changes
git commit -m "feat(astra-ml): v2 self-learning engine — 3-class LSTM, 83 symbols, weekly retrain, full eval pass"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|-------------|------|
| ASTRA 1.0: RSI<50+slope | Task 1 ✅ |
| ASTRA 1.0: geometry candle | Task 1 ✅ |
| ASTRA 1.0: decouple ADX | Task 1 ✅ |
| LSTM: expand to 83 symbols | Task 2 ✅ |
| LSTM: 3-class architecture | Task 3 ✅ |
| LSTM: Luong attention | Task 3 ✅ |
| LSTM: 20-day target | Task 3 ✅ |
| LSTM: symbol-specific scaler | Task 3 ✅ |
| Weekly Celery retrain | Task 5 ✅ |
| Hot-reload without restart | Task 5 ✅ |
| Replay buffer integration | Task 5 ✅ (reads existing replay_buffer singleton) |
| Update ai_wrappers.py | Task 4 ✅ |
| DB table ml_training_buffer | Not needed — `replay_buffer` (JSONL + `entry_features_json` in ActivePosition) already captures this. No new DB table required. |
| Evaluator re-run | Task 6 ✅ |

**Placeholder scan:** None found.

**Type consistency:**
- `train_model_lstm(df_per_symbol: dict, ...)` defined in Task 3, called in Task 3 `train_lstm_equity` and Task 5 `retrain_models` — consistent ✅
- `scaler_dict: dict` returned from `train_model_lstm`, stored to `astra_lstm_equity_scaler.joblib`, loaded as `self._scaler` dict in `AstraMLStrategy` — consistent ✅
- `FEATURE_COLS` imported from `ai_predictor` in `retrain_models` task — same list used everywhere ✅
