"""
Train ASTRA's models with an honest hold-out, then train the production models.

  Phase A ("oos"):  train on data up to --cutoff only → models_oos/
                    measure out-of-sample quality on everything after it
  Phase B ("prod"): retrain on all data → app/models/saved_models/

Data comes from exported CSVs (POST /api/data/export on a machine with market
access), so runs are reproducible and need no network:

    ASTRA_OFFLINE_DATA_DIR=/path/to/export python scripts/train_and_validate.py

Writes a JSON summary to scripts/out/training_report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

CRYPTO = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD"]


def load_equity(symbols, asof=None) -> dict:
    from app.services.market_data import market_data
    if asof:
        os.environ["ASTRA_OFFLINE_ASOF"] = asof
    else:
        os.environ.pop("ASTRA_OFFLINE_ASOF", None)
    out = {}
    for s in symbols:
        df = market_data.get_ohlcv(s, period="10y", interval="1d")
        if len(df) > 300:
            df.attrs["symbol"] = s
            out[s] = df
    os.environ.pop("ASTRA_OFFLINE_ASOF", None)
    return out


def rf_oos_metrics(ai_engine, model, full_data: dict, cutoff: str) -> dict:
    """Information coefficient of RF predictions vs realised 5-day returns after cutoff."""
    from app.services.ai_predictor import FEATURE_COLS
    ds = ai_engine.build_rf_dataset(full_data)
    idx = ds.index.tz_localize(None) if ds.index.tz is not None else ds.index
    test = ds[idx > pd.Timestamp(cutoff)]
    if test.empty:
        return {}
    pred = model.predict(test[FEATURE_COLS].values)
    y = test["Target"].values
    ic = float(np.corrcoef(pred, y)[0, 1])
    # rank IC per date (cross-sectional), averaged
    t = test.assign(pred=pred)
    rics = t.groupby(t.index.normalize()).apply(
        lambda g: g["pred"].rank().corr(g["Target"].rank()) if len(g) >= 10 else np.nan).dropna()
    top = t[t["pred"] > np.maximum(1.2, t["Volat_Ratio"] * 0.5)]
    return {
        "test_rows": int(len(test)),
        "ic": round(ic, 4),
        "rank_ic_mean": round(float(rics.mean()), 4),
        "rank_ic_tstat": round(float(rics.mean() / (rics.std() / np.sqrt(len(rics)))), 2) if len(rics) > 2 else None,
        "avg_5d_ret_all": round(float(y.mean()), 3),
        "buy_signals": int(len(top)),
        "avg_5d_ret_when_buy": round(float(top["Target"].mean()), 3) if len(top) else None,
        "hit_rate_when_buy": round(float((top["Target"] > 0).mean()), 3) if len(top) else None,
    }


def lstm_oos_metrics(model, scaler, full_data: dict, cutoff: str, ai_engine,
                     target_days=20, thr=1.5) -> dict:
    from app.services.ai_predictor import lstm_predict_proba_series
    rows = []
    for sym, raw in full_data.items():
        f = ai_engine._compute_features(raw)
        if f.empty:
            continue
        fwd = f["Close"].pct_change(target_days).shift(-target_days) * 100
        probs = lstm_predict_proba_series(model, scaler, sym, f)
        # label for window ending at t uses return from t+1 → t+1+H (matches training offset)
        lab = fwd.shift(-1)
        j = probs.join(lab.rename("fwd"), how="inner").dropna()
        jidx = j.index.tz_localize(None) if j.index.tz is not None else j.index
        rows.append(j[jidx > pd.Timestamp(cutoff)])
    if not rows:
        return {}
    d = pd.concat(rows)
    y = np.where(d["fwd"] > thr, 2, np.where(d["fwd"] < -thr, 0, 1))
    pred = d[["P_DOWN", "P_NEUTRAL", "P_UP"]].values.argmax(axis=1)
    majority = np.bincount(y, minlength=3).argmax()
    buy = (d["P_UP"] >= 0.45) & (d["P_UP"] - d["P_DOWN"] >= 0.10)
    return {
        "test_windows": int(len(d)),
        "class_dist_test": np.bincount(y, minlength=3).tolist(),
        "accuracy": round(float((pred == y).mean()), 4),
        "majority_baseline": round(float((y == majority).mean()), 4),
        "avg_20d_ret_all": round(float(d["fwd"].mean()), 3),
        "buy_signals": int(buy.sum()),
        "avg_20d_ret_when_buy": round(float(d.loc[buy, "fwd"].mean()), 3) if buy.any() else None,
        "hit_rate_when_buy": round(float((d.loc[buy, "fwd"] > 0).mean()), 3) if buy.any() else None,
        "p_up_vs_ret_corr": round(float(np.corrcoef(d["P_UP"] - d["P_DOWN"], d["fwd"])[0, 1]), 4),
    }


def crypto_oos_metrics(model, scaler, cutoff: str, crypto_engine) -> dict:
    from app.services.crypto_engine import CRYPTO_FEATURE_COLS
    from app.services.market_data import market_data
    lookback = int(model.input_shape[1])
    rows = []
    for sym in CRYPTO:
        raw = market_data.get_ohlcv(sym, period="10y", interval="1d")
        f = crypto_engine._compute_features(raw)
        if len(f) < lookback + 10:
            continue
        fwd = f["Close"].pct_change(3).shift(-3) * 100
        F = f[CRYPTO_FEATURE_COLS]
        ok = F.notna().all(axis=1)
        F = F[ok]
        Z = scaler.transform(F.values)
        idx = np.arange(lookback, len(F))
        X = np.stack([Z[i - lookback:i] for i in idx])
        P = model.predict(X, verbose=0, batch_size=1024)
        df = pd.DataFrame(P, index=F.index[idx], columns=["P_DOWN", "P_NEUTRAL", "P_UP"])
        df["fwd"] = fwd.reindex(df.index)
        df = df.dropna()
        didx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        rows.append(df[didx > pd.Timestamp(cutoff)])
    d = pd.concat(rows)
    y = np.where(d["fwd"] > 1.5, 2, np.where(d["fwd"] < -1.5, 0, 1))
    pred = d[["P_DOWN", "P_NEUTRAL", "P_UP"]].values.argmax(axis=1)
    return {
        "test_windows": int(len(d)),
        "class_dist_test": np.bincount(y, minlength=3).tolist(),
        "accuracy": round(float((pred == y).mean()), 4),
        "majority_baseline": round(float((y == np.bincount(y).argmax()).mean()), 4),
        "avg_3d_ret_when_up_pred": round(float(d.loc[pred == 2, "fwd"].mean()), 3) if (pred == 2).any() else None,
        "avg_3d_ret_all": round(float(d["fwd"].mean()), 3),
    }


def train_all(phase: str, cutoff: str, out_dir: Path, report: dict, stride: int, epochs: int):
    import joblib
    from app.services.ai_predictor import ai_engine
    from app.services.crypto_engine import crypto_engine
    from app.services.intraday_universe import INTRADAY_UNIVERSE
    from app.services.tree_ensemble import export_forest

    out_dir.mkdir(parents=True, exist_ok=True)
    asof = cutoff if phase == "oos" else None
    res = report.setdefault(phase, {"cutoff": asof, "models_dir": str(out_dir)})

    t0 = time.time()
    train_data = load_equity(INTRADAY_UNIVERSE, asof=asof)
    res["equity_symbols"] = len(train_data)
    log.info(f"[{phase}] equity symbols loaded: {len(train_data)}")

    # ── RF ──
    rf = ai_engine.train_model_rf(train_data)
    joblib.dump(rf, out_dir / "astra_rf.joblib")
    export_forest(rf, str(out_dir / "astra_rf_trees.npz"))
    res["rf"] = {"oob_r2": round(float(rf.oob_score_), 4), "seconds": round(time.time() - t0)}
    if phase == "oos":
        full = load_equity(INTRADAY_UNIVERSE, asof=None)
        res["rf"]["oos"] = rf_oos_metrics(ai_engine, rf, full, cutoff)
    log.info(f"[{phase}] RF: {res['rf']}")

    # ── Equity LSTM ──
    t1 = time.time()
    model, scalers = ai_engine.train_model_lstm(train_data, lookback=30, target_days=20,
                                                stride=stride, epochs=epochs)
    if model is not None:
        model.save(out_dir / "astra_lstm_equity.keras")
        joblib.dump(scalers, out_dir / "astra_lstm_equity_scaler.joblib")
        res["lstm_equity"] = {"symbols": len(scalers), "seconds": round(time.time() - t1),
                              "epochs_run": len(model.history.history.get("loss", [])) if model.history else None,
                              "best_val_acc": round(max(model.history.history.get("val_accuracy", [0])), 4) if model.history else None}
        if phase == "oos":
            res["lstm_equity"]["oos"] = lstm_oos_metrics(model, scalers, full, cutoff, ai_engine)
    else:
        res["lstm_equity"] = {"error": "training failed"}
    log.info(f"[{phase}] LSTM equity: {res['lstm_equity']}")

    # ── Crypto LSTM ──
    t2 = time.time()
    if asof:
        os.environ["ASTRA_OFFLINE_ASOF"] = asof
    cm, cs = crypto_engine.train_lstm_crypto(symbols=CRYPTO, lookback=20, limit=4000)
    os.environ.pop("ASTRA_OFFLINE_ASOF", None)
    if cm is not None:
        cm.save(out_dir / "astra_lstm_crypto.keras")
        joblib.dump(cs, out_dir / "astra_lstm_crypto_scaler.joblib")
        res["lstm_crypto"] = {"seconds": round(time.time() - t2),
                              "best_val_acc": round(max(cm.history.history.get("val_accuracy", [0])), 4) if cm.history else None}
        if phase == "oos":
            res["lstm_crypto"]["oos"] = crypto_oos_metrics(cm, cs, cutoff, crypto_engine)
    else:
        res["lstm_crypto"] = {"error": "training failed"}
    log.info(f"[{phase}] LSTM crypto: {res['lstm_crypto']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2024-06-30")
    ap.add_argument("--phases", default="oos,prod")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if not os.getenv("ASTRA_OFFLINE_DATA_DIR"):
        sys.exit("Set ASTRA_OFFLINE_DATA_DIR to the exported CSV folder")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rep_path = out / "training_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    for phase in args.phases.split(","):
        d = BACKEND / "models_oos" if phase == "oos" else BACKEND / "app" / "models" / "saved_models"
        train_all(phase, args.cutoff, d, report, args.stride, args.epochs)
        rep_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
