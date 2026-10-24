"""
Train the production ASTRA.AI v2 model with the validation-selected
configuration on ALL available data and export it for the app:

  app/models/saved_models/astra_ai_v2_trees.npz   numpy tree ensemble (no sklearn at runtime)
  app/models/saved_models/astra_ai_v2.json        features, trading rules, validation/test metrics

  ASTRA_OFFLINE_DATA_DIR=... python research/export_model.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from research.run_research import LABELS, OUT, make_model, load_ds   # noqa: E402
from app.services.features_v2 import feature_columns                  # noqa: E402
from app.services.tree_ensemble import export_forest, NumpyForestRegressor  # noqa: E402


def main():
    rep = json.loads((OUT / "research_trade.json").read_text())
    wf_path = OUT / "walkforward_tb15_et.json"
    if wf_path.exists():                       # prefer the multi-fold (walk-forward) selection
        wf = json.loads(wf_path.read_text())
        g = wf["selected"]["config"]
        lname, algo = wf["label"], wf["algo"]
        sel = {"valid": {"walk_forward_folds": wf["selected"]["folds"],
                         "mean_sharpe": wf["selected"]["mean_sharpe"]},
               "test": {"static_model": wf["test_static"], "rolling_refit": wf["test_rolling"]}}
    else:
        sel = rep["selected"]
        cand, g = sel["key"].split("|", 1)
        g = json.loads(g)
        lname, algo = cand.split("/")
    col, _kind = LABELS[lname]
    ds = load_ds()
    feats = feature_columns()
    d = ds.dropna(subset=feats + [col])
    model = make_model("reg", algo)
    model.fit(d[feats].values, d[col].values.astype(float))
    models_dir = BACKEND / "app" / "models" / "saved_models"
    npz = models_dir / "astra_ai_v2_trees.npz"
    export_forest(model, str(npz))
    # parity check: numpy export == sklearn
    X = d[feats].values[-500:]
    assert np.allclose(NumpyForestRegressor.load(str(npz)).predict(X), model.predict(X), atol=1e-9)
    cfg = {
        "version": "2.0.0", "label": lname, "label_column": col, "algo": algo,
        "features": feats, "trained_until": str(d["date"].max().date()), "exported_on": str(date.today()),
        "train_rows": int(len(d)),
        "q": g["q"], "regime": g["regime"], "max_pos": g["max_pos"],
        "stop_k": g["stop_k"], "tgt_k": g["tgt_k"], "max_bars": g["max_bars"],
        "validation": sel["valid"], "test": sel["test"],
        "benchmark_nifty": rep.get("nifty"),
        "benchmark_equal_weight": json.loads((OUT / "benchmark_equal_weight.json").read_text())
                                  if (OUT / "benchmark_equal_weight.json").exists() else None,
        "notes": "Rules chosen by 5-fold walk-forward validation (2019–2024-04); test 2024-07→ is "
                 "out-of-sample. Production model refit on all data. Backtests use today's large-cap "
                 "universe (survivorship bias) and daily bars; past results don't guarantee future returns.",
    }
    (models_dir / "astra_ai_v2.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(json.dumps({k: cfg[k] for k in ("label", "algo", "trained_until", "q", "regime", "stop_k", "tgt_k", "max_bars")}))
    print("npz bytes:", npz.stat().st_size)


if __name__ == "__main__":
    main()
