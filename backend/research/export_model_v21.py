"""
Export ASTRA.AI v2.1 — the two-model ranking ensemble with rotation rules.

  Models   ExtraTrees on the 1.5/3.0-ATR (tb15) and 2.0/4.0-ATR (tb20) triple-barrier
           labels; the live score is the mean of each model's percentile rank in
           today's universe.
  Rules    (chosen on the 2019–2024-04 walk-forward folds by research/live_rules_study.py)
           hold up to `top_n` stocks; buy from today's top `top_n`; after `min_hold`
           trading days sell once the stock's rank drops below `keep_pct`;
           catastrophic stop at entry − stop_k × ATR; no fixed target.
  Output   app/models/saved_models/astra_ai_v2_trees.npz      (tb15)
           app/models/saved_models/astra_ai_v2_tb20_trees.npz (tb20)
           app/models/saved_models/astra_ai_v2.json

  ASTRA_OFFLINE_DATA_DIR=... python research/export_model_v21.py
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

from research.run_research import OUT, load_ds   # noqa: E402
from app.services.features_v2 import feature_columns   # noqa: E402
from app.services.tree_ensemble import export_forest, NumpyForestRegressor   # noqa: E402

MODELS = [("tb15", "tb15_30_20_win", "astra_ai_v2_trees.npz"),
          ("tb20", "tb20_40_40_win", "astra_ai_v2_tb20_trees.npz")]


def main():
    from sklearn.ensemble import ExtraTreesRegressor
    study = json.loads((OUT / "live_rules_study.json").read_text())["best"]
    kw = study["kw"]
    ds = load_ds()
    feats = feature_columns()
    models_dir = BACKEND / "app" / "models" / "saved_models"
    out_models, rows, until = [], 0, None
    for label, col, fname in MODELS:
        d = ds.dropna(subset=feats + [col])
        m = ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=100,
                                max_features=0.5, n_jobs=-1, random_state=42)
        m.fit(d[feats].values, d[col].values.astype(float))
        path = models_dir / fname
        export_forest(m, str(path))
        X = d[feats].values[-500:]
        assert np.allclose(NumpyForestRegressor.load(str(path)).predict(X), m.predict(X), atol=1e-9)
        out_models.append({"label": label, "label_column": col, "file": fname})
        rows, until = len(d), str(d["date"].max().date())
        print(label, "exported", path.stat().st_size, "bytes", flush=True)
    cfg = {
        "version": "2.1.0", "rule": "rotation", "algo": "et", "label": "tb15+tb20",
        "models": out_models, "features": feats, "trained_until": until,
        "exported_on": str(date.today()), "train_rows": rows,
        "top_n": kw["top"], "max_pos": kw["slots"], "min_hold_days": kw["min_hold"],
        "keep_pct": kw["keep_pct"], "stop_k": kw["stop_k"], "tgt_k": None, "regime": False,
        "validation": {"walk_forward_fold_returns_pct": study["fold_ret"], "mean_sharpe": study["mean_sharpe"],
                       "folds": ["2019", "2020", "2021", "2022", "2023-01→2024-04"]},
        "test": {"rotation_ensemble": study["test"], "same_rules_single_model_tb15": study["test_et_tb15_same_rules"]},
        "full_oos_2019_on": study["full_oos"],
        "notes": "Scores are walk-forward out-of-sample (refit yearly to 2023, then every 6 months, "
                 "training data ending 61 days before each segment). Rules chosen on 2019–2024-04 folds only; "
                 "the Jul 2024 → test window was never used for choosing. The universe is today's large caps "
                 "(survivorship bias inflates 2019–2023). Past results don't guarantee future returns.",
    }
    (models_dir / "astra_ai_v2.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(json.dumps({k: cfg[k] for k in ("version", "trained_until", "top_n", "min_hold_days", "keep_pct", "stop_k")}))


if __name__ == "__main__":
    main()
