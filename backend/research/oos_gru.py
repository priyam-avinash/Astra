"""
Out-of-sample scores from a small sequence model (GRU) — candidate to replace the
BiLSTM behind ASTRA.ML.

Input   last 60 trading days per stock: daily return, log volume ratio, distance from
        20/50-day averages, RSI/100, ATR%, 20-day return (each clipped & scaled with
        training-segment statistics only).
Target  the stock's cross-sectional percentile of its next-20-day return (from next open).
Train   walk-forward segments identical to oos_scores.py (data ending 61 days before
        each segment); 3 seeds averaged.
Output  column `gru` merged into scripts/out/oos.pkl.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

from research.run_research import OUT, load_ds   # noqa: E402
from research.oos_scores import segments          # noqa: E402

LOOK = 60
CH = ["r1", "lvr", "d20", "d50", "rsi", "atrp", "r20"]


def build_panel(ds):
    ds = ds.sort_values(["symbol", "date"]).copy()
    g = ds.groupby("symbol")
    ds["r1"] = g["Close"].pct_change()
    ds["lvr"] = np.log((ds["Volume"] + 1) / (g["Volume"].transform(lambda v: v.rolling(20).mean()) + 1))
    ds["d20"] = ds["Close"] / g["Close"].transform(lambda c: c.rolling(20).mean()) - 1
    ds["d50"] = ds["Close"] / g["Close"].transform(lambda c: c.rolling(50).mean()) - 1
    ds["rsi"] = ds["RSI"] / 100 - 0.5
    ds["atrp"] = ds["ATR"] / ds["Close"]
    ds["r20"] = g["Close"].pct_change(20)
    nxt = g["Open"].shift(-1)
    ds["y"] = g["Close"].shift(-20) / nxt - 1
    ds["y"] = ds.groupby("date")["y"].rank(pct=True)
    return ds


def windows(ds, idx_rows):
    """Return X[n, LOOK, C] for the given row indices (rows of a symbol-sorted frame)."""
    arr = ds[CH].values.astype(np.float32)
    sym = ds["symbol"].values
    X = np.zeros((len(idx_rows), LOOK, len(CH)), np.float32)
    ok = np.zeros(len(idx_rows), bool)
    for k, i in enumerate(idx_rows):
        j = i - LOOK + 1
        if j >= 0 and sym[j] == sym[i]:
            w = arr[j:i + 1]
            if not np.isnan(w).any():
                X[k] = w
                ok[k] = True
    return X, ok


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(len(CH), 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    ds = build_panel(load_ds()).reset_index(drop=True)
    ds["gru"] = np.nan
    dates = ds["date"]
    t0 = time.time()
    for ss, se in segments(ds["date"].max()):
        te = ss - pd.Timedelta(days=61)
        tr_rows = np.where((dates <= te) & ds["y"].notna() & (ds["date"].dt.dayofweek == 2))[0]   # weekly samples
        Xtr, ok = windows(ds, tr_rows)
        Xtr, ytr = Xtr[ok], ds["y"].values[tr_rows][ok].astype(np.float32)
        lo, hi = np.percentile(Xtr.reshape(-1, len(CH)), [1, 99], axis=0)
        mu, sd = Xtr.reshape(-1, len(CH)).mean(0), Xtr.reshape(-1, len(CH)).std(0) + 1e-6
        norm = lambda X: ((np.clip(X, lo, hi) - mu) / sd).astype(np.float32)   # noqa: E731
        Xtr = norm(Xtr)
        te_rows = np.where((dates >= ss) & (dates <= se))[0]
        Xte, okte = windows(ds, te_rows)
        Xte = norm(Xte)
        preds = []
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            net = Net()
            opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
            xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr - 0.5)
            for ep in range(6):
                perm = torch.randperm(len(xt))
                for b in range(0, len(xt), 512):
                    i = perm[b:b + 512]
                    opt.zero_grad()
                    loss = ((net(xt[i]) - yt[i]) ** 2).mean()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                preds.append(net(torch.from_numpy(Xte)).numpy())
        p = np.mean(preds, axis=0)
        p[~okte] = np.nan
        ds.loc[te_rows, "gru"] = p
        print(f"{ss.date()} trained on {len(Xtr)} windows ({time.time()-t0:.0f}s)", flush=True)
    full = pd.read_pickle(OUT / "oos.pkl")
    full = full.drop(columns=["gru"], errors="ignore").merge(ds[["date", "symbol", "gru"]], on=["date", "symbol"], how="left")
    full.to_pickle(OUT / "oos.pkl")
    print("merged")


if __name__ == "__main__":
    main()
