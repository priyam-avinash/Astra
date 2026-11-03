"""Rule strategies filtered by the ASTRA.AI v2.1 ensemble rank (out-of-sample scores).
A rule trade is kept only if, on its signal day (the bar before entry), the stock's
ensemble rank percentile was >= thr. thr is chosen on 2019-01 → 2024-04, reported on
the untouched 2024-07 → window."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "scripts"))
import pandas as pd
from research.run_research import OUT
from research.compare_models import add_scores
from portfolio_backtest import load_universe, generate_trades, run_portfolio, make_strategy, _naive
from app.services.intraday_universe import INTRADAY_UNIVERSE
from app.services import strategy_evaluator as ev

SEL = ("2019-01-01", "2024-04-30"); TEST = ("2024-07-01", None)

def main(names):
    ev.EVAL_WINDOW["history_period"] = "10y"
    ds = add_scores(pd.read_pickle(OUT / "oos.pkl").reset_index(drop=True))
    ds["ens_tb"] = (ds["r_et_tb15"] + ds["r_et_tb20"]) / 2
    ds["p"] = ds.groupby("date")["ens_tb"].rank(pct=True)
    pct = {(s, d): v for s, d, v in zip(ds["symbol"], ds["date"], ds["p"])}
    dates = sorted(ds["date"].unique())
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    data = load_universe(INTRADAY_UNIVERSE)
    out = {}
    for name in names:
        trades = generate_trades(make_strategy(name), data, SEL[0], None)
        def keep(t, thr):
            d = prev.get(pd.Timestamp(_naive(t.entry_ts)).normalize())
            v = pct.get((t.symbol.replace(".NS", ""), d))
            return v is not None and v >= thr
        rows = {}
        for thr in (0.0, 0.3, 0.5, 0.7):
            tr = [t for t in trades if thr == 0.0 or keep(t, thr)]
            r = {}
            for lab, (a, b) in (("select", SEL), ("test", TEST)):
                sub = [t for t in tr if _naive(t.entry_ts) >= pd.Timestamp(a) and (b is None or _naive(t.entry_ts) <= pd.Timestamp(b))]
                res = run_portfolio(sub, data, a, b, max_pos=10); res.pop("_curve", None)
                r[lab] = {k: res.get(k) for k in ("cagr_pct", "max_drawdown_pct", "sharpe", "trades_taken", "yearly_return_pct")}
            rows[thr] = r
            print(name, thr, json.dumps(r), flush=True)
        out[name] = rows
    (OUT / "hybrid_study.json").write_text(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main(sys.argv[1:] or ["ASTRA.STAGE2", "ASTRA.PULLBACK", "ASTRA.MOMENTUM", "ASTRA.QUALITY"])
