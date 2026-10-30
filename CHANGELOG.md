# Changelog

Every change ships as its own version (semantic versioning: MAJOR.MINOR.PATCH).
Each release is tagged in git (`vX.Y.Z`) and the running version is shown in the
app sidebar and at `GET /health`. Paper trading only, in every version.

## 1.14.3 — 2026-10-30
- Live prices: WebSocket sends the login token (required since the WebSocket auth fix), quote polling goes through the authenticated fetch helper

## 1.14.2 — 2026-10-30
- Intraday backtest made look-ahead free: next-bar-open entries, stop checked before target with gap-aware fills, trailing stop applies from the next bar, square-off at the 15:15 open, trend filter and ORB volume use only past bars, daily-reset VWAP (live + backtest), Combined picks the earliest signal, universe backtest no longer learns from its own test; result: no intraday strategy has an edge after costs (about -0.25% per trade), flagged experimental in the UI

## 1.14.1 — 2026-10-24
- Audit fixes: atomic position closes (no double booking), one-time signal execution, SL/TP re-anchored to fills, INR portfolio with live USD/INR, no fills on after-hours/stale quotes, NSE market-hours check, brokerage in P&L, IST trading day; mutual funds priced at the order date's NAV (pending until published), SIP double-booking fix; security: API keys scrubbed from errors, Upstox OAuth state + browser binding, WebSocket auth, owner-only settings/kill switch (persisted), safer registration, JWT typ claim, encrypted-credential key, Docker ports

## 1.14.0 — 2026-10-24
- ASTRA.AI v2: 47-feature ranking model (Extra Trees, target-before-stop label), walk-forward-selected rules,
  live universe snapshot scoring; default engine in scanners and AI Predictions.
- Retrained production models (RF v1, 3-class equity LSTM, 3-class crypto LSTM); LSTM engines marked experimental.
- Mutual funds: AMFI NAVs, returns/risk analytics, SIP calculator, paper lumpsum/SIP/redeem.
- Verified multi-year portfolio backtests shown on the Strategies page; ASTRA 1.0 and QUALITY flagged.
- Backtest engine: fixed a look-ahead bug (0% trailing stop exited at the bar high), gap-aware stops,
  out-of-sample trade windows; offline data mode for reproducible research.
- Pullback sector-regime filter; renamed tickers (ETERNAL, TMPV, LTM).
- Dashboard: mutual fund value, correct sign on negative live P&L.

## 1.13.x — 2026-09/10
- Unified market-data layer, paper-only broker, Vercel deploy with login, UI fixes.
