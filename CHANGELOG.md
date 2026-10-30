# Changelog

Every change ships as its own version (semantic versioning: MAJOR.MINOR.PATCH).
Each release is tagged in git (`vX.Y.Z`) and the running version is shown in the
app sidebar and at `GET /health`. Paper trading only, in every version.

## 1.14.8 — 2026-10-30
- ASTRA.CRYPTO altcoins/memes: breakouts only while BTC is above its 200-day average (3x ATR stop, 4.5x target, 20-day limit) - untouched 2024-2026 test improved from -10.3%/yr (-70% drawdown) to +0.6%/yr (-47%); majors keep the regime switch (+9.2%/yr on the test; the tuned variant was worse)

## 1.14.7 — 2026-10-30
- Tuned rule strategies (chosen on 2017-2024, kept only where the untouched 2024-2026 test also improved): STAGE2 12.0 -> 15.4%/yr, PULLBACK 7.9 -> 10.2%/yr, QUALITY -3.1 -> +1.0%/yr; MOMENTUM unchanged (tuning hurt the test); recommended book is now STAGE2 + PULLBACK: 13.1%/yr, max drawdown -16.7% vs NIFTY 9.8%/yr, -38.4%

## 1.14.6 — 2026-10-30
- Mutual funds: funds with no NAV for over 7 days (closed or merged) don't accept orders; pending orders that get no NAV within 10 days are cancelled and reserved units returned; a NAV that only appears long after the order date is not used

## 1.14.5 — 2026-10-30
- ASTRA.AI v2: stocks whose last bar trails the freshest bar by more than 4 days (suspended, feed gap, stale cache) are no longer ranked as if their old prices were today's

## 1.14.4 — 2026-10-30
- Fractional crypto quantities (e.g. 0.0025 BTC) for paper orders and positions; stocks still trade in whole shares; Postgres column migrated

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
