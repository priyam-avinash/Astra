# ASTRA Hybrid Portfolio v2.0 — Design Spec
Date: 2026-05-18  
Goal: ≥5% monthly portfolio gain with positive expectancy and capital protection

---

## Context & Problem

All four ASTRA engines currently fail the evaluator gate. Best current result:

| Engine | WR | Exp/trade | PF |
|---|---|---|---|
| PULLBACK v1.3 | 32% | −0.31% | 0.90 |
| STAGE2 | 32% | −0.33% | 0.91 |
| MOMENTUM | 36% | −1.82% | 0.62 |
| QUALITY | 50% | −1.60% | 0.64 |

Root causes identified via diagnostic:
- 68% of PULLBACK losses exit via SMA50 breakdown (false recovery entries)
- NIFTY regime gate too strict (SMA150 → only passes 26% of days)
- Fixed % stop doesn't adapt to stock volatility
- No market-relative strength filter
- No portfolio-level capital allocation across strategies

---

## Approach

**Three-phase sequential build:**

1. **PULLBACK v2.0** — redesign with professional indicators (Supertrend + EMA + ATR + RS)
2. **ORB Intraday v2.0** — fix volume filter, add Supertrend 15m + ATR targets + day bias
3. **PortfolioAllocator** — rolling-Sharpe² dynamic allocation across both engines

Phases 1 and 2 must show positive expectancy individually before Phase 3 is deployed.

---

## Phase 1: PULLBACK v2.0

### Indicators

| Indicator | Params | Purpose |
|---|---|---|
| Supertrend | ATR(10), mult=3.0 | Primary trend gate |
| EMA | 20 / 50 / 200 | Alignment (EMA > SMA — less lag) |
| ATR | 14-period daily | Volatility-calibrated stop |
| RSI | 14-period | Momentum quality |
| Relative Strength | 20-day vs NIFTY | Market-beat filter |

### Entry — all 9 required

1. Supertrend(10,3) bullish (Close > Supertrend line)
2. EMA20 > EMA50 > EMA200
3. Close ∈ [EMA50, EMA50 × 1.08] — pullback zone
4. Close > EMA20 — short-term trend intact
5. RSI was < 40 in last 15 bars; now > 50 AND RSI[-1] > RSI[-2]
6. Bounce: Close > prev_Close × 1.003 AND High > prev_High
7. Volume > 1.5× 20-day avg
8. NIFTY Close > NIFTY EMA200
9. stock_return_20d > nifty_return_20d (RS positive)

### Exit — first triggered

| Trigger | Condition | Fill |
|---|---|---|
| Supertrend flip | Supertrend turns bearish | Next open |
| Target | Close ≥ EMA50 × 1.20 | Intrabar at target price |
| Hard stop | Close ≤ entry − 2×ATR(14) | Intrabar at stop price |
| Trail stop | peak × 0.93 breached, after +4% gain | Intrabar at trail level |
| Time stop | 40 trading days elapsed | Next open |

### Position Sizing

Half-Kelly from signal geometry, normalised to 0.10 reference:
```
stop_dist   = entry - hard_stop
target_dist = target - entry
rr          = target_dist / stop_dist
kelly_raw   = WIN_PROB - (1 - WIN_PROB) / rr   # WIN_PROB = 0.45
size_fraction = clamp(kelly_raw / 2 / 0.10, 0.5, 1.5)
```

### Key differences vs v1.3

| Parameter | v1.3 | v2.0 |
|---|---|---|
| Trend MA | SMA50/200 | EMA20/50/200 |
| Trend gate | None | Supertrend(10,3) required bullish |
| Stop | Fixed −8% | entry − 2×ATR(14) |
| Exit trigger | Close < SMA50×0.98 | Supertrend flip (dynamic) |
| Target | SMA50 × 1.15 | EMA50 × 1.20 |
| Trail | 10% from peak after +5% | 7% from peak after +4% |
| Market regime | NIFTY > SMA200 | NIFTY > EMA200 |
| RS filter | None | 20-day RS vs NIFTY > 0 |
| Sector filter | Disabled | Disabled |

### Expected outcome

- WR: 45–50% (Supertrend gate historically eliminates ~35% of false entries on NSE daily)
- Expectancy: +1.5–2.0% per trade
- Regime pass-rate: ≥60%

---

## Phase 2: ORB Intraday v2.0

### Changes vs current ORB

| Parameter | Current | v2.0 |
|---|---|---|
| Volume multiplier | 1.1× | 1.1× (keep) |
| Supertrend gate | None | Supertrend(7,3) on 15m must be bullish |
| Day-open filter | None | NIFTY must open above prev day close |
| Target | Fixed % | entry + 1.5 × ATR(14 on 15m) |
| Stop | Fixed % | entry − 0.75 × ATR(14 on 15m) |
| R:R floor | None | Only take trade if R:R ≥ 2.0 |
| Signal window | All day | 09:30–11:00 IST only |
| Max trades/symbol | 1 | 2 per day |

### Expected outcome

- ORB already has a structural edge on high-volatility opens
- ATR target + 2:1 R:R floor + Supertrend gate should push WR to 45%+
- Contribution: +1.0–1.5%/month on allocated capital

---

## Phase 3: PortfolioAllocator

### Mechanism

Runs weekly (every Monday pre-market). Re-weights capital between PULLBACK and ORB.

**Weight formula:**
```
sharpe_i = mean(net_pct_last_20_trades) / std(net_pct_last_20_trades)

if sharpe_i <= 0:
    weight_i = 0          # strategy paused
else:
    weight_i = sharpe_i² / Σ(sharpe_j² for j where sharpe_j > 0)
```

**Bootstrap:** Equal weight (50/50) until each strategy has ≥ 20 completed trades.

**Hard portfolio risk cap:**
```
total_open_risk = Σ (position_size × ATR_stop_pct)
if total_open_risk > 0.20 × portfolio_value:
    reject new entries until risk drops below cap
```

**No floor** — allocator can go 100% one strategy when the other has negative Sharpe.

### DB schema addition

```sql
CREATE TABLE strategy_performance (
    id          INTEGER PRIMARY KEY,
    strategy    TEXT NOT NULL,         -- "ASTRA.PULLBACK", "ASTRA.ORB", etc.
    entry_ts    DATETIME,
    exit_ts     DATETIME,
    net_pct     REAL,
    size_frac   REAL,
    user_id     INTEGER REFERENCES users(id)
);
```

### API additions

- `GET /portfolio/allocation` — current weights + rolling Sharpe per strategy
- `POST /portfolio/rebalance` — trigger manual rebalance (admin only)

---

## Target metrics after all 3 phases

| Metric | Now | Target |
|---|---|---|
| PULLBACK WR | 32% | 45–50% |
| PULLBACK exp/trade | −0.31% | +1.5–2.0% |
| ORB WR | failing | 45%+ |
| Portfolio monthly gain | ~−0.5% | **+5–6%** |
| Max monthly drawdown | uncontrolled | ≤ 8% (ATR cap) |

---

## Implementation order

1. `app/strategies/pullback.py` — v2.0 (Supertrend + EMA + ATR + RS)
2. `app/services/strategy_evaluator.py` — precompute Supertrend helper shared across strategies
3. `app/strategies/orb.py` — v2.0 (Supertrend 15m + ATR targets + day bias)
4. `app/models/database.py` — `strategy_performance` table
5. `app/services/portfolio_allocator.py` — Sharpe² allocator
6. `app/api/portfolio_router.py` — allocation + rebalance endpoints
7. Tests for each phase before moving to next

---

## Out of scope

- Options overlays
- Leverage
- Live order execution changes (paper trading lock stays)
- Frontend UI changes (allocator exposed via API only)
