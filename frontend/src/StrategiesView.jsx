/**
 * Strategy Marketplace
 * =====================
 * Lists all registered ASTRA engines via /strategies, lets the user trigger
 * a backtest, and renders the honest evaluator report (verdict, base/stress
 * stats, regime stability).
 *
 * This view is the visible payoff of:
 *   - Phase 1 strategy framework + registry
 *   - Strategy evaluator (cost stress + regime buckets)
 *   - /strategies API
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  Brain, Play, Loader, CheckCircle, AlertTriangle, XCircle,
  TrendingUp, TrendingDown, Activity, Layers, RefreshCw
} from 'lucide-react';
import { apiFetch } from './auth';


function VerdictBadge({ verdict }) {
  const map = {
    PASS:     { color: '#10b981', bg: 'rgba(16,185,129,0.15)', Icon: CheckCircle,   label: 'PASS' },
    MARGINAL: { color: '#f59e0b', bg: 'rgba(245,158,11,0.15)', Icon: AlertTriangle, label: 'MARGINAL' },
    FAIL:     { color: '#ef4444', bg: 'rgba(239,68,68,0.15)',  Icon: XCircle,       label: 'FAIL' },
  };
  const cfg = map[verdict] || map.FAIL;
  const { Icon } = cfg;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '6px',
      padding: '4px 10px', borderRadius: '6px',
      background: cfg.bg, color: cfg.color,
      fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.04em',
    }}>
      <Icon size={12} /> {cfg.label}
    </span>
  );
}


function StatCard({ label, value, hint, accent }) {
  return (
    <div style={{
      flex: 1, minWidth: '130px',
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: '8px', padding: '12px 14px',
    }}>
      <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.1rem', fontWeight: 600, color: accent || 'rgba(255,255,255,0.92)' }}>
        {value}
      </div>
      {hint && <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.35)', marginTop: '2px' }}>{hint}</div>}
    </div>
  );
}


function RegimeBuckets({ buckets }) {
  if (!buckets || buckets.length === 0) return null;
  const max = Math.max(...buckets.map(b => Math.abs(b)));
  return (
    <div style={{ display: 'flex', gap: '4px', alignItems: 'flex-end', height: '40px' }}>
      {buckets.map((v, i) => {
        const h = max ? Math.max(4, Math.abs(v) / max * 40) : 4;
        return (
          <div key={i} title={`Q${i + 1}: ₹${v.toLocaleString()}`} style={{
            flex: 1, height: `${h}px`,
            background: v > 0 ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)',
            borderRadius: '2px', minWidth: '8px',
          }} />
        );
      })}
    </div>
  );
}


function StrategyCard({ strategy, onBacktest, runningName, result }) {
  const isRunning = runningName === strategy.name;
  const hasResult = result && result.strategy === strategy.name;

  return (
    <div style={{
      background: 'rgba(255,255,255,0.025)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: '12px', padding: '20px',
      marginBottom: '14px',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px', gap: '14px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <span style={{ fontSize: '1.05rem', fontWeight: 700, color: 'rgba(255,255,255,0.95)' }}>
              {strategy.name}
            </span>
            <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)', fontFamily: 'monospace' }}>
              v{strategy.version}
            </span>
            <span style={{
              fontSize: '0.65rem', padding: '2px 8px', borderRadius: '4px',
              background: 'rgba(99,102,241,0.15)', color: '#a5b4fc',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              {strategy.timeframe}
            </span>
            {hasResult && <VerdictBadge verdict={result.verdict} />}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.55)', lineHeight: 1.45 }}>
            {strategy.description}
          </div>
          {strategy.backtest && (
            <div style={{ marginTop: 10, fontSize: '0.78rem', color: 'rgba(255,255,255,0.7)' }}>
              <div>
                <b style={{ color: strategy.backtest.recommended ? '#22c55e' : '#ef4444' }}>
                  {strategy.backtest.recommended ? 'Verified backtest' : 'Not recommended'}
                </b>
                {' '}· 2017–2026: {strategy.backtest.cagr_pct}%/yr · max drawdown {strategy.backtest.max_drawdown_pct}% · Sharpe {strategy.backtest.sharpe}
                {strategy.backtest.verdict_note ? ` — ${strategy.backtest.verdict_note}` : ''}
              </div>
              {strategy.backtest.tuning_note && (
                <div style={{ marginTop: 4, color: 'rgba(255,255,255,0.5)' }}>{strategy.backtest.tuning_note}</div>
              )}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
                {Object.entries(strategy.backtest.yearly_return_pct || {}).map(([y, v]) => (
                  <span key={y} style={{ fontSize: '0.68rem', padding: '2px 6px', borderRadius: 4,
                    background: v >= 0 ? 'rgba(34,197,94,.12)' : 'rgba(239,68,68,.12)', color: v >= 0 ? '#86efac' : '#fca5a5' }}>
                    {y}: {v > 0 ? '+' : ''}{v}%
                  </span>
                ))}
              </div>
            </div>
          )}
          <div style={{ marginTop: '8px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {(strategy.tags || []).map(t => (
              <span key={t} style={{
                fontSize: '0.65rem', padding: '2px 8px', borderRadius: '999px',
                background: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.55)',
              }}>{t}</span>
            ))}
          </div>
        </div>

        <button
          onClick={() => onBacktest(strategy.name)}
          disabled={!!runningName}
          style={{
            padding: '10px 16px', borderRadius: '8px', cursor: runningName ? 'not-allowed' : 'pointer',
            background: isRunning ? 'rgba(99,102,241,0.2)' : 'var(--color-primary, #6366f1)',
            color: 'white', border: 'none', fontWeight: 600, fontSize: '0.85rem',
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            opacity: (runningName && !isRunning) ? 0.4 : 1,
            transition: 'opacity 0.15s',
          }}
        >
          {isRunning ? <Loader size={14} className="spin" /> : <Play size={14} />}
          {isRunning ? 'Running…' : 'Run Backtest'}
        </button>
      </div>

      {/* Results */}
      {hasResult && (
        <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '12px' }}>
            <StatCard label="Trades" value={result.base_stats.n_trades} />
            <StatCard label="Win Rate" value={`${result.base_stats.win_rate}%`} />
            <StatCard label="R:R" value={result.base_stats.rr_ratio.toFixed(2)} />
            <StatCard
              label="Expectancy"
              value={`${result.base_stats.expectancy >= 0 ? '+' : ''}${result.base_stats.expectancy}%`}
              accent={result.base_stats.expectancy >= 0 ? '#10b981' : '#ef4444'}
            />
            <StatCard label="Profit Factor" value={result.base_stats.profit_factor.toFixed(2)} />
            <StatCard
              label="Total P&L"
              value={`₹${result.base_stats.total_pnl_rs.toLocaleString()}`}
              accent={result.base_stats.total_pnl_rs >= 0 ? '#10b981' : '#ef4444'}
            />
            <StatCard label="Avg Hold" value={`${Math.round(result.base_stats.avg_days_held)}d`} />
            <StatCard label="Regime Pass" value={`${Math.round(result.regime_pass_rate * 100)}%`} hint={`${result.regime_buckets.length} quarters`} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>
                Regime Stability — 8 quarters
              </div>
              <RegimeBuckets buckets={result.regime_buckets} />
              <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)', marginTop: '6px' }}>
                Each bar = one quarter's P&L. Green = profitable, red = losing.
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>
                Cost Stress Test
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'rgba(255,255,255,0.55)' }}>1.5× cost</span>
                  <span style={{ color: result.stress_stats.expectancy >= 0 ? '#10b981' : '#ef4444' }}>
                    {result.stress_stats.expectancy >= 0 ? '+' : ''}{result.stress_stats.expectancy}%
                    {' '}exp · PF {result.stress_stats.profit_factor.toFixed(2)}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'rgba(255,255,255,0.55)' }}>2.0× cost</span>
                  <span style={{ color: result.extreme_stats.expectancy >= 0 ? '#10b981' : '#ef4444' }}>
                    {result.extreme_stats.expectancy >= 0 ? '+' : ''}{result.extreme_stats.expectancy}%
                    {' '}exp · PF {result.extreme_stats.profit_factor.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {result.reasons && result.reasons.length > 0 && (
            <div style={{ marginTop: '14px', padding: '10px 14px', borderRadius: '8px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)' }}>
              <div style={{ fontSize: '0.72rem', color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px', fontWeight: 600 }}>
                Verdict reasons
              </div>
              <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '0.82rem', color: 'rgba(255,255,255,0.7)' }}>
                {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}

          <div style={{ marginTop: '10px', fontSize: '0.72rem', color: 'rgba(255,255,255,0.35)' }}>
            Evaluation completed in {result.elapsed_sec}s · cost model: 0.25% round-trip + 0.05% slippage per side
          </div>
        </div>
      )}
    </div>
  );
}


export default function StrategiesView() {
  const [strategies, setStrategies]   = useState([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState('');
  const [runningName, setRunningName] = useState(null);
  const [results, setResults]         = useState({});  // {strategy_name → report}
  const [summary, setSummary]         = useState(null);  // verified multi-year backtests
  const [runError, setRunError]       = useState('');

  useEffect(() => {
    apiFetch('/strategies/backtests/summary').then(r => (r.ok ? r.json() : null)).then(setSummary).catch(() => {});
  }, []);

  const fetchStrategies = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const r = await apiFetch('/strategies');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setStrategies(data.strategies || []);
    } catch (e) {
      setError(e.message || 'Failed to load strategies');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchStrategies(); }, [fetchStrategies]);

  const runBacktest = useCallback(async (name) => {
    setRunningName(name);
    setRunError('');
    try {
      const r = await apiFetch(`/strategies/${encodeURIComponent(name)}/backtest`, { method: 'POST' });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || `HTTP ${r.status}`);
      }
      const data = await r.json();
      setResults(prev => ({ ...prev, [name]: data }));
    } catch (e) {
      setRunError(`Backtest failed for ${name}: ${e.message}`);
    } finally {
      setRunningName(null);
    }
  }, []);

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1200px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Layers size={22} /> Strategy Marketplace
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.5)', margin: '6px 0 0', fontSize: '0.9rem' }}>
            First-party ASTRA engines. Each runs through the same honest evaluator
            (walk-forward + cost stress + 8-quarter regime stability).
            “Run backtest” re-tests the last 2 years live; the verified multi-year results are shown on each card.
          </p>
        </div>
        <button onClick={fetchStrategies} disabled={loading} style={{
          padding: '8px 14px', borderRadius: '8px', cursor: 'pointer',
          background: 'rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.85)',
          border: '1px solid rgba(255,255,255,0.08)', fontSize: '0.8rem',
          display: 'inline-flex', alignItems: 'center', gap: '6px',
        }}>
          <RefreshCw size={13} className={loading ? 'spin' : ''} /> Reload
        </button>
      </div>

      {summary?.combined_rules_book && (
        <div style={{ marginTop: 20, padding: 16, borderRadius: 12, background: 'rgba(34,197,94,.06)', border: '1px solid rgba(34,197,94,.25)', fontSize: '0.82rem', lineHeight: 1.6 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Verified backtests (paper) — what to expect</div>
          <div><b>Recommended book</b> ({(summary.combined_rules_book.sleeves || []).map(x => x.replace('ASTRA.', '')).join(' + ')}, equal capital): {summary.combined_rules_book.cagr_pct}%/yr 2017–2026,
            max drawdown {summary.combined_rules_book.max_drawdown_pct}% · NIFTY 50: {summary.benchmarks?.NIFTY50?.cagr_pct}%/yr, max drawdown {summary.benchmarks?.NIFTY50?.max_drawdown_pct}%.</div>
          {summary.intraday && <div><b>Intraday</b> — {summary.intraday.verdict_note}</div>}
          {summary.astra_ai_v2 && <div><b>ASTRA.AI v2</b> — out-of-sample Jul 2024–Oct 2026: {summary.astra_ai_v2.test_2024_07_to_2026_10?.total_return_pct}% total
            (NIFTY {summary.astra_ai_v2.nifty_same_window_pct}%, equal-weight universe {summary.astra_ai_v2.equal_weight_same_window_pct}% over the same window); walk-forward years: {Object.entries(summary.astra_ai_v2.walk_forward_years || {}).map(([y, v]) => `${y} ${v > 0 ? '+' : ''}${v}%`).join(' · ')}.</div>}
          <div style={{ color: 'rgba(255,255,255,0.5)', marginTop: 6, fontSize: '0.74rem' }}>{summary.method}</div>
        </div>
      )}
      {runError && <div role="alert" style={{ marginTop: 12, color: '#ef4444', fontSize: '0.85rem' }}>{runError}</div>}

      <div style={{ marginTop: '24px' }}>
        {loading && (
          <div style={{ padding: '40px', textAlign: 'center', color: 'rgba(255,255,255,0.4)' }}>
            <Loader size={20} className="spin" /> Loading strategies…
          </div>
        )}

        {error && (
          <div style={{ padding: '14px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', color: '#fca5a5' }}>
            Error: {error}
          </div>
        )}

        {!loading && !error && strategies.length === 0 && (
          <div style={{ padding: '32px', textAlign: 'center', color: 'rgba(255,255,255,0.4)' }}>
            No strategies registered.
          </div>
        )}

        {strategies.map(s => (
          <StrategyCard
            key={s.name}
            strategy={s}
            onBacktest={runBacktest}
            runningName={runningName}
            result={results[s.name]}
          />
        ))}

        {runningName && (
          <div style={{
            marginTop: '12px', padding: '10px 14px',
            background: 'rgba(99,102,241,0.08)',
            border: '1px solid rgba(99,102,241,0.2)',
            borderRadius: '8px', color: '#a5b4fc', fontSize: '0.82rem',
          }}>
            ⏳ Backtest in progress — this can take 30-90 seconds (fetches 2y data for 86 symbols, runs walk-forward, cost stress, regime analysis)
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  );
}
