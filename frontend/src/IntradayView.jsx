import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Timer, TrendingUp, TrendingDown, Minus, RefreshCw,
  Activity, BarChart2, Zap, AlertTriangle, CheckCircle,
  ArrowUpRight, ArrowDownRight, Clock, Target, Shield,
  ChevronDown, ChevronUp, Play, Search
} from 'lucide-react';

const API = 'http://localhost:8000';

const QUICK_SYMBOLS = [
  'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
  'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
  'HINDUNILVR', 'ITC', 'WIPRO', 'HCLTECH', 'BAJFINANCE',
];

function isMarketHours() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  const h = ist.getHours(), m = ist.getMinutes();
  const mins = h * 60 + m;
  return mins >= 555 && mins <= 930; // 09:15–15:30
}

function getSessionLabel() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  const h = ist.getHours(), m = ist.getMinutes();
  const mins = h * 60 + m;
  if (mins < 540)  return { label: 'Pre-Market', color: '#f59e0b' };
  if (mins < 555)  return { label: 'Opening',    color: '#3b82f6' };
  if (mins <= 930) return { label: 'Market Open',color: '#10b981' };
  return { label: 'After Hours', color: '#6b7280' };
}

// ── Signal badge ──────────────────────────────────────────────────────────────

function SignalBadge({ signal, confidence, strategy }) {
  if (!signal || signal === 'HOLD') {
    return (
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        padding: '3px 10px', borderRadius: 6,
        background: 'rgba(107,114,128,0.15)', color: '#9ca3af',
        fontSize: 11, fontWeight: 700, letterSpacing: '0.06em',
      }}>
        <Minus size={10} /> HOLD
      </span>
    );
  }
  const isBuy = signal === 'BUY';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        padding: '3px 12px', borderRadius: 6, fontSize: 12, fontWeight: 800,
        letterSpacing: '0.08em',
        background: isBuy ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
        color: isBuy ? '#10b981' : '#ef4444',
      }}>
        {isBuy ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
        {signal}
      </span>
      {confidence > 0 && (
        <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontWeight: 600 }}>
          {confidence}% · {strategy}
        </span>
      )}
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: 'var(--card-bg)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '14px 16px', minWidth: 120,
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 800, color: color || 'var(--text-primary)' }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ── Strategy row in backtest results ─────────────────────────────────────────

function StrategyRow({ name, stats }) {
  if (!stats) return null;
  const pnlColor = stats.total_pnl_rs >= 0 ? '#10b981' : '#ef4444';
  return (
    <tr style={{ borderBottom: '1px solid var(--border)' }}>
      <td style={{ padding: '10px 12px', fontWeight: 700, fontSize: 12 }}>{name}</td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'center' }}>{stats.trades}</td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'center', color: '#10b981' }}>{stats.wins}</td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'center', color: '#ef4444' }}>{stats.losses}</td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'center', fontWeight: 700 }}>
        {stats.win_rate_pct != null ? `${stats.win_rate_pct}%` : '—'}
      </td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'right', fontWeight: 700, color: pnlColor }}>
        {stats.total_pnl_rs != null ? `₹${stats.total_pnl_rs.toLocaleString()}` : '—'}
      </td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'center' }}>
        {stats.sharpe != null ? stats.sharpe : '—'}
      </td>
      <td style={{ padding: '10px 12px', fontSize: 12, textAlign: 'right', color: '#ef4444' }}>
        {stats.max_drawdown_rs != null ? `₹${Math.abs(stats.max_drawdown_rs).toLocaleString()}` : '—'}
      </td>
    </tr>
  );
}

// ── Equity curve mini-chart (SVG) ─────────────────────────────────────────────

function EquityCurve({ data }) {
  if (!data || data.length < 2) return null;
  const W = 320, H = 80, PAD = 8;
  const values = data.map(d => d.cum_pnl_rs);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = PAD + (i / (values.length - 1)) * (W - PAD * 2);
    const y = H - PAD - ((v - min) / range) * (H - PAD * 2);
    return `${x},${y}`;
  }).join(' ');
  const zeroY = H - PAD - ((0 - min) / range) * (H - PAD * 2);
  const finalPnl = values[values.length - 1];
  const lineColor = finalPnl >= 0 ? '#10b981' : '#ef4444';
  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY}
        stroke="rgba(255,255,255,0.1)" strokeWidth={1} strokeDasharray="3,3" />
      <polyline points={pts} fill="none" stroke={lineColor} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  );
}

// ── Main view ─────────────────────────────────────────────────────────────────

export default function IntradayView() {
  const [symbol, setSymbol]             = useState('RELIANCE');
  const [inputVal, setInputVal]         = useState('RELIANCE');
  const [signal, setSignal]             = useState(null);
  const [signalLoading, setSignalLoading] = useState(false);
  const [scanResults, setScanResults]   = useState([]);
  const [scanLoading, setScanLoading]   = useState(false);
  const [backtest, setBacktest]         = useState(null);
  const [btLoading, setBtLoading]       = useState(false);
  const [btDays, setBtDays]             = useState(30);
  const [showTradeLog, setShowTradeLog] = useState(false);
  const [error, setError]               = useState('');
  const abortRef = useRef(null);

  const session = getSessionLabel();

  // ── Fetch single-symbol signal ──────────────────────────────────────────────

  const fetchSignal = useCallback(async (sym) => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setSignalLoading(true);
    setError('');
    try {
      const r = await fetch(`${API}/api/intraday/${sym}`, { signal: abortRef.current.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setSignal(data);
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message);
    } finally {
      setSignalLoading(false);
    }
  }, []);

  // ── Scan NIFTY 50 ─────────────────────────────────────────────────────────

  const fetchScan = useCallback(async () => {
    setScanLoading(true);
    setError('');
    try {
      const r = await fetch(`${API}/api/intraday/scan/top?top_k=15`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setScanResults(data.signals || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setScanLoading(false);
    }
  }, []);

  // ── Run backtest ──────────────────────────────────────────────────────────

  const runBacktest = useCallback(async (sym, days) => {
    setBtLoading(true);
    setBacktest(null);
    setError('');
    try {
      const r = await fetch(`${API}/api/intraday/backtest/${sym}?days=${days}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setBacktest(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setBtLoading(false);
    }
  }, []);

  // On mount / symbol change
  useEffect(() => {
    fetchSignal(symbol);
    fetchScan();
  }, [symbol]); // eslint-disable-line

  // Auto-refresh during market hours
  useEffect(() => {
    if (!isMarketHours()) return;
    const id = setInterval(() => fetchSignal(symbol), 30_000);
    return () => clearInterval(id);
  }, [symbol, fetchSignal]);

  const handleSymbolSubmit = () => {
    const s = inputVal.trim().toUpperCase().replace('.NS', '');
    if (!s) return;
    setSymbol(s);
    setBacktest(null);
  };

  const entryPrice = signal?.entry_price;
  const sl         = signal?.sl;
  const tp         = signal?.tp;

  return (
    <div style={{ padding: '24px', maxWidth: 1100, margin: '0 auto' }}>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Timer size={22} color="var(--accent)" />
          <div>
            <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--text-primary)' }}>Intraday Signals</div>
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>ORB · EMA 5/13 Crossover · Momentum · Auto square-off 15:15</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            padding: '4px 12px', borderRadius: 20, fontSize: 11, fontWeight: 700,
            background: `${session.color}22`, color: session.color,
            border: `1px solid ${session.color}44`,
          }}>● {session.label}</span>
          {signal?.time_remaining_min > 0 && (
            <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
              <Clock size={10} style={{ marginRight: 3 }} />
              {signal.time_remaining_min}m left
            </span>
          )}
        </div>
      </div>

      {/* ── Symbol search ────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 0, flex: 1, minWidth: 200, maxWidth: 340 }}>
          <input
            value={inputVal}
            onChange={e => setInputVal(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleSymbolSubmit()}
            placeholder="NSE symbol…"
            style={{
              flex: 1, padding: '8px 12px', background: 'var(--input-bg)',
              border: '1px solid var(--border)', borderRight: 'none',
              borderRadius: '8px 0 0 8px', color: 'var(--text-primary)', fontSize: 13,
              outline: 'none',
            }}
          />
          <button onClick={handleSymbolSubmit} style={{
            padding: '8px 14px', background: 'var(--accent)', color: '#000',
            border: 'none', borderRadius: '0 8px 8px 0', cursor: 'pointer', fontWeight: 700,
          }}>
            <Search size={14} />
          </button>
        </div>

        {/* Quick picks */}
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {QUICK_SYMBOLS.slice(0, 8).map(s => (
            <button key={s} onClick={() => { setSymbol(s); setInputVal(s); setBacktest(null); }}
              style={{
                padding: '5px 10px', fontSize: 10, fontWeight: 700, borderRadius: 6, cursor: 'pointer',
                background: symbol === s ? 'var(--accent)' : 'var(--card-bg)',
                color: symbol === s ? '#000' : 'var(--text-secondary)',
                border: `1px solid ${symbol === s ? 'var(--accent)' : 'var(--border)'}`,
              }}>{s}</button>
          ))}
        </div>
      </div>

      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 8, marginBottom: 16, fontSize: 12, color: '#ef4444' }}>
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* ── Signal card ──────────────────────────────────────────────── */}
      <div style={{
        background: 'var(--card-bg)', border: '1px solid var(--border)',
        borderRadius: 12, padding: '20px 24px', marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Activity size={16} color="var(--accent)" />
            <span style={{ fontWeight: 800, fontSize: 16 }}>{symbol}</span>
            <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>· 15m Signal</span>
          </div>
          <button onClick={() => fetchSignal(symbol)} disabled={signalLoading}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
              background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8,
              cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)' }}>
            <RefreshCw size={12} style={{ animation: signalLoading ? 'spin 1s linear infinite' : 'none' }} />
            Refresh
          </button>
        </div>

        {signalLoading && !signal ? (
          <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-tertiary)', fontSize: 13 }}>
            Fetching 15m data…
          </div>
        ) : signal ? (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
              <StatCard label="Signal" value={<SignalBadge signal={signal.signal} confidence={signal.confidence} strategy={signal.strategy} />} />
              {entryPrice > 0 && <StatCard label="Entry" value={`₹${entryPrice.toLocaleString()}`} />}
              {sl > 0 && <StatCard label="Stop Loss" value={`₹${sl.toLocaleString()}`} color="#ef4444" sub={`−${((Math.abs(entryPrice - sl) / entryPrice) * 100).toFixed(2)}%`} />}
              {tp > 0 && <StatCard label="Target" value={`₹${tp.toLocaleString()}`} color="#10b981" sub={`+${((Math.abs(tp - entryPrice) / entryPrice) * 100).toFixed(2)}%`} />}
              {signal.time_remaining_min > 0 && <StatCard label="Time Left" value={`${signal.time_remaining_min}m`} />}
            </div>

            {/* Auto square-off warning */}
            {signal.auto_squareoff_active && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
                background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)',
                borderRadius: 8, fontSize: 11, color: '#f59e0b', marginBottom: 12 }}>
                <AlertTriangle size={13} />
                Auto square-off active — positions will close at {signal.squareoff_time} IST
              </div>
            )}

            {/* Strategy breakdown */}
            {signal.strategies_detail && Object.keys(signal.strategies_detail).length > 0 && (
              <div>
                <div style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Strategy Breakdown</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {Object.entries(signal.strategies_detail).map(([name, s]) => (
                    <div key={name} style={{
                      padding: '6px 12px', borderRadius: 8, fontSize: 11,
                      background: s.signal === 'BUY' ? 'rgba(16,185,129,0.1)' : s.signal === 'SELL' ? 'rgba(239,68,68,0.1)' : 'rgba(107,114,128,0.1)',
                      border: `1px solid ${s.signal === 'BUY' ? 'rgba(16,185,129,0.2)' : s.signal === 'SELL' ? 'rgba(239,68,68,0.2)' : 'rgba(107,114,128,0.2)'}`,
                      color: s.signal === 'BUY' ? '#10b981' : s.signal === 'SELL' ? '#ef4444' : '#9ca3af',
                    }}>
                      <span style={{ fontWeight: 700 }}>{name}</span>
                      <span style={{ margin: '0 4px' }}>·</span>
                      {s.signal}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
              {signal.trend_bias && signal.trend_bias !== 'NEUTRAL' && (
                <span style={{
                  padding: '2px 9px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                  background: signal.trend_bias === 'BULLISH' ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                  color: signal.trend_bias === 'BULLISH' ? '#10b981' : '#ef4444',
                  border: `1px solid ${signal.trend_bias === 'BULLISH' ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'}`,
                }}>
                  {signal.trend_bias === 'BULLISH' ? '↑' : '↓'} {signal.trend_bias} TREND (SMA50)
                </span>
              )}
              <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
                Signal at {signal.time_of_signal} IST · {signal.data_source || 'market_data'} · Paper mode
              </span>
            </div>
          </div>
        ) : null}
      </div>

      {/* ── Backtest section ──────────────────────────────────────────── */}
      <div style={{
        background: 'var(--card-bg)', border: '1px solid var(--border)',
        borderRadius: 12, padding: '20px 24px', marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart2 size={16} color="var(--accent)" />
            <span style={{ fontWeight: 800, fontSize: 16 }}>Backtest · {symbol}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <select value={btDays} onChange={e => setBtDays(Number(e.target.value))}
              style={{ padding: '5px 8px', background: 'var(--input-bg)', border: '1px solid var(--border)',
                borderRadius: 6, color: 'var(--text-primary)', fontSize: 12 }}>
              <option value={15}>15 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
            </select>
            <button onClick={() => runBacktest(symbol, btDays)} disabled={btLoading}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '7px 14px',
                background: 'var(--accent)', color: '#000', border: 'none', borderRadius: 8,
                cursor: btLoading ? 'not-allowed' : 'pointer', fontWeight: 700, fontSize: 12 }}>
              {btLoading ? <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={12} />}
              {btLoading ? 'Running…' : 'Run Backtest'}
            </button>
          </div>
        </div>

        {btLoading && (
          <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-tertiary)', fontSize: 13 }}>
            Fetching historical 15m data and running walk-forward simulation…
          </div>
        )}

        {backtest && !btLoading && (
          <div>
            {backtest.error ? (
              <div style={{ color: '#ef4444', fontSize: 13 }}>{backtest.error}</div>
            ) : (
              <>
                {/* Summary header */}
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
                  <StatCard label="Days Tested" value={backtest.trading_days_tested} sub={`${backtest.period_days}d requested`} />
                  <StatCard label="Total Bars" value={backtest.data_bars?.toLocaleString()} />
                  <StatCard label="Combined P&L"
                    value={`₹${backtest.strategies?.Combined?.total_pnl_rs?.toLocaleString() || 0}`}
                    color={backtest.strategies?.Combined?.total_pnl_rs >= 0 ? '#10b981' : '#ef4444'} />
                  <StatCard label="Combined Win Rate"
                    value={`${backtest.strategies?.Combined?.win_rate_pct || 0}%`}
                    color={backtest.strategies?.Combined?.win_rate_pct >= 50 ? '#10b981' : '#ef4444'} />
                  <StatCard label="Sharpe"
                    value={backtest.strategies?.Combined?.sharpe ?? '—'}
                    color={backtest.strategies?.Combined?.sharpe > 1 ? '#10b981' : 'var(--text-primary)'} />
                </div>

                {/* Equity curve */}
                {backtest.equity_curve?.length > 1 && (
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Combined Equity Curve</div>
                    <EquityCurve data={backtest.equity_curve} />
                  </div>
                )}

                {/* Strategy table */}
                <div style={{ overflowX: 'auto', marginBottom: 12 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Strategy','Trades','Wins','Losses','Win Rate','Total P&L','Sharpe','Max DD'].map(h => (
                          <th key={h} style={{ padding: '8px 12px', textAlign: h === 'Strategy' ? 'left' : 'center',
                            color: 'var(--text-tertiary)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {['ORB','EMA_Cross','Momentum','Combined'].map(name => (
                        <StrategyRow key={name} name={name} stats={backtest.strategies?.[name]} />
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Trade log toggle */}
                {backtest.trade_log?.length > 0 && (
                  <>
                    <button onClick={() => setShowTradeLog(p => !p)}
                      style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
                        background: 'none', border: '1px solid var(--border)', borderRadius: 6,
                        cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
                      {showTradeLog ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      {showTradeLog ? 'Hide' : 'Show'} Trade Log ({backtest.trade_log.length} trades)
                    </button>

                    {showTradeLog && (
                      <div style={{ maxHeight: 280, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 8 }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                          <thead style={{ position: 'sticky', top: 0, background: 'var(--card-bg)' }}>
                            <tr style={{ borderBottom: '1px solid var(--border)' }}>
                              {['Date','Strategy','Signal','Entry','Exit','Outcome','P&L %','P&L ₹'].map(h => (
                                <th key={h} style={{ padding: '6px 10px', textAlign: 'left',
                                  color: 'var(--text-tertiary)', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {backtest.trade_log.map((t, i) => (
                              <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                                <td style={{ padding: '5px 10px' }}>{t.date}</td>
                                <td style={{ padding: '5px 10px' }}>{t.strategy}</td>
                                <td style={{ padding: '5px 10px', color: t.signal === 'BUY' ? '#10b981' : '#ef4444', fontWeight: 700 }}>{t.signal}</td>
                                <td style={{ padding: '5px 10px' }}>₹{t.entry_price?.toFixed(1)}</td>
                                <td style={{ padding: '5px 10px' }}>₹{t.exit_price?.toFixed(1)}</td>
                                <td style={{ padding: '5px 10px', color: t.outcome === 'TP' ? '#10b981' : t.outcome === 'SL' ? '#ef4444' : '#f59e0b', fontWeight: 700 }}>{t.outcome}</td>
                                <td style={{ padding: '5px 10px', color: t.pnl_pct >= 0 ? '#10b981' : '#ef4444' }}>{t.pnl_pct?.toFixed(2)}%</td>
                                <td style={{ padding: '5px 10px', color: t.pnl_rs >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>₹{t.pnl_rs?.toLocaleString()}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 8 }}>
                      ⚠ {backtest.note}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}

        {!backtest && !btLoading && (
          <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-tertiary)', fontSize: 13 }}>
            Select a symbol and click "Run Backtest" to simulate historical intraday P&L.
          </div>
        )}
      </div>

      {/* ── NIFTY 50 Scan ────────────────────────────────────────────── */}
      <div style={{ background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Zap size={16} color="var(--accent)" />
            <span style={{ fontWeight: 800, fontSize: 16 }}>NIFTY 50 Scan</span>
            {scanResults.length > 0 && (
              <span style={{ padding: '2px 8px', borderRadius: 10, background: 'rgba(99,102,241,0.15)',
                color: '#818cf8', fontSize: 10, fontWeight: 700 }}>{scanResults.length} signals</span>
            )}
          </div>
          <button onClick={fetchScan} disabled={scanLoading}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
              background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8,
              cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)' }}>
            <RefreshCw size={12} style={{ animation: scanLoading ? 'spin 1s linear infinite' : 'none' }} />
            Rescan
          </button>
        </div>

        {scanLoading ? (
          <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-tertiary)', fontSize: 13 }}>
            Scanning NIFTY 50 universe…
          </div>
        ) : scanResults.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-tertiary)', fontSize: 13 }}>
            No actionable signals right now — market may be outside prime signal window (09:45–14:30 IST).
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
            {scanResults.map((r, i) => (
              <div key={i}
                onClick={() => { setSymbol(r.symbol); setInputVal(r.symbol); setBacktest(null); }}
                style={{ padding: '12px 14px', borderRadius: 10, cursor: 'pointer',
                  background: 'var(--bg-secondary)',
                  border: `1px solid ${r.signal === 'BUY' ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'}`,
                  transition: 'transform 0.1s', ':hover': { transform: 'scale(1.01)' } }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontWeight: 800, fontSize: 13 }}>{r.symbol}</span>
                  <SignalBadge signal={r.signal} confidence={r.confidence} strategy={r.strategy} />
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-tertiary)' }}>
                  {r.entry_price > 0 && <span>Entry ₹{r.entry_price.toLocaleString()}</span>}
                  {r.tp > 0 && <span style={{ color: '#10b981' }}>TP ₹{r.tp.toLocaleString()}</span>}
                  {r.sl > 0 && <span style={{ color: '#ef4444' }}>SL ₹{r.sl.toLocaleString()}</span>}
                </div>
                <div style={{ fontSize: 9, color: 'var(--text-tertiary)', marginTop: 4 }}>
                  {r.strategy} · {r.confidence}% confidence
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Model Learning Card ──────────────────────────────────────── */}
      <ModelLearningCard />

    </div>
  );
}

// ── Self-learning status card ─────────────────────────────────────────────────

function ModelLearningCard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/model/learning-stats`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setStats(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const card = {
    background: 'var(--card-bg)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    padding: '16px 20px',
    marginTop: 20,
  };

  return (
    <div style={card}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <Activity size={15} color="var(--accent)" />
        <span style={{ fontWeight: 700, fontSize: 13 }}>Model Learning</span>
        <span style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 'auto' }}>
          Updates every Sunday 2am IST · run <code style={{ fontSize: 9, background: 'rgba(255,255,255,0.06)', padding: '1px 5px', borderRadius: 4 }}>python retrain_from_trades.py</code> manually
        </span>
      </div>

      {loading ? (
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Loading…</div>
      ) : stats === null ? (
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
          No learning data yet. Execute trades to start accumulating feedback.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
          {[
            { label: 'Labelled Trades',   value: stats.total_labelled ?? '—' },
            { label: 'Correct Signals',   value: stats.correct ?? '—',   color: '#10b981' },
            { label: 'Incorrect Signals', value: stats.incorrect ?? '—', color: '#ef4444' },
            { label: 'Accuracy',          value: stats.accuracy_pct != null ? `${stats.accuracy_pct}%` : '—', color: '#818cf8' },
            { label: 'Current RF OOB R²', value: stats.rf_oob != null ? stats.rf_oob.toFixed(3) : '—' },
            { label: 'Last Retrained',    value: stats.last_retrain ?? 'Never' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: '10px 14px' }}>
              <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
              <div style={{ fontWeight: 800, fontSize: 18, color: color || 'var(--text-primary)' }}>{value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
