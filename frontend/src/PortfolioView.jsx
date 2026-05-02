import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, TrendingDown, Activity, DollarSign, Wallet, RefreshCw, ChevronDown, ChevronUp, Brain } from 'lucide-react';

const API = 'http://localhost:8000';
const fmt = (n) => `₹${Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
const fmtPct = (n) => `${n >= 0 ? '+' : ''}${Number(n || 0).toFixed(2)}%`;

// ── Trade History Table with expandable ASTRA reflections ────────────────────
function TradeHistoryTable({ history }) {
  const [expanded, setExpanded] = useState({});
  const toggle = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }));

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
      <thead>
        <tr style={{ background: 'var(--bg-elevated)' }}>
          {['Asset', 'Action', 'Qty', 'Price', 'P&L', 'Status', 'Time', ''].map((h, i) => (
            <th key={i} style={{ padding: '10px 20px', fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {history.slice(0, 20).map((h, i) => {
          const pnl = h.pnl || 0;
          const isOpen = expanded[i];
          const hasReflection = !!h.reflection;
          return (
            <React.Fragment key={i}>
              <tr style={{ borderBottom: isOpen ? 'none' : '1px solid var(--border-subtle)', fontSize: 13, background: isOpen ? 'rgba(59,130,246,0.04)' : undefined }}>
                <td style={{ padding: '12px 20px', fontWeight: 600 }}>{h.asset?.replace('.NS', '').split(' ')[0]}</td>
                <td style={{ padding: '12px 20px' }}>
                  <span className={`badge badge-${h.action?.toLowerCase().split(' ')[0]}`}>{h.action}</span>
                </td>
                <td style={{ padding: '12px 20px', fontVariantNumeric: 'tabular-nums' }}>{h.quantity?.toLocaleString()}</td>
                <td style={{ padding: '12px 20px', fontVariantNumeric: 'tabular-nums' }}>₹{Number(h.price || 0).toFixed(2)}</td>
                <td style={{ padding: '12px 20px', fontWeight: 700, fontVariantNumeric: 'tabular-nums', color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {pnl !== 0 ? `${pnl >= 0 ? '+' : ''}${fmt(pnl)}` : '—'}
                </td>
                <td style={{ padding: '12px 20px', fontSize: 11, color: 'var(--text-tertiary)' }}>{h.status}</td>
                <td style={{ padding: '12px 20px', fontSize: 11, color: 'var(--text-tertiary)' }}>{h.time || h.timestamp || '—'}</td>
                <td style={{ padding: '12px 20px' }}>
                  {hasReflection && (
                    <button
                      onClick={() => toggle(i)}
                      title="ASTRA Reflection"
                      style={{ display: 'flex', alignItems: 'center', gap: 4, background: isOpen ? 'rgba(59,130,246,0.15)' : 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 11, fontWeight: 600, color: isOpen ? '#3b82f6' : 'var(--text-secondary)', whiteSpace: 'nowrap' }}
                    >
                      <Brain size={11} /> {isOpen ? <ChevronUp size={11}/> : <ChevronDown size={11}/>}
                    </button>
                  )}
                </td>
              </tr>
              {isOpen && hasReflection && (
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td colSpan={8} style={{ padding: '0 20px 16px 20px' }}>
                    <div style={{ background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: 8, padding: '12px 16px', display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                      <Brain size={14} color="#3b82f6" style={{ flexShrink: 0, marginTop: 1 }} />
                      <div>
                        <div style={{ fontSize: 10, fontWeight: 700, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>ASTRA Reflection</div>
                        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{h.reflection}</div>
                        {h.outcome_return_pct != null && (
                          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 6 }}>
                            Return: <span style={{ color: h.outcome_return_pct >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>{fmtPct(h.outcome_return_pct)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

// ── Charts ─────────────────────────────────────────────────────────────────
const DonutChart = ({ data }) => {
  const total = data.reduce((s, d) => s + d.v, 0) || 1;
  const colors = ['#3b82f6', '#10d87a', '#fbbf24', '#f43f5e', '#a78bfa', '#f97316'];
  let cumulative = 0;
  return (
    <svg viewBox="0 0 100 100" width="150" height="150">
      {data.map((d, i) => {
        const start = (cumulative / total) * 2 * Math.PI;
        cumulative += d.v;
        const end = (cumulative / total) * 2 * Math.PI;
        if (end - start < 0.01) return null;
        const x1 = 50 + 40 * Math.cos(start - Math.PI / 2);
        const y1 = 50 + 40 * Math.sin(start - Math.PI / 2);
        const x2 = 50 + 40 * Math.cos(end - Math.PI / 2);
        const y2 = 50 + 40 * Math.sin(end - Math.PI / 2);
        const largeArc = end - start > Math.PI ? 1 : 0;
        return (
          <path key={i}
            d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
            fill={colors[i % colors.length]}
            stroke="var(--bg-base)" strokeWidth="1.5"
          />
        );
      })}
      <circle cx="50" cy="50" r="26" fill="var(--bg-elevated)" />
      <text x="50" y="53" fontSize="7" fontWeight="700" fill="var(--text-tertiary)" textAnchor="middle">PORTFOLIO</text>
    </svg>
  );
};

const MiniPnlChart = ({ points }) => {
  if (!points || points.length < 2) return (
    <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-tertiary)', fontSize: 12 }}>
      No history yet
    </div>
  );
  const W = 400, H = 80;
  const vals = points.map(p => p.v);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = max - min || 1;
  const coords = points.map((p, i) =>
    `${(i / (points.length - 1)) * W},${H - ((p.v - min) / range) * H * 0.9 - H * 0.05}`
  ).join(' ');
  const isUp = vals[vals.length - 1] >= vals[0];
  const color = isUp ? '#10d87a' : '#f43f5e';
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`${coords} ${W},${H} 0,${H}`} fill="url(#pnlGrad)" />
      <polyline points={coords} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {points.map((p, i) => (
        <circle key={i} cx={(i / (points.length - 1)) * W} cy={H - ((p.v - min) / range) * H * 0.9 - H * 0.05}
          r="3" fill={color} />
      ))}
    </svg>
  );
};

// ── Skeleton row ────────────────────────────────────────────────────────────
const SkeletonRow = () => (
  <tr>
    {[...Array(7)].map((_, i) => (
      <td key={i} style={{ padding: '14px 20px' }}>
        <div className="skeleton" style={{ height: 14, width: i === 0 ? '80%' : '60%', borderRadius: 6 }} />
      </td>
    ))}
  </tr>
);

// ── Stat card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, icon, color }) {
  return (
    <div className="card" style={{ padding: '18px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-tertiary)', marginBottom: 10, fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ── Main ────────────────────────────────────────────────────────────────────
export default function PortfolioView() {
  const [positions, setPositions]   = useState([]);
  const [history,   setHistory]     = useState([]);
  const [loading,   setLoading]     = useState(true);
  const [lastFetch, setLastFetch]   = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [posRes, histRes] = await Promise.all([
        fetch(`${API}/api/positions`),
        fetch(`${API}/api/history`),
      ]);
      if (posRes.ok)  { const d = await posRes.json();  setPositions(d.positions || []); }
      if (histRes.ok) { const d = await histRes.json(); setHistory(d.history || []); }
      setLastFetch(new Date());
    } catch (e) {
      console.error('Portfolio fetch failed:', e);
    }
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, []);

  // ── Derived stats ──────────────────────────────────────────────────────
  const stats = useMemo(() => {
    const openValue     = positions.reduce((s, p) => s + (p.current_price || p.entry_price) * p.quantity, 0);
    const openCost      = positions.reduce((s, p) => s + p.entry_price * p.quantity, 0);
    const unrealizedPnl = positions.reduce((s, p) => s + (p.pnl || 0), 0);
    const closedTrades  = history.filter(h => h.pnl !== undefined && h.pnl !== null);
    const realizedPnl   = closedTrades.reduce((s, h) => s + (h.pnl || 0), 0);
    const winCount      = closedTrades.filter(h => h.pnl > 0).length;
    const winRate       = closedTrades.length ? ((winCount / closedTrades.length) * 100).toFixed(0) : 0;

    // Allocation donut — group open positions by direction
    const alloc = positions.reduce((acc, p) => {
      const key = p.direction === 'BUY' ? 'Long' : 'Short';
      acc[key] = (acc[key] || 0) + (p.current_price || p.entry_price) * p.quantity;
      return acc;
    }, {});
    const allocData = Object.entries(alloc).map(([k, v]) => ({ k, v }));

    // P&L sparkline from history (cumulative)
    let cum = 0;
    const pnlSeries = closedTrades.slice(-8).map(h => {
      cum += (h.pnl || 0);
      return { v: cum, label: h.asset };
    });

    return { openValue, openCost, unrealizedPnl, realizedPnl, winRate, allocData, pnlSeries, totalTrades: history.length };
  }, [positions, history]);

  const noData = !loading && positions.length === 0 && history.length === 0;

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Header */}
      <div className="topbar">
        <div>
          <h2 className="page-title">My Portfolio</h2>
          <p className="page-subtitle">
            Paper trading · {positions.length} open position{positions.length !== 1 ? 's' : ''}
            {lastFetch && <span style={{ color: 'var(--text-tertiary)' }}> · updated {lastFetch.toLocaleTimeString('en-IN')}</span>}
          </p>
        </div>
        <button className="btn-secondary" onClick={fetchData} disabled={loading} style={{ gap: 6 }}>
          <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 14 }}>
        <StatCard label="Open Position Value" value={fmt(stats.openValue)}
          sub={`Cost basis ${fmt(stats.openCost)}`} icon={<Wallet size={12} />} />
        <StatCard label="Unrealized P&L" value={fmt(stats.unrealizedPnl)}
          sub={stats.openCost > 0 ? fmtPct((stats.unrealizedPnl / stats.openCost) * 100) : '—'}
          icon={<TrendingUp size={12} />}
          color={stats.unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)'} />
        <StatCard label="Realized Gains" value={fmt(stats.realizedPnl)}
          sub={`${stats.totalTrades} total trades`}
          icon={<DollarSign size={12} />}
          color={stats.realizedPnl >= 0 ? 'var(--green)' : 'var(--red)'} />
        <StatCard label="Win Rate" value={`${stats.winRate}%`}
          sub={`${history.filter(h => h.pnl > 0).length}W / ${history.filter(h => h.pnl < 0).length}L`}
          icon={<Activity size={12} />}
          color={stats.winRate >= 50 ? 'var(--green)' : 'var(--red)'} />
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16 }}>
        <div className="card" style={{ padding: '20px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Cumulative P&L</span>
            <span style={{ fontSize: 11, color: 'var(--accent)', background: 'var(--accent-dim)', padding: '2px 8px', borderRadius: 6 }}>PAPER</span>
          </div>
          <MiniPnlChart points={stats.pnlSeries} />
          <div style={{ display: 'flex', gap: 16, marginTop: 12, justifyContent: 'flex-end' }}>
            {stats.pnlSeries.slice(-3).map((p, i) => (
              <span key={i} style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{p.label?.replace('.NS', '') || `T-${i+1}`}</span>
            ))}
          </div>
        </div>

        <div className="card" style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', gap: 24 }}>
          <DonutChart data={stats.allocData.length ? stats.allocData : [{ k: 'Empty', v: 1 }]} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Allocation</div>
            {stats.allocData.length > 0 ? stats.allocData.map((c, i) => {
              const colors = ['#3b82f6', '#10d87a', '#fbbf24', '#f43f5e'];
              return (
                <div key={c.k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10, fontSize: 12 }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)' }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: colors[i % 4], flexShrink: 0 }} />
                    {c.k}
                  </span>
                  <span style={{ fontWeight: 700 }}>{((c.v / (stats.openValue || 1)) * 100).toFixed(0)}%</span>
                </div>
              );
            }) : (
              <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>No open positions</div>
            )}
          </div>
        </div>
      </div>

      {/* Empty state */}
      {noData && (
        <div className="card" style={{ padding: '60px 32px', textAlign: 'center' }}>
          <Wallet size={36} style={{ opacity: 0.2, margin: '0 auto 16px' }} />
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No positions yet</div>
          <div style={{ fontSize: 13, color: 'var(--text-tertiary)', maxWidth: 320, margin: '0 auto' }}>
            Approve signals in Auto Mode to open paper trades. They'll appear here with live P&L tracking.
          </div>
        </div>
      )}

      {/* Open Positions table */}
      {(loading || positions.length > 0) && (
        <div className="card" style={{ overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Open Positions</span>
            <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{positions.length} active</span>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'var(--bg-elevated)' }}>
                {['Asset', 'Direction', 'Qty', 'Entry', 'LTP', 'Value', 'P&L'].map(h => (
                  <th key={h} style={{ padding: '10px 20px', fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? [...Array(3)].map((_, i) => <SkeletonRow key={i} />) :
                positions.map(p => {
                  const ltp   = p.current_price || p.entry_price;
                  const value = ltp * p.quantity;
                  const pnl   = p.pnl || 0;
                  const pnlPct = p.pnl_pct || 0;
                  const isBuy = p.direction === 'BUY';
                  return (
                    <tr key={p.id} style={{ borderBottom: '1px solid var(--border-subtle)', fontSize: 13 }}>
                      <td style={{ padding: '14px 20px', fontWeight: 700 }}>
                        {p.asset?.replace('.NS', '').replace('.BO', '')}
                        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>{p.asset?.includes('.NS') ? 'NSE' : p.asset?.includes('=F') ? 'Commodity' : 'Other'}</div>
                      </td>
                      <td style={{ padding: '14px 20px' }}>
                        <span className={`badge badge-${p.direction?.toLowerCase()}`}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          {isBuy ? <TrendingUp size={9} /> : <TrendingDown size={9} />}
                          {p.direction}
                        </span>
                      </td>
                      <td style={{ padding: '14px 20px', fontVariantNumeric: 'tabular-nums' }}>{p.quantity?.toLocaleString()}</td>
                      <td style={{ padding: '14px 20px', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>₹{p.entry_price?.toFixed(2)}</td>
                      <td style={{ padding: '14px 20px', fontVariantNumeric: 'tabular-nums' }}>₹{ltp?.toFixed(2)}</td>
                      <td style={{ padding: '14px 20px', fontVariantNumeric: 'tabular-nums' }}>{fmt(value)}</td>
                      <td style={{ padding: '14px 20px', textAlign: 'right', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                        color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {pnl >= 0 ? '+' : ''}{fmt(pnl)}
                        <div style={{ fontSize: 10, fontWeight: 400, opacity: 0.7 }}>{fmtPct(pnlPct)}</div>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade History table */}
      {history.length > 0 && (
        <div className="card" style={{ overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Closed Trades</span>
            <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{history.length} trades</span>
          </div>
          <TradeHistoryTable history={history} />
        </div>
      )}
    </div>
  );
}
