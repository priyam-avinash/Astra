import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, Activity, DollarSign, Wallet } from 'lucide-react';

const mockPortfolio = [
  { id: '1', asset: 'RELIANCE', type: 'Equity', avgPrice: 2800.50, ltp: 2855.10, qty: 100 },
  { id: '2', asset: 'TCS', type: 'Equity', avgPrice: 3950.00, ltp: 3890.25, qty: 50 },
  { id: '3', asset: 'NIFTY 24MAY 22500 CE', type: 'F&O', avgPrice: 135.00, ltp: 165.50, qty: 500 },
  { id: '4', asset: 'HDFCBANK', type: 'Equity', avgPrice: 1420.00, ltp: 1445.60, qty: 200 },
  { id: '5', asset: 'Gold USD', type: 'Commodity', avgPrice: 2150.00, ltp: 2210.00, qty: 10 },
];

const histData = [
  { d: 'Mar 1', v: 1180000 }, { d: 'Mar 5', v: 1195000 }, { d: 'Mar 10', v: 1170000 },
  { d: 'Mar 15', v: 1220000 }, { d: 'Mar 20', v: 1210000 }, { d: 'Today', v: 1245670 }
];

// ── Components ──────────────────────────────────────────────────────────────

const DonutChart = ({ data }) => {
  const total = data.reduce((s, d) => s + d.v, 0);
  let cumulative = 0;
  const colors = ['#3b82f6', '#10b981', '#fbbf24', '#f87171', '#a78bfa'];

  return (
    <svg viewBox="0 0 100 100" width="160" height="160">
      {data.map((d, i) => {
        const start = (cumulative / total) * 2 * Math.PI;
        cumulative += d.v;
        const end = (cumulative / total) * 2 * Math.PI;
        const x1 = 50 + 40 * Math.cos(start - Math.PI/2);
        const y1 = 50 + 40 * Math.sin(start - Math.PI/2);
        const x2 = 50 + 40 * Math.cos(end - Math.PI/2);
        const y2 = 50 + 40 * Math.sin(end - Math.PI/2);
        const largeArc = end - start > Math.PI ? 1 : 0;
        return (
          <path key={i} d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`} fill={colors[i%colors.length]} stroke="var(--panel-bg)" strokeWidth="1" />
        );
      })}
      <circle cx="50" cy="50" r="28" fill="var(--panel-bg)" />
      <text x="50" y="54" fontSize="8" fontWeight="bold" fill="rgba(255,255,255,0.4)" textAnchor="middle">HOLDINGS</text>
    </svg>
  );
};

const AreaChart = ({ data, width=400, height=120 }) => {
  if(!data.length) return null;
  const min = Math.min(...data.map(d=>d.v)) * 0.99;
  const max = Math.max(...data.map(d=>d.v)) * 1.01;
  const range = max - min;
  const points = data.map((d, i) => `${(i / (data.length - 1)) * width},${height - ((d.v - min) / range) * height}`).join(' ');
  const areaPoints = `${points} ${width},${height} 0,${height}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id="gradP" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round" />
      <polygon points={areaPoints} fill="url(#gradP)" />
      {data.map((d, i) => (
        <circle key={i} cx={(i / (data.length - 1)) * width} cy={height - ((d.v - min) / range) * height} r="3" fill="#3b82f6" />
      ))}
    </svg>
  );
};

function Stat({ label, val, color, icon }) {
  return (
    <div className="glass-panel" style={{ padding: '16px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'rgba(255,255,255,0.3)', marginBottom: 8, fontSize: '0.7rem', letterSpacing: '0.05em' }}>
        {icon} {label.toUpperCase()}
      </div>
      <div style={{ fontSize: '1.4rem', fontWeight: 700, color }}>{val}</div>
    </div>
  );
}

// ── Main View ───────────────────────────────────────────────────────────────

export default function PortfolioView() {
  const totals = useMemo(() => {
    const value = mockPortfolio.reduce((acc, pos) => acc + (pos.ltp * pos.qty), 0);
    const cost = mockPortfolio.reduce((acc, pos) => acc + (pos.avgPrice * pos.qty), 0);
    const cats = mockPortfolio.reduce((acc, pos) => {
      acc[pos.type] = (acc[pos.type] || 0) + (pos.ltp * pos.qty);
      return acc;
    }, {});
    return { value, cost, pnl: value-cost, cats: Object.entries(cats).map(([k,v]) => ({k,v})) };
  }, []);

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="topbar">
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Portfolio Analytics</h2>
          <p style={{ color: 'rgba(255,255,255,0.4)', marginTop: '4px' }}>Visual insights into your capital allocation and performance.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <Stat label="Net Portfolio Value" val={`₹${totals.value.toLocaleString()}`} color="#fff" icon={<Wallet size={14}/>}/>
        <Stat label="Total Unrealized P&L" val={`+₹${totals.pnl.toLocaleString()}`} color="#10b981" icon={<TrendingUp size={14}/>}/>
        <Stat label="Realized Gains" val="₹42,850" color="#60a5fa" icon={<DollarSign size={14}/>}/>
        <Stat label="Day Return" val="+₹8,210 (0.6%)" color="#10b981" icon={<Activity size={14}/>}/>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
            <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>Performance History</h3>
            <span style={{ fontSize: '0.7rem', color: '#3b82f6', background: 'rgba(59,130,246,0.1)', padding: '2px 8px', borderRadius: '4px' }}>MONTHLY GROWTH</span>
          </div>
          <AreaChart data={histData} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', color: 'rgba(255,255,255,0.2)', fontSize: '0.65rem' }}>
            {histData.map(d=><span key={d.d}>{d.d}</span>)}
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '32px' }}>
          <DonutChart data={totals.cats} />
          <div style={{ flex: 1 }}>
            <h3 style={{ margin: '0 0 20px 0', fontSize: '0.9rem', fontWeight: 600 }}>Asset Allocation</h3>
            {totals.cats.map((c, i) => (
              <div key={c.k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '0.8rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'rgba(255,255,255,0.5)' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '2px', background: ['#3b82f6', '#10b981', '#fbbf24', '#f87171'][i%4] }}/>
                  {c.k}
                </span>
                <span style={{ fontWeight: 600, color: '#fff' }}>{((c.v / totals.value) * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.05)', fontWeight: 600, fontSize: '0.9rem' }}>Current Holdings</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.01)', color: 'rgba(255,255,255,0.3)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              <th style={{ padding: '14px 24px' }}>Asset</th>
              <th style={{ padding: '14px 24px' }}>Qty</th>
              <th style={{ padding: '14px 24px' }}>Avg Price</th>
              <th style={{ padding: '14px 24px' }}>LTP</th>
              <th style={{ padding: '14px 24px' }}>Value</th>
              <th style={{ padding: '14px 24px', textAlign: 'right' }}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {mockPortfolio.map((p) => {
              const pnl = (p.ltp - p.avgPrice) * p.qty;
              return (
                <tr key={p.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '0.85rem' }} className="table-row-hover">
                  <td style={{ padding: '16px 24px', fontWeight: 600 }}>{p.asset}</td>
                  <td style={{ padding: '16px 24px' }}>{p.qty}</td>
                  <td style={{ padding: '16px 24px', color: 'rgba(255,255,255,0.4)' }}>₹{p.avgPrice.toFixed(2)}</td>
                  <td style={{ padding: '16px 24px' }}>₹{p.ltp.toFixed(2)}</td>
                  <td style={{ padding: '16px 24px' }}>₹{(p.ltp * p.qty).toLocaleString()}</td>
                  <td style={{ padding: '16px 24px', textAlign: 'right', color: pnl>=0?'#10b981':'#ef4444', fontWeight: 700 }}>
                    {pnl>=0?'+':''}₹{pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <style>{`
        .table-row-hover:hover {
          background: rgba(255,255,255,0.02);
        }
      `}</style>
    </div>
  );
}
