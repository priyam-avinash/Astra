import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Search, RefreshCw, TrendingUp, TrendingDown, ChevronRight, ZoomIn, ZoomOut, Maximize2, ExternalLink, DollarSign, BarChart3, Target, PieChart, Bitcoin } from 'lucide-react';
import { createChart, ColorType, CrosshairMode, LineStyle } from 'lightweight-charts';
import FullChartModal from './FullChartModal.jsx';

/* ── STYLES ── */
const S = {
  page: { height: '100%', width: '100%', display: 'flex', flexDirection: 'column', paddingTop: 64, paddingLeft: 24, paddingRight: 24, paddingBottom: 32, background: '#0f1118', color: '#e1e4ea', fontFamily: "'Inter', -apple-system, sans-serif", overflowY: 'auto', boxSizing: 'border-box' },
  inner: { maxWidth: 1120, width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 32 },
  searchRow: { display: 'flex', alignItems: 'center', gap: 12 },
  searchInput: { flex: 1, maxWidth: 420, padding: '10px 16px 10px 40px', borderRadius: 8, border: '1px solid #2a2e39', background: '#1e222d', color: '#fff', fontSize: 14, fontFamily: "'Inter', sans-serif", outline: 'none' },
  searchIcon: { position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: '#6b7280' },
  btn: { padding: '10px 22px', borderRadius: 8, border: 'none', cursor: 'pointer', background: '#2962ff', color: '#fff', fontWeight: 600, fontSize: 14, transition: 'background .2s' },
  avatar: { width: 56, height: 56, borderRadius: '50%', background: '#2962ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, fontWeight: 800, color: '#fff', flexShrink: 0 },
  tabBar: { display: 'flex', gap: 0, borderBottom: '1px solid #2a2e39', overflowX: 'auto' },
  tab: (a) => ({ padding: '10px 18px', fontSize: 14, fontWeight: a ? 600 : 400, cursor: 'pointer', borderBottom: a ? '2px solid #2962ff' : '2px solid transparent', color: a ? '#fff' : '#787b86', background: 'none', border: 'none', transition: 'all .2s', fontFamily: "'Inter', sans-serif", whiteSpace: 'nowrap' }),
  card: { borderRadius: 12, border: '1px solid #2a2e39', background: '#1e222d', padding: 24 },
  chartWrap: { borderRadius: 12, overflow: 'hidden', border: '1px solid #2a2e39', background: '#131722' },
  chartHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 16px', background: '#1e222d', borderBottom: '1px solid #2a2e39' },
  chartBtn: { padding: '4px 10px', borderRadius: 4, border: '1px solid #2a2e39', background: '#131722', color: '#787b86', cursor: 'pointer', fontSize: 13, display: 'inline-flex', alignItems: 'center', gap: 4 },
  kdGrid: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0 },
  kdItem: { padding: '16px 0' },
  kdLabel: { fontSize: 13, color: '#787b86', marginBottom: 4 },
  kdValue: { fontSize: 18, fontWeight: 600, color: '#fff' },
  secTitle: { fontSize: 22, fontWeight: 700, color: '#fff', margin: 0, display: 'flex', alignItems: 'center', gap: 8 },
  secSub: { fontSize: 13, color: '#787b86', marginTop: 4 },
  loadWrap: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 400, border: '1px solid #2a2e39', borderRadius: 12, background: '#131722' },
  errWrap: { padding: 40, borderRadius: 12, border: '1px solid #ef535033', background: '#ef53501a', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 },
  signalBadge: (s) => ({ display: 'inline-flex', padding: '6px 16px', borderRadius: 6, fontWeight: 700, fontSize: 14, background: s === 'BUY' ? 'rgba(38,166,154,.15)' : s === 'SELL' ? 'rgba(239,83,80,.15)' : 'rgba(120,123,134,.15)', color: s === 'BUY' ? '#26a69a' : s === 'SELL' ? '#ef5350' : '#787b86', border: `1px solid ${s === 'BUY' ? '#26a69a33' : s === 'SELL' ? '#ef535033' : '#787b8633'}` }),
  engineBar: { display: 'flex', gap: 4, background: '#1e222d', padding: 4, borderRadius: 8, width: 'fit-content' },
  engineTab: (a) => ({ padding: '8px 16px', borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: 'pointer', background: a ? '#2a2e39' : 'transparent', color: a ? '#fff' : '#787b86', border: 'none', fontFamily: "'Inter', sans-serif", transition: 'all .2s' }),
  gaugeGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, padding: '24px 0' },
};

/* ── GAUGE ── */
function Gauge({ value, label }) {
  const safe = isNaN(value) || value == null ? 50 : Math.min(Math.max(value, 0), 100);
  const angle = -90 + (safe / 100) * 180;
  let status, color;
  if (safe <= 30) { status = 'Strong sell'; color = '#ef5350'; }
  else if (safe <= 45) { status = 'Sell'; color = '#f77c80'; }
  else if (safe >= 70) { status = 'Strong buy'; color = '#26a69a'; }
  else if (safe >= 55) { status = 'Buy'; color = '#4dd0c8'; }
  else { status = 'Neutral'; color = '#787b86'; }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ color: '#787b86', fontSize: 14, fontWeight: 600, marginBottom: 12 }}>{label}</div>
      <div style={{ position: 'relative', width: 180, height: 90 }}>
        <svg viewBox="0 0 200 100" style={{ width: '100%', height: '100%' }}>
          <path d="M 20 95 A 80 80 0 0 1 180 95" fill="none" stroke="#2a2e39" strokeWidth="10" strokeLinecap="round"/>
          <path d="M 20 95 A 80 80 0 0 1 56 28" fill="none" stroke="#ef5350" strokeWidth="10" strokeLinecap="round" opacity=".7"/>
          <path d="M 144 28 A 80 80 0 0 1 180 95" fill="none" stroke="#26a69a" strokeWidth="10" strokeLinecap="round" opacity=".7"/>
          <path d="M 56 28 A 80 80 0 0 1 144 28" fill="none" stroke="#787b86" strokeWidth="10" strokeLinecap="round" opacity=".3"/>
          <g transform={`rotate(${angle} 100 95)`}><line x1="100" y1="95" x2="100" y2="25" stroke={color} strokeWidth="2.5" strokeLinecap="round"/><circle cx="100" cy="95" r="5" fill={color}/><circle cx="100" cy="95" r="2" fill="#131722"/></g>
        </svg>
      </div>
      <div style={{ color, fontSize: 15, fontWeight: 700, marginTop: 6 }}>{status}</div>
    </div>
  );
}

/* ── HELPERS ── */
const fmt = (v) => v != null ? Number(v).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';
const fmtBig = (v) => { if (!v) return '—'; if (v >= 1e12) return '₹' + (v / 1e12).toFixed(2) + 'T'; if (v >= 1e9) return '₹' + (v / 1e9).toFixed(2) + 'B'; if (v >= 1e7) return '₹' + (v / 1e7).toFixed(2) + 'Cr'; if (v >= 1e5) return '₹' + (v / 1e5).toFixed(2) + 'L'; return '₹' + v.toLocaleString('en-IN'); };
const fmtPct = (v) => v != null ? (v * 100).toFixed(2) + '%' : '—';
const fmtVol = (v) => { if (!v) return '0'; if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'; if (v >= 1e7) return (v / 1e7).toFixed(2) + 'Cr'; if (v >= 1e5) return (v / 1e5).toFixed(2) + 'L'; if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K'; return v.toString(); };

const KDRow = ({ label, value }) => <div style={S.kdItem}><div style={S.kdLabel}>{label}</div><div style={S.kdValue}>{value}</div></div>;

/* ── FINANCIALS TAB ── */
function FinancialsTab({ fin }) {
  if (!fin) return <div style={{ padding: 40, textAlign: 'center', color: '#787b86' }}>Loading financials…</div>;
  const k = fin.keyFacts || {}, g = fin.growth || {}, d = fin.dividends || {}, p = fin.profile || {}, f = fin.forecasts || {};
  const subTabs = ['Overview', 'Dividends', 'Valuation', 'Growth'];
  const [sub, setSub] = useState('Overview');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Sub-tab bar */}
      <div style={{ display: 'flex', gap: 8 }}>
        {subTabs.map(t => (
          <button key={t} onClick={() => setSub(t)} style={{ padding: '8px 16px', borderRadius: 20, border: sub === t ? '1px solid #2962ff' : '1px solid #2a2e39', background: sub === t ? '#2962ff22' : '#1e222d', color: sub === t ? '#2962ff' : '#787b86', cursor: 'pointer', fontSize: 13, fontWeight: 600, fontFamily: "'Inter', sans-serif" }}>{t}</button>
        ))}
      </div>

      {sub === 'Overview' && (
        <>
          {/* Key Facts */}
          <div>
            <h3 style={{ ...S.secTitle, fontSize: 18 }}>Key facts</h3>
            <div style={{ ...S.kdGrid, marginTop: 12, borderTop: '1px solid #2a2e39' }}>
              <KDRow label="Market capitalization" value={fmtBig(k.marketCap)} />
              <KDRow label="Dividend yield" value={fmtPct(k.dividendYield)} />
              <KDRow label="P/E ratio (TTM)" value={k.trailingPE?.toFixed(2) || '—'} />
              <KDRow label="EPS (TTM)" value={k.trailingEps ? '₹' + k.trailingEps.toFixed(2) : '—'} />
              <KDRow label="52-week high" value={'₹' + fmt(k.fiftyTwoWeekHigh)} />
              <KDRow label="52-week low" value={'₹' + fmt(k.fiftyTwoWeekLow)} />
              <KDRow label="Average volume" value={fmtVol(k.averageVolume)} />
              <KDRow label="Beta" value={k.beta?.toFixed(2) || '—'} />
            </div>
          </div>
          {/* About */}
          <div style={S.card}>
            <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 12 }}>About</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 32px', fontSize: 13 }}>
              <div><span style={{ color: '#787b86' }}>Sector: </span><span style={{ color: '#e1e4ea' }}>{p.sector || '—'}</span></div>
              <div><span style={{ color: '#787b86' }}>Industry: </span><span style={{ color: '#e1e4ea' }}>{p.industry || '—'}</span></div>
              <div><span style={{ color: '#787b86' }}>Employees: </span><span style={{ color: '#e1e4ea' }}>{p.employees ? Number(p.employees).toLocaleString() : '—'}</span></div>
              <div><span style={{ color: '#787b86' }}>Country: </span><span style={{ color: '#e1e4ea' }}>{p.country || '—'}</span></div>
              {p.website && <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#787b86' }}>Website: </span><a href={p.website} target="_blank" rel="noreferrer" style={{ color: '#2962ff', textDecoration: 'none' }}>{p.website} <ExternalLink size={11} style={{ verticalAlign: 'middle' }} /></a></div>}
            </div>
            {p.description && <p style={{ color: '#a0a4ab', fontSize: 13, lineHeight: 1.6, marginTop: 16, maxHeight: 120, overflow: 'hidden' }}>{p.description}</p>}
          </div>
          {/* Ownership & Capital */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            <div style={S.card}>
              <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 16 }}>Ownership</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                <div style={{ position: 'relative', width: 100, height: 100 }}>
                  <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#2a2e39" strokeWidth="3"/>
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#2962ff" strokeWidth="3" strokeDasharray={`${(k.heldPercentInsiders||0)*100} 100`} strokeLinecap="round"/>
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#26a69a" strokeWidth="3" strokeDasharray={`${(k.heldPercentInstitutions||0)*100} 100`} strokeDashoffset={`-${(k.heldPercentInsiders||0)*100}`} strokeLinecap="round"/>
                  </svg>
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>{fmtBig(k.sharesOutstanding)}</div>
                </div>
                <div style={{ fontSize: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#2962ff', marginRight: 6 }}/>Insiders: {fmtPct(k.heldPercentInsiders)}</div>
                  <div><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#26a69a', marginRight: 6 }}/>Institutions: {fmtPct(k.heldPercentInstitutions)}</div>
                </div>
              </div>
            </div>
            <div style={S.card}>
              <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 16 }}>Capital structure</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Market cap</span><span style={{ color: '#26a69a', fontWeight: 600 }}>{fmtBig(k.marketCap)}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Total debt</span><span style={{ color: '#ef5350', fontWeight: 600 }}>{fmtBig(g.totalDebt)}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Total cash</span><span style={{ color: '#2962ff', fontWeight: 600 }}>{fmtBig(g.totalCash)}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>D/E ratio</span><span style={{ color: '#fff', fontWeight: 600 }}>{g.debtToEquity?.toFixed(2) || '—'}</span></div>
              </div>
            </div>
          </div>
        </>
      )}

      {sub === 'Dividends' && (
        <div style={S.card}>
          <h3 style={{ ...S.secTitle, fontSize: 18, marginBottom: 16 }}>Dividend summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24 }}>
            <div><div style={S.kdLabel}>Dividend yield (TTM)</div><div style={S.kdValue}>{fmtPct(d.dividendYield)}</div></div>
            <div><div style={S.kdLabel}>Annual dividend rate</div><div style={S.kdValue}>{d.dividendRate ? '₹' + d.dividendRate.toFixed(2) : '—'}</div></div>
            <div><div style={S.kdLabel}>Payout ratio</div><div style={S.kdValue}>{fmtPct(d.payoutRatio)}</div></div>
          </div>
        </div>
      )}

      {sub === 'Valuation' && (
        <div style={S.card}>
          <h3 style={{ ...S.secTitle, fontSize: 18, marginBottom: 16 }}>Valuation metrics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0, borderTop: '1px solid #2a2e39' }}>
            <KDRow label="P/E (TTM)" value={k.trailingPE?.toFixed(2) || '—'} />
            <KDRow label="P/E (Forward)" value={k.forwardPE?.toFixed(2) || '—'} />
            <KDRow label="P/B" value={k.priceToBook?.toFixed(2) || '—'} />
            <KDRow label="P/S" value={k.priceToSales?.toFixed(2) || '—'} />
            <KDRow label="EPS (TTM)" value={k.trailingEps ? '₹' + k.trailingEps.toFixed(2) : '—'} />
            <KDRow label="EPS (Forward)" value={k.forwardEps ? '₹' + k.forwardEps.toFixed(2) : '—'} />
            <KDRow label="Market cap" value={fmtBig(k.marketCap)} />
            <KDRow label="Beta" value={k.beta?.toFixed(2) || '—'} />
          </div>
        </div>
      )}

      {sub === 'Growth' && (
        <>
          <div style={S.card}>
            <h3 style={{ ...S.secTitle, fontSize: 18, marginBottom: 16 }}>Growth & Profitability</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0, borderTop: '1px solid #2a2e39' }}>
              <KDRow label="Revenue growth" value={fmtPct(g.revenueGrowth)} />
              <KDRow label="Earnings growth" value={fmtPct(g.earningsGrowth)} />
              <KDRow label="Gross margins" value={fmtPct(g.grossMargins)} />
              <KDRow label="Operating margins" value={fmtPct(g.operatingMargins)} />
              <KDRow label="Profit margins" value={fmtPct(g.profitMargins)} />
              <KDRow label="Return on equity" value={fmtPct(g.returnOnEquity)} />
              <KDRow label="Return on assets" value={fmtPct(g.returnOnAssets)} />
              <KDRow label="Current ratio" value={g.currentRatio?.toFixed(2) || '—'} />
            </div>
          </div>
          <div style={S.card}>
            <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 16 }}>Cash Flow</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 0, borderTop: '1px solid #2a2e39' }}>
              <KDRow label="Total revenue" value={fmtBig(g.totalRevenue)} />
              <KDRow label="Net income" value={fmtBig(g.netIncome)} />
              <KDRow label="Free cash flow" value={fmtBig(g.freeCashflow)} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* ── FORECASTS TAB ── */
function ForecastsTab({ fin, currentPrice }) {
  if (!fin?.forecasts) return <div style={{ padding: 40, textAlign: 'center', color: '#787b86' }}>Loading forecasts…</div>;
  const f = fin.forecasts;
  const recColor = { buy: '#26a69a', strong_buy: '#26a69a', hold: '#ffc107', sell: '#ef5350', strong_sell: '#ef5350' };
  const upside = f.targetMeanPrice && currentPrice ? ((f.targetMeanPrice - currentPrice) / currentPrice * 100).toFixed(2) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={S.card}>
        <h3 style={{ ...S.secTitle, fontSize: 18, marginBottom: 20 }}>Analyst Estimates</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 32 }}>
          {/* Recommendation badge */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, minWidth: 120 }}>
            <div style={{ padding: '12px 24px', borderRadius: 12, background: (recColor[f.recommendationKey] || '#787b86') + '22', color: recColor[f.recommendationKey] || '#787b86', fontWeight: 800, fontSize: 20, textTransform: 'uppercase', border: `1px solid ${recColor[f.recommendationKey] || '#787b86'}44` }}>
              {f.recommendationKey || 'N/A'}
            </div>
            <span style={{ color: '#787b86', fontSize: 12 }}>{f.numberOfAnalystOpinions || 0} analysts</span>
          </div>
          {/* Price targets */}
          <div style={{ flex: 1 }}>
            <div style={{ position: 'relative', height: 40, background: '#131722', borderRadius: 8, overflow: 'hidden', margin: '8px 0' }}>
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px', fontSize: 11, color: '#787b86' }}>
                <span>₹{fmt(f.targetLowPrice)}</span>
                <span style={{ color: '#fff', fontWeight: 700 }}>₹{fmt(f.targetMeanPrice)}</span>
                <span>₹{fmt(f.targetHighPrice)}</span>
              </div>
              {/* Current price marker */}
              {currentPrice && f.targetLowPrice && f.targetHighPrice && (
                <div style={{ position: 'absolute', left: `${Math.min(Math.max(((currentPrice - f.targetLowPrice) / (f.targetHighPrice - f.targetLowPrice)) * 100, 5), 95)}%`, top: 0, bottom: 0, width: 2, background: '#2962ff' }}>
                  <div style={{ position: 'absolute', top: -18, left: -20, fontSize: 10, color: '#2962ff', fontWeight: 700, width: 50, textAlign: 'center' }}>CMP</div>
                </div>
              )}
              <div style={{ position: 'absolute', left: '10%', right: '10%', top: '40%', height: 4, borderRadius: 2, background: 'linear-gradient(90deg, #ef5350, #ffc107, #26a69a)' }}/>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#787b86', marginTop: 4 }}>
              <span>Low target</span>
              <span>Mean target</span>
              <span>High target</span>
            </div>
          </div>
        </div>
        {upside && (
          <div style={{ marginTop: 20, padding: '12px 16px', borderRadius: 8, background: Number(upside) > 0 ? 'rgba(38,166,154,.1)' : 'rgba(239,83,80,.1)', border: Number(upside) > 0 ? '1px solid #26a69a33' : '1px solid #ef535033', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#e1e4ea', fontSize: 14 }}>Potential upside from current price</span>
            <span style={{ color: Number(upside) > 0 ? '#26a69a' : '#ef5350', fontWeight: 700, fontSize: 18 }}>{Number(upside) > 0 ? '+' : ''}{upside}%</span>
          </div>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div style={S.card}>
          <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 12 }}>Price targets</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Target high</span><span style={{ color: '#26a69a', fontWeight: 600 }}>₹{fmt(f.targetHighPrice)}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Target mean</span><span style={{ color: '#fff', fontWeight: 600 }}>₹{fmt(f.targetMeanPrice)}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Target median</span><span style={{ color: '#fff', fontWeight: 600 }}>₹{fmt(f.targetMedianPrice)}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Target low</span><span style={{ color: '#ef5350', fontWeight: 600 }}>₹{fmt(f.targetLowPrice)}</span></div>
          </div>
        </div>
        <div style={S.card}>
          <h3 style={{ ...S.secTitle, fontSize: 16, marginBottom: 12 }}>Recommendation</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Consensus</span><span style={{ color: recColor[f.recommendationKey] || '#fff', fontWeight: 700, textTransform: 'uppercase' }}>{f.recommendationKey || '—'}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Rating (1=Buy, 5=Sell)</span><span style={{ color: '#fff', fontWeight: 600 }}>{f.recommendationMean?.toFixed(2) || '—'}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: '#787b86' }}>Number of opinions</span><span style={{ color: '#fff', fontWeight: 600 }}>{f.numberOfAnalystOpinions || '—'}</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── ERROR BOUNDARY ── */
class ErrorBoundary extends React.Component {
  constructor(p) { super(p); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(e) { return { hasError: true, error: e }; }
  render() {
    if (this.state.hasError) {
      return <div style={{ ...S.page, display: 'flex', justifyContent: 'center', alignItems: 'center' }}><div style={{ ...S.errWrap, maxWidth: 500 }}><h2 style={{ color: '#ef5350' }}>Render Error</h2><p style={{ color: '#f77c80', fontSize: 14 }}>{this.state.error?.toString()}</p><button onClick={() => window.location.reload()} style={{ ...S.btn, marginTop: 16 }}>Reload</button></div></div>;
    }
    return this.props.children;
  }
}

/* ── HELPERS ── */
const CRYPTO_SYMBOLS = new Set(['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX']);
const isCrypto = (sym) => {
  if (!sym) return false;
  const base = sym.split('-')[0].toUpperCase();
  return CRYPTO_SYMBOLS.has(base) || sym.includes('-USD') || sym.includes('-INR') || sym.includes('-USDT');
};

/* ── MAIN COMPONENT ── */
const TABS = ['Overview', 'Financials', 'Technicals', 'Forecasts'];
const CRYPTO_QUICK = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD'];
const EQUITY_QUICK = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'AAPL'];

function AnalysisViewImpl({ initialSymbol, initialMarket }) {
  const [symbol, setSymbol] = useState(initialSymbol || 'RELIANCE');
  const [engine, setEngine] = useState('astra');
  const [cryptoMarket, setCryptoMarket] = useState(initialMarket || 'international');
  const [activeTab, setActiveTab] = useState('Overview');
  const [data, setData] = useState(null);
  const [fin, setFin] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showFullChart, setShowFullChart] = useState(false);
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);

  // Update symbol/market if parent navigation changes
  useEffect(() => {
    if (initialSymbol) {
      setSymbol(initialSymbol);
      if (initialMarket) setCryptoMarket(initialMarket);
    }
  }, [initialSymbol, initialMarket]);

  const isCryptoMode = isCrypto(symbol);

  const analyze = useCallback(async (sym, eng, mkt) => {
    sym = sym || symbol; eng = eng || engine; mkt = mkt || cryptoMarket;
    if (!sym) return;
    setLoading(true); setError(''); setData(null); setFin(null);
    try {
      const token = localStorage.getItem('astra_token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      const isC = isCrypto(sym);
      let payload;

      if (isC) {
        // Route to crypto endpoint
        const cryptoEngine = eng.startsWith('astra_crypto') ? eng : 'astra_crypto';
        const res = await fetch(
          `http://localhost:8000/api/crypto/analyze/${sym}?market=${mkt}&engine=${cryptoEngine}`,
          { headers }
        );
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        payload = json;
      } else {
        // Route to equity endpoint
        const [resA, resF] = await Promise.all([
          fetch(`http://localhost:8000/api/analyze/${sym}?engine=${eng}`, { headers }),
          fetch(`http://localhost:8000/api/financials/${sym}`, { headers }),
        ]);
        if (!resA.ok) throw new Error(`Server returned ${resA.status}`);
        const json = await resA.json();
        payload = json.result || json;
        if (payload.error) throw new Error(payload.error);
        try { const fj = await resF.json(); setFin(fj); } catch {}
      }

      setData(payload);
    } catch (e) { setError(e.message || 'Connection failed'); }
    finally { setLoading(false); }
  }, [symbol, engine, cryptoMarket]);

  const loadInterval = useCallback(async (interval, period) => {
    try {
      const token = localStorage.getItem('astra_token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const isC = isCrypto(symbol);
      let res;
      if (isC) {
        res = await fetch(`http://localhost:8000/api/crypto/analyze/${symbol}?market=${cryptoMarket}&timeframe=${interval}`, { headers });
      } else {
        res = await fetch(`http://localhost:8000/api/analyze/${symbol}?engine=${engine}&interval=${interval}&period=${period}`, { headers });
      }
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const json = await res.json();
      const payload = json.result || json;
      setData(payload);
    } catch (e) { console.error('Interval load failed:', e); }
  }, [symbol, engine, cryptoMarket]);

  useEffect(() => { analyze(); }, [initialSymbol]);
  const handleSubmit = (e) => { e.preventDefault(); analyze(symbol, engine, cryptoMarket); };
  const switchEngine = (eng) => { setEngine(eng); analyze(symbol, eng, cryptoMarket); };

  const kd = useMemo(() => {
    if (!data?.chartData?.length) return null;
    const c = data.chartData, last = c[c.length - 1], prev = c.length > 1 ? c[c.length - 2] : last;
    const diff = last.close - prev.close, pct = prev.close ? (diff / prev.close * 100) : 0;
    const rsi = last.rsi != null ? last.rsi : 50;
    const smaG = last.sma200 && last.close ? (last.close / last.sma200) * 50 : 50;
    return { volume: fmtVol(last.volume), prevClose: fmt(prev.close), open: fmt(last.open), dayRange: `${fmt(last.low)} — ${fmt(last.high)}`, diff: diff.toFixed(2), pct: pct.toFixed(2), positive: diff >= 0, rsi, smaGauge: Math.min(Math.max(smaG, 0), 100), summary: Math.min(Math.max((rsi + smaG) / 2, 0), 100) };
  }, [data]);

  // Mini chart
  useEffect(() => {
    if (!data?.chartData?.length || !chartContainerRef.current) return;
    if (chartRef.current) { try { chartRef.current.remove(); } catch {} chartRef.current = null; }
    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#131722' }, textColor: 'rgba(255,255,255,.55)' },
      grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#2a2e39' },
      timeScale: { borderColor: '#2a2e39', timeVisible: false },
      autoSize: true,
    });
    chartRef.current = chart;
    const cs = chart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' });
    const vs = chart.addHistogramSeries({ color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: '', scaleMargins: { top: 0.85, bottom: 0 } });
    const sm = chart.addLineSeries({ color: '#ffc107', lineWidth: 1.5, crosshairMarkerVisible: false });
    const candles = [], volumes = [], smas = [], used = new Set();
    const sorted = [...data.chartData].sort((a, b) => (a.time > b.time ? 1 : -1));
    for (const d of sorted) { if (used.has(d.time)) continue; used.add(d.time); candles.push({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }); volumes.push({ time: d.time, value: d.volume || 0, color: d.close >= d.open ? 'rgba(38,166,154,.35)' : 'rgba(239,83,80,.35)' }); if (d.sma200 != null) smas.push({ time: d.time, value: d.sma200 }); }
    cs.setData(candles); vs.setData(volumes); if (smas.length) sm.setData(smas);

    // ── Signal price lines (entry / target / stop-loss) ──────────────────────
    const entryP = data.entry_price || data.entryPrice;
    const targetP = data.target;
    const slP = data.stop_loss || data.stopLoss;
    if (entryP && entryP > 0) {
      cs.createPriceLine({
        price: entryP,
        color: '#2962ff',
        lineWidth: 1.5,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Entry ₹${Number(entryP).toFixed(2)}`,
      });
    }
    if (targetP && targetP > 0) {
      cs.createPriceLine({
        price: targetP,
        color: '#26a69a',
        lineWidth: 1.5,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Target ₹${Number(targetP).toFixed(2)}`,
      });
    }
    if (slP && slP > 0) {
      cs.createPriceLine({
        price: slP,
        color: '#ef5350',
        lineWidth: 1.5,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `SL ₹${Number(slP).toFixed(2)}`,
      });
    }

    chart.timeScale().fitContent();
    return () => { try { chart.remove(); } catch {} chartRef.current = null; };
  }, [data]);

  return (
    <div style={S.page}>
      <div style={S.inner}>
        {/* SEARCH */}
        <form onSubmit={handleSubmit} style={S.searchRow}>
          <div style={{ position: 'relative', flex: 1, maxWidth: 420 }}>
            <Search size={16} style={S.searchIcon}/>
            <input
              style={S.searchInput}
              placeholder={isCryptoMode ? 'Crypto: BTC-USD, ETH-INR, SOL-USD…' : 'Equity: RELIANCE, TCS, AAPL…'}
              value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
            />
          </div>
          <button type="submit" style={S.btn}>Analyze</button>
        </form>

        {/* Quick presets */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: '#787b86', fontWeight: 600, textTransform: 'uppercase' }}>Quick:</span>
          {EQUITY_QUICK.map(s => (
            <button key={s} onClick={() => { setSymbol(s); analyze(s, engine, cryptoMarket); }}
              style={{ padding: '4px 12px', borderRadius: 6, border: '1px solid #2a2e39', background: symbol === s ? '#2962ff22' : '#1e222d', color: symbol === s ? '#2962ff' : '#787b86', cursor: 'pointer', fontSize: 12, fontFamily: "'Inter', sans-serif" }}>
              {s}
            </button>
          ))}
          <span style={{ color: '#2a2e39', fontSize: 14 }}>|</span>
          <Bitcoin size={13} color="#f7931a"/>
          {CRYPTO_QUICK.map(s => (
            <button key={s} onClick={() => { setSymbol(s); analyze(s, engine, cryptoMarket); }}
              style={{ padding: '4px 12px', borderRadius: 6, border: '1px solid #2a2e39', background: symbol === s ? '#f7931a22' : '#1e222d', color: symbol === s ? '#f7931a' : '#787b86', cursor: 'pointer', fontSize: 12, fontFamily: "'Inter', sans-serif" }}>
              {s}
            </button>
          ))}
        </div>

        {/* Engine bar — show crypto engines for crypto symbols */}
        {isCryptoMode ? (
          <div style={S.engineBar}>
            {[
              ['astra_crypto', 'ASTRA.CRYPTO'],
              ['astra_crypto_ml', 'ASTRA.CRYPTO.ML'],
            ].map(([e, label]) => (
              <button key={e} style={S.engineTab(engine === e)} onClick={() => switchEngine(e)}>{label}</button>
            ))}
            <div style={{ width: 1, background: '#2a2e39', margin: '4px 4px' }}/>
            {[['international', '🌐 Intl (USD)'], ['indian', '🇮🇳 India (INR)']].map(([m, label]) => (
              <button key={m} style={S.engineTab(cryptoMarket === m)}
                onClick={() => { setCryptoMarket(m); analyze(symbol, engine, m); }}>{label}</button>
            ))}
          </div>
        ) : (
          <div style={S.engineBar}>
            {['astra', 'astra_ai', 'astra_ml'].map(e => (
              <button key={e} style={S.engineTab(engine === e)} onClick={() => switchEngine(e)}>
                {e === 'astra' ? 'ASTRA 1.0' : e === 'astra_ai' ? 'ASTRA.AI' : 'ASTRA.ML'}
              </button>
            ))}
          </div>
        )}

        {loading && <div style={S.loadWrap}><RefreshCw size={28} color="#787b86" style={{ animation: 'spin 1.5s linear infinite' }}/><p style={{ color: '#787b86', marginTop: 16, fontSize: 14 }}>Analyzing {symbol}…</p><style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style></div>}
        {error && !loading && <div style={S.errWrap}><p style={{ color: '#ef5350', fontSize: 18, fontWeight: 700 }}>Analysis Failed</p><p style={{ color: '#f77c80', fontSize: 13 }}>{error}</p><button onClick={() => analyze()} style={{ ...S.btn, marginTop: 12 }}>Retry</button></div>}

        {data && kd && !loading && !error && (
          <>
            {/* ASSET HEADER */}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, borderBottom: '1px solid #2a2e39', paddingBottom: 20 }}>
              <div style={S.avatar}>{(data.asset || symbol).charAt(0)}</div>
              <div style={{ flex: 1 }}>
                <h1 style={{ fontSize: 28, fontWeight: 700, lineHeight: 1.1, color: '#fff', margin: 0 }}>{fin?.profile?.name || data.asset || symbol}</h1>
                <div style={{ fontSize: 13, color: '#787b86', marginTop: 2, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ background: '#2962ff22', color: '#2962ff', padding: '2px 8px', borderRadius: 4, fontWeight: 600, fontSize: 11 }}>NSE</span>
                  <span>•</span><span>{fin?.profile?.sector || (data.engine?.toUpperCase() || 'ASTRA') + ' Engine'}</span>
                  {fin?.profile?.industry && <><span>•</span><span>{fin.profile.industry}</span></>}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: '#fff', letterSpacing: -1 }}>₹{data.current_price?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</div>
                <div style={{ fontSize: 14, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 4, color: kd.positive ? '#26a69a' : '#ef5350' }}>
                  {kd.positive ? <TrendingUp size={14}/> : <TrendingDown size={14}/>}
                  {kd.positive ? '+' : ''}{kd.diff} ({kd.pct}%)
                </div>
              </div>
            </div>

            {/* TABS */}
            <div style={S.tabBar}>
              {TABS.map(t => <button key={t} style={S.tab(activeTab === t)} onClick={() => setActiveTab(t)}>{t}</button>)}
            </div>

            {/* OVERVIEW */}
            {activeTab === 'Overview' && (
              <>
                <div style={S.chartWrap}>
                  <div style={S.chartHeader}>
                    <span style={{ color: '#787b86', fontSize: 13, fontWeight: 600 }}>Chart</span>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button style={S.chartBtn} onClick={() => chartRef.current?.timeScale().scrollToPosition?.(-10, false)}><ZoomOut size={14}/></button>
                      <button style={S.chartBtn} onClick={() => chartRef.current?.timeScale().scrollToPosition?.(10, false)}><ZoomIn size={14}/></button>
                      <button style={S.chartBtn} onClick={() => setShowFullChart(true)}><Maximize2 size={14}/> Full chart</button>
                    </div>
                  </div>
                  <div ref={chartContainerRef} style={{ width: '100%', height: 420 }}/>
                </div>
                <div><h2 style={S.secTitle}>Key data points</h2><div style={{ ...S.kdGrid, marginTop: 16, borderTop: '1px solid #2a2e39' }}><KDRow label="Volume" value={kd.volume}/><KDRow label="Previous close" value={'₹'+kd.prevClose}/><KDRow label="Open" value={'₹'+kd.open}/><KDRow label="Day's range" value={kd.dayRange}/></div></div>
                <div style={S.card}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ ...S.secTitle, fontSize: 18 }}>ASTRA Signal</h2>
                    <span style={S.signalBadge(data.signal)}>{data.signal} — {data.confidence?.toFixed(1)}%</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginTop: 16 }}>
                    <div><span style={S.kdLabel}>Entry Price</span><div style={{ ...S.kdValue, color: '#2962ff' }}>₹{fmt(data.entry_price)}</div></div>
                    <div><span style={S.kdLabel}>Target</span><div style={{ ...S.kdValue, color: '#26a69a' }}>₹{fmt(data.target)}</div></div>
                    <div><span style={S.kdLabel}>Stop Loss</span><div style={{ ...S.kdValue, color: '#ef5350' }}>₹{fmt(data.stop_loss)}</div></div>
                  </div>
                </div>
                <div>
                  <h2 style={S.secTitle}>Technicals <ChevronRight size={18} color="#787b86"/></h2>
                  <p style={S.secSub}>Summarizing what the indicators are suggesting.</p>
                  <div style={S.gaugeGrid}><Gauge value={100 - kd.rsi} label="Oscillators"/><Gauge value={kd.summary} label="Summary"/><Gauge value={kd.smaGauge} label="Moving Averages"/></div>
                </div>
              </>
            )}

            {/* FINANCIALS */}
            {activeTab === 'Financials' && <FinancialsTab fin={fin}/>}

            {/* TECHNICALS */}
            {activeTab === 'Technicals' && (
              <div>
                <h2 style={S.secTitle}>Technicals <ChevronRight size={18} color="#787b86"/></h2>
                <p style={S.secSub}>Summarizing what the indicators are suggesting.</p>
                <div style={{ display: 'flex', justifyContent: 'center', padding: '32px 0' }}><Gauge value={kd.summary} label="Summary"/></div>
                <div style={S.gaugeGrid}><Gauge value={100 - kd.rsi} label="Oscillators"/><Gauge value={kd.summary} label="Summary"/><Gauge value={kd.smaGauge} label="Moving Averages"/></div>
              </div>
            )}

            {/* FORECASTS */}
            {activeTab === 'Forecasts' && <ForecastsTab fin={fin} currentPrice={data.current_price}/>}
          </>
        )}
      </div>
      {showFullChart && data?.chartData && <FullChartModal data={data} symbol={symbol} onClose={() => setShowFullChart(false)} onLoadInterval={loadInterval}/>}
    </div>
  );
}

export default function AnalysisView({ initialSymbol, initialMarket }) {
  return <ErrorBoundary><AnalysisViewImpl initialSymbol={initialSymbol} initialMarket={initialMarket}/></ErrorBoundary>;
}
