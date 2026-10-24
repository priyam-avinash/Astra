import React, { useCallback, useEffect, useState } from 'react';
import { Landmark, Search, RefreshCw, TrendingUp, Repeat, X } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { API_URL } from './config';

// Mutual funds (v1.14) — NAVs from AMFI via mfapi.in. Paper investing only.
const inr = (n, d = 2) => (n == null || Number.isNaN(Number(n)) ? '—'
  : '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (n) => (n == null ? '—' : `${n > 0 ? '+' : ''}${Number(n).toFixed(2)}%`);
const tone = (n) => (n == null ? 'var(--text-secondary)' : n >= 0 ? '#22c55e' : '#ef4444');
const QUICK = ['Nifty 50 Index', 'Flexi Cap', 'Parag Parikh', 'Mid Cap', 'Small Cap', 'ELSS', 'Liquid'];

async function api(path, opts = {}) {
  const r = await fetch(`${API_URL}${path}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    signal: AbortSignal.timeout(30000),
  });
  let body = null;
  try { body = await r.json(); } catch (_) { /* empty */ }
  if (!r.ok) throw new Error(body?.detail || `Request failed (${r.status})`);
  return body;
}

function Stat({ label, value, color }) {
  return (
    <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,.03)', borderRadius: 10 }}>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, marginTop: 4, color: color || 'inherit' }}>{value}</div>
    </div>
  );
}

export default function MutualFundsView() {
  const [q, setQ] = useState('');
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [fund, setFund] = useState(null);
  const [chart, setChart] = useState([]);
  const [period, setPeriod] = useState('5y');
  const [loadingFund, setLoadingFund] = useState(false);
  const [sipCalc, setSipCalc] = useState({ amount: 10000, years: 5, result: null });
  const [lump, setLump] = useState(25000);
  const [sip, setSip] = useState({ amount: 5000, day: 5, start: '' });
  const [pf, setPf] = useState(null);
  const [msg, setMsg] = useState(null);       // {type:'ok'|'err', text}
  const [busy, setBusy] = useState(false);

  const loadPortfolio = useCallback(async () => {
    try { setPf(await api('/api/mf/portfolio')); } catch (e) { setMsg({ type: 'err', text: e.message }); }
  }, []);
  useEffect(() => { loadPortfolio(); }, [loadPortfolio]);

  const runSearch = async (text) => {
    const term = (text ?? q).trim();
    if (term.length < 2) return;
    setQ(term); setSearching(true); setMsg(null);
    try { setResults((await api(`/api/mf/search?q=${encodeURIComponent(term)}`)).results || []); }
    catch (e) { setMsg({ type: 'err', text: e.message }); setResults([]); }
    finally { setSearching(false); }
  };

  const openFund = async (code) => {
    setLoadingFund(true); setMsg(null); setSipCalc(s => ({ ...s, result: null }));
    try {
      const [info, ch] = await Promise.all([api(`/api/mf/scheme/${code}`), api(`/api/mf/scheme/${code}/chart?period=${period}`)]);
      setFund(info); setChart(ch.points || []);
    } catch (e) { setMsg({ type: 'err', text: e.message }); }
    finally { setLoadingFund(false); }
  };

  const changePeriod = async (p) => {
    setPeriod(p);
    if (!fund) return;
    try { setChart((await api(`/api/mf/scheme/${fund.scheme_code}/chart?period=${p}`)).points || []); } catch (_) { /* keep old */ }
  };

  const calcSip = async () => {
    try {
      const r = await api(`/api/mf/scheme/${fund.scheme_code}/sip-calc?amount=${sipCalc.amount}&years=${sipCalc.years}`);
      setSipCalc(s => ({ ...s, result: r }));
    } catch (e) { setSipCalc(s => ({ ...s, result: { error: e.message } })); }
  };

  const act = async (fn, okText) => {
    setBusy(true); setMsg(null);
    try { await fn(); setMsg({ type: 'ok', text: okText }); await loadPortfolio(); }
    catch (e) { setMsg({ type: 'err', text: e.message }); }
    finally { setBusy(false); }
  };

  const invest = () => act(() => api('/api/mf/invest', { method: 'POST', body: JSON.stringify({ scheme_code: fund.scheme_code, amount: Number(lump) }) }),
    `Paper investment of ${inr(lump, 0)} booked at NAV ${fund.nav}`);
  const startSip = () => act(() => api('/api/mf/sip', { method: 'POST', body: JSON.stringify({
    scheme_code: fund.scheme_code, amount: Number(sip.amount), day_of_month: Number(sip.day), start_date: sip.start || null }) }),
    `Paper SIP of ${inr(sip.amount, 0)}/month started`);
  const redeem = (h) => act(() => api('/api/mf/redeem', { method: 'POST', body: JSON.stringify({ scheme_code: h.scheme_code, units: h.units }) }),
    `Redeemed all units of ${h.scheme_name} (paper)`);
  const stopSip = (id) => act(() => api(`/api/mf/sip/${id}`, { method: 'DELETE' }), 'SIP stopped');

  const r = fund?.returns || {};
  const risk = fund?.risk || {};

  return (
    <div className="animate-fade-in">
      <div className="topbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: 10 }}><Landmark size={20} /> Mutual Funds</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 4 }}>
            Every AMFI scheme with daily NAVs, returns and risk. Lumpsum and SIP investing here is paper only — nothing is sent to an AMC.
          </p>
        </div>
        <button className="btn-ghost" onClick={loadPortfolio}><RefreshCw size={14} /> Refresh</button>
      </div>

      {msg && <div role="status" style={{ margin: '12px 0', fontSize: 13, color: msg.type === 'ok' ? '#22c55e' : '#ef4444' }}>{msg.text}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 340px) 1fr', gap: 16, marginTop: 16 }} className="mf-grid">
        {/* Search */}
        <div className="glass-panel" style={{ padding: 16 }}>
          <form onSubmit={(e) => { e.preventDefault(); runSearch(); }} style={{ display: 'flex', gap: 8 }}>
            <input className="search-input" aria-label="Search mutual funds" placeholder="Search funds, e.g. HDFC Flexi Cap"
                   value={q} onChange={e => setQ(e.target.value)} style={{ flex: 1 }} />
            <button className="btn-primary" type="submit" disabled={searching} aria-label="Search"><Search size={14} /></button>
          </form>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
            {QUICK.map(t => <button key={t} className="btn-ghost" style={{ fontSize: 11, padding: '4px 8px' }} onClick={() => runSearch(t)}>{t}</button>)}
          </div>
          <div style={{ marginTop: 12, maxHeight: 420, overflowY: 'auto' }}>
            {searching && <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Searching…</div>}
            {!searching && results.length === 0 && <div style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>Search by fund or AMC name. Direct-plan growth options are listed first.</div>}
            {results.map(f => (
              <button key={f.scheme_code} onClick={() => openFund(f.scheme_code)}
                      style={{ display: 'block', width: '100%', textAlign: 'left', padding: '8px 10px', marginBottom: 4, borderRadius: 8,
                               background: fund?.scheme_code === f.scheme_code ? 'rgba(99,102,241,.15)' : 'transparent',
                               border: '1px solid var(--panel-border)', color: 'inherit', cursor: 'pointer', fontSize: 12.5 }}>
                {f.scheme_name}<span style={{ color: 'var(--text-tertiary)', marginLeft: 6 }}>#{f.scheme_code}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Fund detail */}
        <div className="glass-panel" style={{ padding: 16, minHeight: 300 }}>
          {loadingFund && <div style={{ color: 'var(--text-secondary)' }}>Loading fund…</div>}
          {!loadingFund && !fund && <div style={{ color: 'var(--text-tertiary)', padding: 40, textAlign: 'center' }}>Pick a fund to see NAV history, returns, risk and a SIP calculator.</div>}
          {!loadingFund && fund && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: 18 }}>{fund.scheme_name}</h3>
                  <div style={{ color: 'var(--text-secondary)', fontSize: 12.5, marginTop: 4 }}>{fund.fund_house} · {fund.category}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>{inr(fund.nav, 4)}</div>
                  <div style={{ fontSize: 12, color: fund.nav_stale_days > 5 ? '#f59e0b' : 'var(--text-tertiary)' }}>NAV as of {fund.nav_date}</div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 6, margin: '14px 0 6px' }}>
                {['1y', '3y', '5y', '10y', 'max'].map(p => (
                  <button key={p} className="btn-ghost" onClick={() => changePeriod(p)}
                          style={{ fontSize: 11, padding: '3px 8px', background: period === p ? 'rgba(255,255,255,.08)' : undefined }}>{p.toUpperCase()}</button>
                ))}
              </div>
              <div style={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chart} margin={{ top: 5, right: 8, left: 0, bottom: 0 }}>
                    <defs><linearGradient id="mfg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6366f1" stopOpacity={0.35} /><stop offset="100%" stopColor="#6366f1" stopOpacity={0} /></linearGradient></defs>
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#787b86' }} minTickGap={40} />
                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: '#787b86' }} width={48} />
                    <Tooltip formatter={(v) => inr(v, 4)} contentStyle={{ background: '#111827', border: '1px solid #2a2e39', fontSize: 12 }} />
                    <Area type="monotone" dataKey="value" stroke="#6366f1" fill="url(#mfg)" strokeWidth={1.6} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: 8, marginTop: 12 }}>
                {[['1M', r['1m']], ['6M', r['6m']], ['1Y', r['1y']], ['3Y CAGR', r['3y']], ['5Y CAGR', r['5y']], ['10Y CAGR', r['10y']], ['Since launch', r.since_inception]]
                  .map(([l, v]) => <Stat key={l} label={l} value={pct(v)} color={tone(v)} />)}
                <Stat label="Volatility (3Y)" value={risk.volatility_3y_pct == null ? '—' : `${risk.volatility_3y_pct}%`} />
                <Stat label="Max drawdown" value={pct(risk.max_drawdown_pct)} color="#ef4444" />
                {risk.rolling_3y_cagr && <Stat label="Rolling 3Y (median)" value={pct(risk.rolling_3y_cagr.median)} />}
                {risk.rolling_3y_cagr && <Stat label="3Y windows positive" value={`${risk.rolling_3y_cagr.pct_positive}%`} />}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, marginTop: 16 }}>
                <div style={{ border: '1px solid var(--panel-border)', borderRadius: 12, padding: 12 }}>
                  <div style={{ fontWeight: 600, display: 'flex', gap: 6, alignItems: 'center' }}><TrendingUp size={14} /> SIP calculator (actual NAVs)</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <label style={{ fontSize: 12, flex: 1 }}>Monthly ₹<input className="search-input" type="number" min="100" value={sipCalc.amount} onChange={e => setSipCalc(s => ({ ...s, amount: e.target.value }))} style={{ width: '100%' }} /></label>
                    <label style={{ fontSize: 12, width: 80 }}>Years<input className="search-input" type="number" min="1" max="30" value={sipCalc.years} onChange={e => setSipCalc(s => ({ ...s, years: e.target.value }))} style={{ width: '100%' }} /></label>
                  </div>
                  <button className="btn-ghost" style={{ marginTop: 8 }} onClick={calcSip}>Calculate</button>
                  {sipCalc.result && (sipCalc.result.error
                    ? <div style={{ color: '#ef4444', fontSize: 12, marginTop: 6 }}>{sipCalc.result.error}</div>
                    : <div style={{ fontSize: 13, marginTop: 8, lineHeight: 1.6 }}>
                        Invested {inr(sipCalc.result.invested, 0)} over {sipCalc.result.instalments} months → worth <b>{inr(sipCalc.result.value, 0)}</b> today
                        <span style={{ color: tone(sipCalc.result.xirr_pct) }}> (XIRR {pct(sipCalc.result.xirr_pct)})</span>
                      </div>)}
                </div>
                <div style={{ border: '1px solid var(--panel-border)', borderRadius: 12, padding: 12 }}>
                  <div style={{ fontWeight: 600 }}>Paper lumpsum</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <input className="search-input" type="number" min="100" aria-label="Lumpsum amount" value={lump} onChange={e => setLump(e.target.value)} style={{ flex: 1 }} />
                    <button className="btn-primary" disabled={busy} onClick={invest}>Invest</button>
                  </div>
                  <div style={{ fontWeight: 600, marginTop: 12, display: 'flex', gap: 6, alignItems: 'center' }}><Repeat size={14} /> Paper SIP</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
                    <input className="search-input" type="number" min="100" aria-label="SIP amount" value={sip.amount} onChange={e => setSip(s => ({ ...s, amount: e.target.value }))} style={{ width: 100 }} />
                    <input className="search-input" type="number" min="1" max="28" aria-label="SIP day of month" value={sip.day} onChange={e => setSip(s => ({ ...s, day: e.target.value }))} style={{ width: 60 }} />
                    <input className="search-input" type="date" aria-label="SIP start date (optional, can be in the past)" value={sip.start} onChange={e => setSip(s => ({ ...s, start: e.target.value }))} style={{ width: 150 }} />
                    <button className="btn-primary" disabled={busy} onClick={startSip}>Start SIP</button>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 6 }}>Amount · day of month · optional past start date (past instalments are booked at their real NAVs).</div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Portfolio */}
      <div className="glass-panel" style={{ padding: 16, marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <h3 style={{ margin: 0, fontSize: 16 }}>My mutual funds <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>(paper)</span></h3>
          {pf && <div style={{ display: 'flex', gap: 18, fontSize: 13 }}>
            <span>Invested <b>{inr(pf.total_invested, 0)}</b></span>
            <span>Value <b>{inr(pf.total_value, 0)}</b></span>
            <span style={{ color: tone(pf.total_gain) }}>{inr(pf.total_gain, 0)} ({pct(pf.total_gain_pct)})</span>
          </div>}
        </div>
        {!pf && <div style={{ color: 'var(--text-secondary)', marginTop: 10 }}>Loading…</div>}
        {pf && pf.holdings.length === 0 && <div style={{ color: 'var(--text-tertiary)', marginTop: 10, fontSize: 13 }}>No holdings yet — pick a fund above to invest.</div>}
        {pf && pf.holdings.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginTop: 10 }}>
              <thead><tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: 11, textTransform: 'uppercase' }}>
                {['Fund', 'Units', 'Invested', 'Value', 'Gain', 'XIRR', ''].map(h => <th key={h} style={{ padding: '8px 10px' }}>{h}</th>)}
              </tr></thead>
              <tbody>{pf.holdings.map(h => (
                <tr key={h.scheme_code} style={{ borderTop: '1px solid var(--panel-border)' }}>
                  <td style={{ padding: '8px 10px' }}><button onClick={() => openFund(h.scheme_code)} style={{ background: 'none', border: 0, color: 'inherit', cursor: 'pointer', textAlign: 'left', padding: 0 }}>{h.scheme_name}</button></td>
                  <td style={{ padding: '8px 10px' }}>{h.units.toFixed(3)}</td>
                  <td style={{ padding: '8px 10px' }}>{inr(h.invested, 0)}</td>
                  <td style={{ padding: '8px 10px' }}>{inr(h.value, 0)}</td>
                  <td style={{ padding: '8px 10px', color: tone(h.gain) }}>{inr(h.gain, 0)} ({pct(h.gain_pct)})</td>
                  <td style={{ padding: '8px 10px', color: tone(h.xirr_pct) }}>{pct(h.xirr_pct)}</td>
                  <td style={{ padding: '8px 10px' }}><button className="btn-ghost" disabled={busy} style={{ fontSize: 11 }} onClick={() => redeem(h)}>Redeem all</button></td>
                </tr>))}
              </tbody>
            </table>
          </div>
        )}
        {pf && pf.sips.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Active SIPs</div>
            {pf.sips.map(s => (
              <div key={s.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', fontSize: 13, borderTop: '1px solid var(--panel-border)' }}>
                <span>{s.scheme_name} · {inr(s.amount, 0)} on day {s.day_of_month} · last booked {s.last_run_date || '—'}</span>
                <button className="btn-ghost" aria-label="Stop SIP" style={{ fontSize: 11 }} disabled={busy} onClick={() => stopSip(s.id)}><X size={12} /> Stop</button>
              </div>))}
          </div>
        )}
        {pf && pf.transactions.length > 0 && (
          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer', fontSize: 13, color: 'var(--text-secondary)' }}>Recent transactions</summary>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 8 }}><tbody>
              {pf.transactions.map((t, i) => (
                <tr key={i} style={{ borderTop: '1px solid var(--panel-border)' }}>
                  <td style={{ padding: '6px 8px' }}>{t.date}</td><td style={{ padding: '6px 8px' }}>{t.kind}</td>
                  <td style={{ padding: '6px 8px' }}>{t.scheme_name}</td><td style={{ padding: '6px 8px' }}>{inr(t.amount)}</td>
                  <td style={{ padding: '6px 8px' }}>{t.units.toFixed(3)} @ {t.nav.toFixed(4)}</td>
                </tr>))}
            </tbody></table>
          </details>
        )}
      </div>
      <style>{`@media (max-width: 820px) { .mf-grid { grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
