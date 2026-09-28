import React, { useCallback, useEffect, useState } from 'react';
import { Package, RefreshCw, Send } from 'lucide-react';
import { API_URL } from './config';

// Live commodities view (v1.13) — backed by /api/commodities/scan (Yahoo futures data)
const authHeaders = () => {
  const t = localStorage.getItem('astra_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};
const sigColor = (s) => (s === 'BUY' ? '#22c55e' : s === 'SELL' ? '#ef4444' : '#f59e0b');
const fmt = (n, d = 2) => (n == null || Number.isNaN(Number(n)) ? '—'
  : Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));

export default function CommoditiesView({ setQueue }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [asOf, setAsOf] = useState('');
  const [pushed, setPushed] = useState({});

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const r = await fetch(`${API_URL}/api/commodities/scan`, { headers: authHeaders(), signal: AbortSignal.timeout(90000) });
      if (!r.ok) throw new Error(`Server returned ${r.status}`);
      const d = await r.json();
      setRows(d.signals || []);
      setAsOf(new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }));
    } catch (e) {
      setError(e.name === 'TimeoutError' ? 'Commodity scan timed out — is the backend running?' : e.message);
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const push = async (c) => {
    try {
      const r = await fetch(`${API_URL}/api/signals`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ asset: c.symbol, type: 'Commodity', signal: c.signal, confidence: c.confidence,
                               entry_price: c.entry_price, target_price: c.target, stop_loss: c.stop_loss,
                               engine: 'commodities' }),
      });
      if (!r.ok) throw new Error();
      const d = await r.json();
      setPushed(p => ({ ...p, [c.symbol]: true }));
      setQueue?.(q => [{ id: d.id, asset: c.symbol, type: 'Commodity', signal: c.signal, entryPrice: c.entry_price,
                        targetPrice: c.target, stopLoss: c.stop_loss, confidence: c.confidence, age: 'Just now' }, ...q]);
    } catch { setError(`Could not queue ${c.symbol}`); }
  };

  return (
    <div className="animate-fade-in">
      <div className="topbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: 10 }}><Package size={20} /> Commodities</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 4 }}>
            Gold · Silver · Crude · Natural Gas · Copper: technicals plus macro (VIX, dollar) and seasonality. Paper trading only.
          </p>
        </div>
        <button className="btn-ghost" onClick={load} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> {loading ? 'Scanning…' : `Refresh${asOf ? ` · ${asOf}` : ''}`}
        </button>
      </div>

      {error && <div style={{ color: '#ef4444', margin: '12px 0', fontSize: 13 }}>{error}</div>}

      <div className="glass-panel" style={{ padding: 0, overflow: 'hidden', marginTop: 16 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: '0.72rem', textTransform: 'uppercase' }}>
              {['Commodity', 'Signal', 'Confidence', 'Price', 'Stop loss', 'Target', 'RSI', 'ADX', 'Season', ''].map(h =>
                <th key={h} style={{ padding: '12px 16px' }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {!rows.length && (
              <tr><td colSpan={10} style={{ padding: 40, textAlign: 'center', color: 'var(--text-secondary)' }}>
                {loading ? 'Fetching live futures data…' : 'No commodity data.'}
              </td></tr>
            )}
            {rows.map(c => (
              <tr key={c.symbol} style={{ borderTop: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '12px 16px', fontWeight: 600 }}>{c.name} <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>{c.symbol}</span></td>
                <td style={{ padding: '12px 16px', color: sigColor(c.signal), fontWeight: 700 }}>{c.signal}</td>
                <td style={{ padding: '12px 16px' }}>{fmt(c.confidence, 0)}%</td>
                <td style={{ padding: '12px 16px' }}>${fmt(c.entry_price)}</td>
                <td style={{ padding: '12px 16px', color: '#ef4444' }}>${fmt(c.stop_loss)}</td>
                <td style={{ padding: '12px 16px', color: '#22c55e' }}>${fmt(c.target)}</td>
                <td style={{ padding: '12px 16px' }}>{fmt(c.rsi, 1)}</td>
                <td style={{ padding: '12px 16px' }}>{fmt(c.adx, 1)}</td>
                <td style={{ padding: '12px 16px', color: sigColor(c.seasonal_bias === 'BULL' ? 'BUY' : c.seasonal_bias === 'BEAR' ? 'SELL' : '') }}>{c.seasonal_bias || '—'}</td>
                <td style={{ padding: '12px 16px' }}>
                  {c.signal !== 'HOLD' && (
                    <button className="btn-ghost" disabled={pushed[c.symbol]} onClick={() => push(c)} style={{ fontSize: 11 }}>
                      <Send size={11} /> {pushed[c.symbol] ? 'Queued' : 'Queue'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
