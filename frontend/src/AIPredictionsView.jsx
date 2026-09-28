import React, { useState, useEffect } from 'react';
import { Zap, RefreshCw } from 'lucide-react';

import { API_URL } from './config';

/**
 * AI Predictions View
 * 
 * This view shows AI-generated signals that have NOT yet been pushed to the Auto Mode queue.
 * When the user clicks "Push to Queue", the signal is:
 * 1. Sent to the backend via POST /api/signals
 * 2. Added to the shared `queue` state so Auto Mode immediately sees it
 * 3. Removed from this local predictions list
 */

// v1.13: predictions come from the live scanners (no more mock seeds)
const authHeaders = () => {
  const t = localStorage.getItem('astra_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};

const money = (pred, v) => {
  const n = Number(v);
  if (v == null || Number.isNaN(n)) return '—';
  const inr = pred.type !== 'Commodity' && !/-USD$|=F$/.test(pred.asset || '');
  return (inr ? '₹' : '$') + n.toLocaleString(inr ? 'en-IN' : 'en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const signalColor = (d) => d === 'BUY' ? '#22c55e' : d === 'SELL' ? '#ef4444' : '#f59e0b';

export default function AIPredictionsView({ queue, setQueue }) {
  const [predictions, setPredictions] = useState([]);
  const [scanNote, setScanNote]       = useState('');
  const [filterTab, setFilterTab]     = useState('All');
  const [engine, setEngine]           = useState('astra_ai'); // Default to Aggressive AI for signals
  const [pushingId, setPushingId]     = useState(null);
  const [pushError, setPushError]     = useState(null);
  const [loading, setLoading]         = useState(false);

  // Auto-refresh when engine changes
  useEffect(() => {
    handleRefresh();
  }, [engine]);

  const filtered = predictions.filter(p => {
    if (filterTab === 'All') return true;
    if (filterTab === 'Stocks') return p.type === 'Equity';
    if (filterTab === 'F&O') return p.type === 'F&O';
    if (filterTab === 'Commodities') return p.type === 'Commodity';
    return true;
  });

  const handlePush = async (pred) => {
    if (pred.direction === 'HOLD') return; 
    setPushingId(pred.id);
    setPushError(null);

    const payload = {
      asset:        pred.asset,
      type:         pred.type,
      signal:       pred.direction,
      entry_price:  pred.entry,
      target_price: pred.target,
      stop_loss:    pred.stopLoss,
      confidence:   pred.confidence,
      engine:       pred.engine,
    };

    try {
      const res = await fetch(`${API_URL}/api/signals`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body:    JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Backend rejected the signal');
      const data = await res.json();

      setQueue(prev => [
        {
          id:          data.id ?? Date.now(),
          asset:       pred.asset,
          type:        pred.type,
          signal:      pred.direction,
          entryPrice:  pred.entry,
          targetPrice: pred.target,
          stopLoss:    pred.stopLoss,
          confidence:  pred.confidence,
          age:         'Just now',
        },
        ...prev,
      ]);

      setPredictions(prev => prev.filter(p => p.id !== pred.id));
    } catch (err) {
      console.error('Push to queue failed:', err);
      setPushError(`Failed to push ${pred.asset}: ${err.message}`);
    } finally {
      setPushingId(null);
    }
  };

  const handleRefresh = async () => {
    setLoading(true);
    setPushError(null);
    setScanNote('');
    try {
      const [eq, com] = await Promise.all([
        fetch(`${API_URL}/api/scan/universe?engine=${engine}&top_k=15&min_confidence=0`, { headers: authHeaders() })
          .then(r => (r.ok ? r.json() : { signals: [] })),
        fetch(`${API_URL}/api/commodities/scan`, { headers: authHeaders() })
          .then(r => (r.ok ? r.json() : { signals: [] })).catch(() => ({ signals: [] })),
      ]);
      const now = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
      const rows = [
        ...(eq.signals || []).map(x => ({
          id: `eq-${x.symbol}`, asset: x.symbol, direction: x.signal, entry: x.entry_price,
          target: x.target, stopLoss: x.stop_loss, confidence: Math.round(x.confidence),
          type: x.symbol?.startsWith('^') ? 'F&O' : 'Equity', time: `Scanned ${now}`, engine,
        })),
        ...(com.signals || []).filter(x => x.signal && x.signal !== 'HOLD').map(x => ({
          id: `c-${x.symbol}`, asset: x.symbol, direction: x.signal, entry: x.entry_price,
          target: x.target, stopLoss: x.stop_loss, confidence: Math.round(x.confidence),
          type: 'Commodity', time: `Scanned ${now}`, engine: 'commodities',
        })),
      ].filter(r => r.direction && r.direction !== 'HOLD')
       .filter(r => !queue.some(q => q.asset === r.asset))
       .sort((a, b) => b.confidence - a.confidence);
      setPredictions(rows);
      if (!rows.length) setScanNote(`No BUY/SELL setups from ${eq.universe_size || 0} stocks right now. The model is holding, not failing.`);
    } catch (e) {
      setPushError(`Scan failed: ${e.message}. Is the backend running?`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="topbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>AI Trading Signals</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
            Probability-weighted predictions using <strong>{engine === 'astra' ? 'Astra 1.0 (Safe)' : engine === 'astra_ai' ? 'Astra.ai 1.0 (Aggressive)' : 'Astra.ml (Deep)'}</strong>.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          
          {/* Model Selector Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--panel-bg)', padding: '4px 10px', borderRadius: 10, border: '1px solid rgba(255,255,255,.08)' }}>
            <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,.4)', fontWeight: 600 }}>Model:</span>
            <select 
              value={engine} 
              onChange={(e) => setEngine(e.target.value)}
              style={{ background: 'transparent', border: 'none', color: '#60a5fa', fontSize: '0.8rem', fontWeight: 700, outline: 'none', cursor: 'pointer' }}
            >
              <option value="astra">Astra 1.0 (Safe)</option>
              <option value="astra_ai">Astra.ai 1.0 (Aggressive)</option>
              <option value="astra_ml">Astra.ml (Deep ANN)</option>
            </select>
          </div>

          {/* Filter tabs */}
          <div style={{ display: 'flex', background: 'var(--panel-bg)', borderRadius: '12px', padding: '4px' }}>
            {['All', 'Stocks', 'F&O', 'Commodities'].map(tab => (
              <button key={tab} onClick={() => setFilterTab(tab)} style={{
                padding: '7px 14px', borderRadius: '8px', border: 'none', fontWeight: 600, cursor: 'pointer', fontSize: '0.82rem', transition: '0.2s',
                background: filterTab === tab ? 'hsla(210,100%,55%,.15)' : 'transparent',
                color:      filterTab === tab ? 'var(--color-primary)' : 'var(--text-secondary)',
              }}>
                {tab}
              </button>
            ))}
          </div>
          <button onClick={handleRefresh} disabled={loading} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,.1)', color: 'rgba(255,255,255,.5)', padding: '7px 12px', borderRadius: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', opacity: loading ? 0.6 : 1 }}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> {loading ? 'Scanning…' : 'Refresh'}
          </button>
        </div>
      </div>

      {pushError && (
        <div style={{ margin: '12px 0', padding: '10px 16px', borderRadius: 10, background: 'rgba(239,68,68,.08)', border: '1px solid rgba(239,68,68,.3)', color: '#f87171', fontSize: '0.85rem' }}>
          ⚠ {pushError}
        </div>
      )}

      <div className="glass-panel" style={{ marginTop: '20px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'hsla(222,47%,8%,.5)', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
              {['Asset', 'Type', 'Signal', 'Entry', 'Target', 'Stop Loss', 'Confidence', 'Action'].map(h => (
                <th key={h} style={{ padding: '14px 20px', fontWeight: 500 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  {loading ? 'Scanning the market with live data…' : (scanNote || 'No open BUY/SELL signals in this category.')}{' '}
                  <span style={{ color: 'var(--color-primary)', cursor: 'pointer' }} onClick={handleRefresh}>Refresh signals</span>
                </td>
              </tr>
            ) : filtered.map(pred => (
              <tr key={pred.id} style={{ borderBottom: '1px solid var(--panel-border)' }} className="table-row-hover">
                <td style={{ padding: '14px 20px', fontWeight: 600 }}>{pred.asset}</td>
                <td style={{ padding: '14px 20px', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  <span style={{ padding: '3px 8px', borderRadius: 6, background: 'rgba(255,255,255,.06)' }}>{pred.type}</span>
                </td>
                <td style={{ padding: '14px 20px' }}>
                  <span style={{ padding: '5px 12px', borderRadius: 20, fontSize: '0.82rem', fontWeight: 700, background: `${signalColor(pred.direction)}18`, color: signalColor(pred.direction) }}>
                    {pred.direction}
                  </span>
                </td>
                <td style={{ padding: '14px 20px' }}>{money(pred, pred.entry)}</td>
                <td style={{ padding: '14px 20px', color: '#22c55e', fontWeight: 600 }}>{money(pred, pred.target)}</td>
                <td style={{ padding: '14px 20px', color: '#ef4444', fontWeight: 600 }}>{money(pred, pred.stopLoss)}</td>
                <td style={{ padding: '14px 20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 56, height: 5, background: 'rgba(255,255,255,.1)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${pred.confidence}%`, height: '100%', background: pred.confidence > 75 ? '#22c55e' : pred.confidence > 60 ? '#f59e0b' : '#ef4444' }} />
                    </div>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{pred.confidence}%</span>
                  </div>
                </td>
                <td style={{ padding: '14px 20px' }}>
                  {pred.direction === 'HOLD' ? (
                    <span style={{ color: '#f59e0b', fontSize: '0.82rem' }}>Monitoring…</span>
                  ) : (
                    <button
                      onClick={() => handlePush(pred)}
                      disabled={pushingId === pred.id}
                      className="btn-primary"
                      style={{ padding: '6px 14px', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 6, opacity: pushingId === pred.id ? 0.7 : 1 }}
                    >
                      <Zap size={13} />
                      {pushingId === pred.id ? 'Pushing…' : 'Push to Queue'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Queue count */}
      <div style={{ marginTop: 16, textAlign: 'right', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
        {queue.length} signal{queue.length !== 1 ? 's' : ''} currently in Auto Mode queue
      </div>

      <style>{`.table-row-hover:hover { background: hsla(210,100%,55%,.04); }`}</style>
    </div>
  );
}
