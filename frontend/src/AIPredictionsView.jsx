import React, { useState, useEffect } from 'react';
import { Zap, RefreshCw } from 'lucide-react';

/**
 * AI Predictions View
 * 
 * This view shows AI-generated signals that have NOT yet been pushed to the Auto Mode queue.
 * When the user clicks "Push to Queue", the signal is:
 * 1. Sent to the backend via POST /api/signals
 * 2. Added to the shared `queue` state so Auto Mode immediately sees it
 * 3. Removed from this local predictions list
 */

const MOCK_SEEDS = [
  { id: 'm1', asset: 'RELIANCE.NS', direction: 'BUY',  entry: 1381, target: 1480, stopLoss: 1350, confidence: 74, type: 'Equity',    time: '5 mins ago' },
  { id: 'm2', asset: 'TCS.NS',      direction: 'SELL', entry: 3960, target: 3750, stopLoss: 4050, confidence: 68, type: 'Equity',    time: '18 mins ago' },
  { id: 'm3', asset: 'GC=F',        direction: 'BUY',  entry: 2678, target: 2720, stopLoss: 2580, confidence: 71, type: 'Commodity', time: '1 hr ago' },
  { id: 'm4', asset: '^NSEI',       direction: 'HOLD', entry: 23200, target: 23500, stopLoss: 22000, confidence: 55, type: 'F&O',   time: '2 hrs ago' },
  { id: 'm5', asset: 'HDFCBANK.NS', direction: 'SELL', entry: 1652, target: 1580, stopLoss: 1720, confidence: 61, type: 'Equity',   time: '3 hrs ago' },
];

const signalColor = (d) => d === 'BUY' ? '#22c55e' : d === 'SELL' ? '#ef4444' : '#f59e0b';

export default function AIPredictionsView({ queue, setQueue }) {
  const [predictions, setPredictions] = useState(MOCK_SEEDS);
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
      asset:       pred.asset,
      type:        pred.type,
      signal:      pred.direction,
      targetPrice: pred.target,
      stopLoss:    pred.stopLoss,
      confidence:  pred.confidence,
    };

    try {
      const res = await fetch('http://localhost:8000/api/signals', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
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
    
    // Simulate a fresh scan with the chosen engine
    // In a real app, this would call /api/signals/scan?engine=...
    // For now, we'll just simulate a delay and slightly randomize seeds to show "model switching" effect
    setTimeout(() => {
      const fresh = MOCK_SEEDS.map(s => ({
        ...s,
        confidence: Math.round(s.confidence + (Math.random() * 10 - 5)),
        time: 'Just now'
      })).filter(s => !queue.some(q => q.asset === s.asset));
      
      setPredictions(fresh);
      setLoading(false);
    }, 800);
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
                  All signals in this category have been pushed to Auto Mode queue.{' '}
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
                <td style={{ padding: '14px 20px' }}>₹{pred.entry.toLocaleString('en-IN')}</td>
                <td style={{ padding: '14px 20px', color: '#22c55e', fontWeight: 600 }}>₹{pred.target.toLocaleString('en-IN')}</td>
                <td style={{ padding: '14px 20px', color: '#ef4444', fontWeight: 600 }}>₹{pred.stopLoss.toLocaleString('en-IN')}</td>
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
