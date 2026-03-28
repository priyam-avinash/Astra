import React, { useState, useEffect } from 'react';
import { Play, Square, TrendingUp, TrendingDown, Clock, Activity, RefreshCw } from 'lucide-react';

export default function ActiveTradesView({ positions, setPositions }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [squaringOff, setSquaringOff] = useState(null);

  const fetchPositions = async () => {
    try {
      const r = await fetch('http://localhost:8000/api/positions');
      if (!r.ok) throw new Error('Failed to fetch positions');
      const d = await r.json();
      setPositions(d.positions || []);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleSquareOff = async (id) => {
    setSquaringOff(id);
    try {
      const r = await fetch(`http://localhost:8000/api/positions/squareoff/${id}`, {
        method: 'POST'
      });
      if (!r.ok) throw new Error('Failed to square off position');
      await fetchPositions();
    } catch (e) {
      console.error(e);
      alert('Square off failed: ' + e.message);
    } finally {
      setSquaringOff(null);
    }
  };

  if (loading && positions.length === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <RefreshCw className="animate-spin" size={32} color="var(--color-primary)" />
      </div>
    );
  }

  const totalPnl = positions.reduce((acc, p) => acc + p.pnl, 0);

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
        <div className="glass-panel" style={{ padding: 20 }}>
          <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', marginBottom: 8 }}>Active Positions</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{positions.length}</div>
        </div>
        <div className="glass-panel" style={{ padding: 20 }}>
          <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', marginBottom: 8 }}>Live Unrealized P&L</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: totalPnl >= 0 ? '#22c55e' : '#ef4444' }}>
            ₹{totalPnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </div>
        </div>
        <div className="glass-panel" style={{ padding: 20 }}>
          <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', marginBottom: 8 }}>Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '1.1rem', fontWeight: 600, color: '#60a5fa' }}>
            <Activity size={18} /> Monitoring Live
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="glass-panel" style={{ border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Active Trades</h3>
          <button onClick={fetchPositions} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'pointer' }}>
            <RefreshCw size={14} />
          </button>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.02)', fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)' }}>
                <th style={{ padding: '12px 20px' }}>Asset</th>
                <th style={{ padding: '12px 20px' }}>Type</th>
                <th style={{ padding: '12px 20px' }}>Qty</th>
                <th style={{ padding: '12px 20px' }}>Avg. Price</th>
                <th style={{ padding: '12px 20px' }}>LTP</th>
                <th style={{ padding: '12px 20px' }}>Tgt / SL</th>
                <th style={{ padding: '12px 20px' }}>P&L</th>
                <th style={{ padding: '12px 20px' }}>Time</th>
                <th style={{ padding: '12px 20px', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td colSpan="9" style={{ padding: '40px 20px', textAlign: 'center', color: 'rgba(255,255,255,0.2)' }}>
                    No active positions. Execute a trade from Auto Mode or Manual order.
                  </td>
                </tr>
              ) : (
                positions.map((p) => (
                  <tr key={p.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '0.9rem' }}>
                    <td style={{ padding: '16px 20px', fontWeight: 600 }}>{p.asset}</td>
                    <td style={{ padding: '16px 20px' }}>
                      <span style={{ 
                        color: p.direction === 'BUY' ? '#22c55e' : '#ef4444', 
                        background: p.direction === 'BUY' ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                        padding: '2px 8px', borderRadius: 4, fontSize: '0.75rem', fontWeight: 700 
                      }}>
                        {p.direction}
                      </span>
                    </td>
                    <td style={{ padding: '16px 20px' }}>{p.quantity}</td>
                    <td style={{ padding: '16px 20px' }}>₹{p.entry_price.toFixed(2)}</td>
                    <td style={{ padding: '16px 20px', fontWeight: 700 }}>₹{p.current_price.toFixed(2)}</td>
                    <td style={{ padding: '16px 20px' }}>
                      <div style={{ fontSize: '0.8rem', color: '#22c55e' }}>T: {p.target_price ? `₹${p.target_price.toFixed(2)}` : '—'}</div>
                      <div style={{ fontSize: '0.8rem', color: '#ef4444' }}>S: {p.stop_loss ? `₹${p.stop_loss.toFixed(2)}` : '—'}</div>
                    </td>
                    <td style={{ padding: '16px 20px' }}>
                      <div style={{ color: p.pnl >= 0 ? '#22c55e' : '#ef4444', fontWeight: 700 }}>
                        {p.pnl >= 0 ? '+' : ''}₹{p.pnl.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: p.pnl >= 0 ? '#22c55e' : '#ef4444', opacity: 0.8 }}>
                        {p.pnl_pct.toFixed(2)}%
                      </div>
                    </td>
                    <td style={{ padding: '16px 20px', color: 'rgba(255,255,255,0.4)', fontSize: '0.8rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Clock size={12} /> {p.time}
                      </div>
                    </td>
                    <td style={{ padding: '16px 20px', textAlign: 'right' }}>
                      <button 
                        onClick={() => handleSquareOff(p.id)}
                        disabled={squaringOff === p.id}
                        style={{ 
                          background: 'rgba(255,255,255,0.05)', 
                          border: '1px solid rgba(255,255,255,0.1)', 
                          color: '#fff', 
                          padding: '6px 12px', 
                          borderRadius: 6, 
                          fontSize: '0.8rem', 
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 6
                        }}
                        onMouseOver={(e) => e.currentTarget.style.background = 'rgba(239,68,68,0.2)'}
                        onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                      >
                        {squaringOff === p.id ? <RefreshCw size={14} className="animate-spin" /> : <Square size={14} />}
                        Square Off
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: 20, background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.1)' }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#60a5fa' }}>💡 Strategy Note</h4>
        <p style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.5)', lineHeight: 1.5 }}>
          Positions are tracked in real-time. Square off manually or wait for the AI to detect a trend reversal. 
          When a trade is squared off, it is moved to Trade History and real-time monitoring stops.
        </p>
      </div>
    </div>
  );
}
