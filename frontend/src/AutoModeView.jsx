import React, { useState } from 'react';
import { Check, X, AlertCircle } from 'lucide-react';

export default function AutoModeView({ queue, setQueue, history, setHistory }) {
  const [autoExecute, setAutoExecute] = useState(false);

  const handleAction = (id, action) => {
    const trade = queue.find(q => q.id === id);
    setQueue(queue.filter(q => q.id !== id));

    if (action === 'approve' && trade) {
      // ✅ Optimistically add to history immediately — don't wait for backend
      const optimisticEntry = {
        id:     `TRD-${Date.now()}`,
        asset:  `${trade.asset} (${trade.type || 'Equity'})`,
        action: trade.signal,
        price:  trade.targetPrice,
        qty:    50,
        pnl:    0,
        time:   new Date().toLocaleString('en-IN'),
        status: 'Executed (AI Signal)',
      };
      setHistory(prev => [optimisticEntry, ...prev]);

      // Also persist to backend (fire and forget)
      fetch('http://localhost:8000/api/execute', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          id: trade.id, 
          asset: trade.asset, 
          action: trade.signal, 
          quantity: 50, 
          price: trade.targetPrice,
          target_price: trade.targetPrice,
          stop_loss: trade.stopLoss
        }),
      })
      .then(res => res.json())
      .then(result => {
        if (result.order_id) {
          // Update the optimistic entry with the real order ID
          setHistory(prev => prev.map(h => h.id === optimisticEntry.id ? { ...h, id: result.order_id } : h));
        }
      })
      .catch(err => console.warn('Backend execute failed (trade still recorded locally):', err));
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="topbar">
        <div>
          <h2>Auto-Mode Queue</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
            Human-in-the-loop approval for AI-generated trades.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontWeight: 600, color: autoExecute ? 'var(--color-danger)' : 'var(--text-secondary)' }}>
            {autoExecute ? "Fully Autonomous Active" : "Human Approval Required"}
          </span>
          <button 
            className="btn-primary" 
            onClick={() => setAutoExecute(!autoExecute)}
            style={{ background: autoExecute ? 'var(--color-danger)' : 'var(--panel-border)' }}
          >
            {autoExecute ? "Disable Auto-Execute" : "Enable Auto-Execute"}
          </button>
        </div>
      </div>

      {autoExecute && (
        <div style={{ 
          background: 'hsla(348, 80%, 55%, 0.1)', 
          border: '1px solid var(--color-danger)', 
          padding: '16px', 
          borderRadius: '8px', 
          marginTop: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          color: 'var(--color-danger)'
        }}>
          <AlertCircle size={20} />
          <span><strong>WARNING:</strong> Trades will execute automatically without human verification. Risk of capital loss.</span>
        </div>
      )}

      <div style={{ marginTop: '32px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {queue.length === 0 ? (
          <div className="glass-panel" style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            No pending trades in the AI queue. The AI is still monitoring the market.
          </div>
        ) : queue.map(trade => (
          <div key={trade.id} className="glass-panel" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <span style={{ 
                  background: trade.signal === 'BUY' ? 'hsla(142, 70%, 50%, 0.15)' : 'hsla(348, 80%, 55%, 0.15)',
                  color: trade.signal === 'BUY' ? 'var(--color-success)' : 'var(--color-danger)',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontWeight: 600,
                  fontSize: '0.8rem'
                }}>
                  {trade.signal}
                </span>
                <span style={{ fontSize: '1.2rem', fontWeight: 600 }}>{trade.asset}</span>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>({trade.type})</span>
              </div>
              <div style={{ display: 'flex', gap: '24px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                <span>Target: <strong style={{color: 'var(--text-primary)'}}>{trade.targetPrice}</strong></span>
                <span>Stop: <strong style={{color: 'var(--text-primary)'}}>{trade.stopLoss}</strong></span>
                <span>AI Confidence: <strong style={{color: 'var(--text-primary)'}}>{trade.confidence}%</strong></span>
                <span>Generated: {trade.age}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                onClick={() => handleAction(trade.id, 'reject')}
                className="btn-primary" 
                style={{ background: 'transparent', border: '1px solid var(--panel-border)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <X size={16} /> Reject
              </button>
              <button 
                onClick={() => handleAction(trade.id, 'approve')}
                className="btn-primary" 
                style={{ background: 'var(--color-success)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Check size={16} /> Approve & Execute
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
