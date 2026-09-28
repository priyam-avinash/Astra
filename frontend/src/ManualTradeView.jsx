import React, { useState, useEffect, useMemo } from 'react';
import { Send, Activity, DollarSign, List, BarChart2, Clock } from 'lucide-react';

import { API_URL } from './config';

// v1.13: live quote panel replaces the random "market depth" mock
const authHeaders = () => {
  const t = localStorage.getItem('astra_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};

export default function ManualTradeView() {
  const [assetType, setAssetType] = useState('Equity'); // New: Equity vs Commodity vs F&O
  const [asset, setAsset] = useState('RELIANCE.NS');
  const [action, setAction] = useState('BUY');
  const [quantity, setQuantity] = useState(10);
  const [price, setPrice] = useState('');
  const [targetPrice, setTargetPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [quote, setQuote] = useState(null);
  const [orderMsg, setOrderMsg] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const suggestions = useMemo(() => ({
    Equity: ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS'],
    Commodity: ['GC=F', 'CL=F', 'SI=F', 'HG=F', 'NG=F'],
    'F&O': ['^NSEI', '^NSEBANK', 'NIFTYBEES.NS', 'BANKBEES.NS']   // index / index-ETF proxies (paper)
  }), []);

  useEffect(() => {
    // Update default asset when type changes
    setAsset(suggestions[assetType][0]);
  }, [assetType, suggestions]);

  // Live quote for the selected asset (refresh every 10s); pre-fills the price
  useEffect(() => {
    let alive = true, first = true;
    const load = async () => {
      try {
        const r = await fetch(`${API_URL}/api/quote/${encodeURIComponent(asset.toUpperCase())}`, { headers: authHeaders() });
        const q = r.ok ? await r.json() : null;
        if (!alive || !q) return;
        setQuote(q);
        if (first && q.price > 0) { setPrice(q.price); first = false; }
      } catch (_) { if (alive) setQuote(null); }
    };
    setQuote(null); setPrice('');
    const t = setTimeout(load, 300);          // debounce typing
    const itv = setInterval(load, 10000);
    return () => { alive = false; clearTimeout(t); clearInterval(itv); };
  }, [asset]);

  useEffect(() => {
    const fetchHistory = () => {
      const headers = authHeaders();
      fetch(`${API_URL}/api/history`, { headers })
      .then(r => r.json())
      .then(d => setHistory(d.history?.filter(h => h.status && h.status.includes('Manual')) || []));
    };
    fetchHistory();
    const hitv = setInterval(fetchHistory, 5000);
    return () => { clearInterval(hitv); };
  }, []);

  const handleExecute = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const headers = { 'Content-Type': 'application/json', ...authHeaders() };
      
      const resp = await fetch(`${API_URL}/api/execute/manual`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          asset: asset.toUpperCase(),
          action,
          quantity: parseInt(quantity),
          price: parseFloat(price),
          target_price: targetPrice ? parseFloat(targetPrice) : null,
          stop_loss: stopLoss ? parseFloat(stopLoss) : null
        })
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body.detail || `Order rejected (HTTP ${resp.status})`);
      setOrderMsg({ ok: true, text: `Paper order filled: ${action} ${quantity} ${asset.toUpperCase()} @ ₹${Number(body.executed_price).toFixed(2)} (${body.order_id})` });
    } catch (err) {
      setOrderMsg({ ok: false, text: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateRows: 'auto 1fr', gap: '24px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
        {/* EXECUTION FORM */}
        <div className="glass-panel" style={{ padding: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: '24px' }}>
            <div style={{ width: 40, height: 40, borderRadius: '50%', background: 'rgba(59,130,246,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Send size={20} color="var(--color-primary)" />
            </div>
            <h3 style={{ margin: 0 }}>Direct Order Entry</h3>
          </div>
          
          <form onSubmit={handleExecute} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', background: 'rgba(255,255,255,0.03)', padding: '4px', borderRadius: '8px' }}>
              <button type="button" onClick={()=>setAssetType('Equity')} style={{ padding: '8px', borderRadius: '6px', border: 'none', cursor: 'pointer', background: assetType==='Equity'?'rgba(59,130,246,0.15)':'transparent', color: assetType==='Equity'?'#60a5fa':'#666', fontSize: '0.75rem', fontWeight: 700 }}>STOCK EQUITY</button>
              <button type="button" onClick={()=>setAssetType('Commodity')} style={{ padding: '8px', borderRadius: '6px', border: 'none', cursor: 'pointer', background: assetType==='Commodity'?'rgba(245,158,11,0.15)':'transparent', color: assetType==='Commodity'?'#f59e0b':'#666', fontSize: '0.75rem', fontWeight: 700 }}>COMMODITIES</button>
              <button type="button" onClick={()=>setAssetType('F&O')} style={{ padding: '8px', borderRadius: '6px', border: 'none', cursor: 'pointer', background: assetType==='F&O'?'rgba(167,139,250,0.15)':'transparent', color: assetType==='F&O'?'#a78bfa':'#666', fontSize: '0.75rem', fontWeight: 700 }}>F&O</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', background: 'rgba(255,255,255,0.03)', padding: '4px', borderRadius: '8px' }}>
              <button type="button" onClick={()=>setAction('BUY')} style={{ padding: '10px', borderRadius: '6px', border: 'none', cursor: 'pointer', background: action==='BUY'?'rgba(34,197,94,0.15)':'transparent', color: action==='BUY'?'#22c55e':'#666', fontWeight: 700 }}>BUY</button>
              <button type="button" onClick={()=>setAction('SELL')} style={{ padding: '10px', borderRadius: '6px', border: 'none', cursor: 'pointer', background: action==='SELL'?'rgba(239,68,68,0.15)':'transparent', color: action==='SELL'?'#ef4444':'#666', fontWeight: 700 }}>SELL</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="input-group">
                <label>SYMBOL</label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input value={asset} onChange={e=>setAsset(e.target.value)} list="symbol-suggestions" style={{ flex: 1, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', color: 'white', padding: '12px', borderRadius: '8px' }}/>
                  <datalist id="symbol-suggestions">
                    {suggestions[assetType].map(s => <option key={s} value={s} />)}
                  </datalist>
                </div>
              </div>
              <div className="input-group">
                <label>QTY</label>
                <input type="number" value={quantity} onChange={e=>setQuantity(e.target.value)} style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', color: 'white', padding: '12px', borderRadius: '8px' }}/>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
              <div className="input-group">
                <label>LIMIT PRICE</label>
                <input type="number" value={price} onChange={e=>setPrice(e.target.value)} step="0.05" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', color: 'white', padding: '10px', borderRadius: '8px' }}/>
              </div>
              <div className="input-group">
                <label>TARGET</label>
                <input type="number" value={targetPrice} onChange={e=>setTargetPrice(e.target.value)} placeholder="Opt" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', color: 'white', padding: '10px', borderRadius: '8px' }}/>
              </div>
              <div className="input-group">
                <label>STOPLOSS</label>
                <input type="number" value={stopLoss} onChange={e=>setStopLoss(e.target.value)} placeholder="Opt" style={{ width: '100%', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', color: 'white', padding: '10px', borderRadius: '8px' }}/>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px', borderRadius: '8px', background: 'rgba(59,130,246,0.05)', marginTop: '8px' }}>
              <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.85rem' }}>Estimated Margin:</span>
              <span style={{ fontWeight: 700 }}>₹{(quantity * price).toLocaleString()}</span>
            </div>

            <button type="submit" className="btn-primary" disabled={loading} style={{ width: '100%', padding: '14px', background: action==='BUY'?'#22c55e':'#ef4444' }}>
              {loading ? 'Processing...' : `Place ${action} Order`}
            </button>
          </form>
        </div>

        {/* LIVE QUOTE */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '20px' }}>
            <BarChart2 size={18} color="rgba(255,255,255,0.4)" />
            <h4 style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.4)' }}>LIVE QUOTE · {asset.toUpperCase()}</h4>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, marginBottom: 6 }}>
            {quote?.price > 0 ? `₹${Number(quote.price).toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : (quote ? 'Unavailable' : 'Loading…')}
          </div>
          <div style={{ fontSize: '0.75rem', color: quote?.stale ? '#f59e0b' : 'rgba(255,255,255,0.4)' }}>
            {quote?.source ? `Source: ${quote.source}${quote.stale ? ' (stale — orders will be rejected)' : ''}` : ''}
            {quote?.ts ? ` · ${quote.ts.slice(11, 19)}` : ''}
          </div>
          <div style={{ marginTop: 20, padding: 12, background: 'rgba(255,255,255,0.02)', borderRadius: 8, fontSize: '0.75rem', color: 'rgba(255,255,255,0.45)', lineHeight: 1.6 }}>
            Paper trading: orders fill at the live price ± 0.05% slippage, with 0.03% brokerage. No real orders are placed.
          </div>
          {orderMsg && (
            <div style={{ marginTop: 16, padding: 12, borderRadius: 8, fontSize: '0.8rem',
                          background: orderMsg.ok ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
                          color: orderMsg.ok ? '#22c55e' : '#ef4444' }}>{orderMsg.text}</div>
          )}
        </div>
      </div>

      {/* ORDER BOOK */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: 8 }}>
          <Clock size={16} color="rgba(255,255,255,0.4)" />
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Manual Order Book</span>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.01)', color: 'rgba(255,255,255,0.3)', fontSize: '0.75rem' }}>
              <th style={{ padding: '12px 24px' }}>TIME</th>
              <th style={{ padding: '12px 24px' }}>ASSET</th>
              <th style={{ padding: '12px 24px' }}>ACTION</th>
              <th style={{ padding: '12px 24px' }}>PRICE</th>
              <th style={{ padding: '12px 24px' }}>QTY</th>
              <th style={{ padding: '12px 24px' }}>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr><td colSpan="6" style={{ padding: '32px', textAlign: 'center', color: 'rgba(255,255,255,0.2)' }}>No manual orders placed in this session.</td></tr>
            ) : (
              history.map((h, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', fontSize: '0.85rem' }}>
                  <td style={{ padding: '14px 24px', color: 'rgba(255,255,255,0.4)' }}>{h.time}</td>
                  <td style={{ padding: '14px 24px', fontWeight: 600 }}>{h.asset}</td>
                  <td style={{ padding: '14px 24px', color: h.action==='BUY'?'#22c55e':'#ef4444' }}>{h.action}</td>
                  <td style={{ padding: '14px 24px' }}>₹{h.price.toFixed(2)}</td>
                  <td style={{ padding: '14px 24px' }}>{h.quantity}</td>
                  <td style={{ padding: '14px 24px' }}>
                    <span style={{ background: 'rgba(59,130,246,0.1)', color: '#60a5fa', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem' }}>EXECUTED</span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <style>{`
        .input-group label { display: block; fontSize: 0.65rem; color: rgba(255,255,255,0.3); margin-bottom: 6px; letter-spacing: 0.05em; }
        .input-group input:focus { border-color: var(--color-primary); outline: none; background: rgba(59,130,246,0.05); }
      `}</style>
    </div>
  );
}
