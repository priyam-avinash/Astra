import React, { useState, useEffect, useMemo } from 'react';
import { Send, Activity, DollarSign, List, BarChart2, Clock } from 'lucide-react';

const mockDepth = () => ({
  bids: Array.from({length: 5}, (_, i) => ({ price: 2950 - i*2, qty: Math.floor(Math.random()*500) + 100 })),
  asks: Array.from({length: 5}, (_, i) => ({ price: 2955 + i*2, qty: Math.floor(Math.random()*500) + 100 })),
});

export default function ManualTradeView() {
  const [assetType, setAssetType] = useState('Equity'); // New: Equity vs Commodity vs F&O
  const [asset, setAsset] = useState('RELIANCE.NS');
  const [action, setAction] = useState('BUY');
  const [quantity, setQuantity] = useState(10);
  const [price, setPrice] = useState(2950.0);
  const [targetPrice, setTargetPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [depth, setDepth] = useState(mockDepth());
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const suggestions = useMemo(() => ({
    Equity: ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS'],
    Commodity: ['GC=F', 'CL=F', 'SI=F', 'HG=F', 'NG=F'],
    'F&O': ['NIFTY24MARFUT', 'BANKNIFTY24MARFUT', 'FINNIFTY24MARFUT', 'RELIANCE24MARFUT']
  }), []);

  useEffect(() => {
    // Update default asset when type changes
    setAsset(suggestions[assetType][0]);
  }, [assetType, suggestions]);

  useEffect(() => {
    const itv = setInterval(() => setDepth(mockDepth()), 2000);
    const fetchHistory = () => {
      const token = localStorage.getItem('token');
      const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
      fetch('http://localhost:8000/api/history', { headers })
      .then(r => r.json())
      .then(d => setHistory(d.history?.filter(h => h.status && h.status.includes('Manual')) || []));
    };
    fetchHistory();
    const hitv = setInterval(fetchHistory, 5000);
    return () => { clearInterval(itv); clearInterval(hitv); };
  }, []);

  const handleExecute = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = `Bearer ${token}`;
      
      const resp = await fetch('http://localhost:8000/api/execute/manual', {
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
      if (!resp.ok) throw new Error('Execution Failed');
      alert(`Order Placed: ${action} ${quantity} ${asset}`);
    } catch (err) {
      alert(err.message);
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

        {/* MARKET DEPTH */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '20px' }}>
            <BarChart2 size={18} color="rgba(255,255,255,0.4)" />
            <h4 style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.4)' }}>MARKET DEPTH (L2)</h4>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1px', background: 'rgba(255,255,255,0.05)' }}>
            <div style={{ background: 'var(--panel-bg)', padding: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.3)', marginBottom: 8 }}>BIDS (BUY)</div>
              {depth.bids.map((b,i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 4, position: 'relative' }}>
                  <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, background: 'rgba(34,197,94,0.05)', width: `${(b.qty/500)*100}%`, zIndex: 0 }}/>
                  <span style={{ color: '#22c55e', position: 'relative' }}>{b.price.toFixed(2)}</span>
                  <span style={{ position: 'relative' }}>{b.qty}</span>
                </div>
              ))}
            </div>
            <div style={{ background: 'var(--panel-bg)', padding: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.3)', marginBottom: 8 }}>ASKS (SELL)</div>
              {depth.asks.map((a,i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 4, position: 'relative' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, background: 'rgba(239,68,68,0.05)', width: `${(a.qty/500)*100}%`, zIndex: 0 }}/>
                  <span style={{ color: '#ef4444', position: 'relative' }}>{a.price.toFixed(2)}</span>
                  <span style={{ position: 'relative' }}>{a.qty}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ marginTop: '16px', textAlign: 'center', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.3)', marginRight: 8 }}>SPREAD:</span>
            <span style={{ fontWeight: 700, color: 'var(--color-primary)' }}>₹5.00 (0.17%)</span>
          </div>
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
