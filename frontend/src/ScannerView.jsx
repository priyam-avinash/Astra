import React, { useState, useCallback } from 'react';
import { Search, RefreshCw, Telescope, TrendingUp, TrendingDown,
         CheckCircle, AlertCircle, Filter, ChevronDown } from 'lucide-react';

const ENGINES = [
  { value: 'astra_ai', label: 'ASTRA.AI (Random Forest)' },
  { value: 'astra_ml', label: 'ASTRA.ML (BiLSTM)' },
  { value: 'astra',    label: 'ASTRA 1.0 (Rules)' },
];

const SIGNALS = [
  { value: '',     label: 'All Signals' },
  { value: 'BUY',  label: 'BUY Only' },
  { value: 'SELL', label: 'SELL Only' },
];

const SECTORS = [
  '', 'Banking', 'IT', 'Pharma', 'Energy', 'FMCG', 'Auto', 'Metals',
  'NBFC', 'Telecom', 'Engineering', 'Cement', 'Consumer', 'Paints',
  'Power', 'Insurance', 'Healthcare', 'Logistics', 'Agrochem', 'Other',
];

export default function ScannerView() {
  const [engine,   setEngine]   = useState('astra_ai');
  const [signal,   setSignal]   = useState('');
  const [sector,   setSector]   = useState('');
  const [topK,     setTopK]     = useState(20);
  const [minConf,  setMinConf]  = useState(55);

  const [results,  setResults]  = useState([]);
  const [loading,  setLoading]  = useState(false);
  const [scanned,  setScanned]  = useState(null);
  const [error,    setError]    = useState(null);
  const [approved, setApproved] = useState(new Set());

  const scan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResults([]);
    setApproved(new Set());
    try {
      let url = `http://localhost:8000/api/scan/universe?engine=${engine}&top_k=${topK}&min_confidence=${minConf}`;
      if (signal)  url += `&signal_filter=${signal}`;
      if (sector)  url += `&sector=${encodeURIComponent(sector)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResults(data.signals || []);
      setScanned(data.universe_size || 141);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }, [engine, signal, sector, topK, minConf]);

  const approveSignal = useCallback(async (sig) => {
    try {
      const res = await fetch('http://localhost:8000/api/signals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset:          sig.symbol,
          signal:         sig.signal,
          confidence:     sig.confidence,
          entry_price:    sig.entry_price,
          target_price:   sig.target,
          stop_loss:      sig.stop_loss,
          engine:         engine,
          signal_type:    engine.toUpperCase(),
        }),
      });
      if (res.ok) {
        setApproved(prev => new Set([...prev, sig.symbol]));
      }
    } catch (e) {
      console.error('Approve failed', e);
    }
  }, [engine]);

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <h2 className="page-title">Universe Scanner</h2>
            <p className="page-subtitle">
              Scan {scanned ? scanned.toLocaleString() : '141'} NSE stocks simultaneously for AI-powered signals
            </p>
          </div>
          <button
            className="btn-primary"
            onClick={scan}
            disabled={loading}
            style={{ flexShrink: 0 }}
          >
            {loading
              ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Scanning…</>
              : <><Search size={14} /> Scan Universe</>
            }
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card" style={{ padding: '16px 20px', marginBottom: 20 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
          {/* Engine */}
          <div style={{ flex: '1 1 180px' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Engine
            </label>
            <select className="select-field" value={engine} onChange={e => setEngine(e.target.value)}>
              {ENGINES.map(e => <option key={e.value} value={e.value}>{e.label}</option>)}
            </select>
          </div>

          {/* Signal */}
          <div style={{ flex: '1 1 130px' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Direction
            </label>
            <select className="select-field" value={signal} onChange={e => setSignal(e.target.value)}>
              {SIGNALS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>

          {/* Sector */}
          <div style={{ flex: '1 1 150px' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Sector
            </label>
            <select className="select-field" value={sector} onChange={e => setSector(e.target.value)}>
              {SECTORS.map(s => <option key={s} value={s}>{s || 'All Sectors'}</option>)}
            </select>
          </div>

          {/* Top K */}
          <div style={{ flex: '1 1 140px' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Results: <span style={{ color: 'var(--accent)' }}>{topK}</span>
            </label>
            <input
              type="range" min="5" max="50" step="5"
              value={topK}
              onChange={e => setTopK(+e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
            />
          </div>

          {/* Min Confidence */}
          <div style={{ flex: '1 1 140px' }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Min Conf: <span style={{ color: 'var(--accent)' }}>{minConf}%</span>
            </label>
            <input
              type="range" min="40" max="85" step="5"
              value={minConf}
              onChange={e => setMinConf(+e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
            />
          </div>
        </div>
      </div>

      {/* Spin animation */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Error */}
      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '14px 18px',
          background: 'var(--red-dim)', border: '1px solid rgba(244,63,94,0.2)',
          borderRadius: 14, marginBottom: 16, fontSize: 13, color: 'var(--red)'
        }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Loading skeletons */}
      {loading && (
        <div className="scanner-grid">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="scanner-card" style={{ gap: 0 }}>
              <div className="skeleton" style={{ height: 20, width: '60%', marginBottom: 10 }} />
              <div className="skeleton" style={{ height: 14, width: '40%', marginBottom: 18 }} />
              <div className="skeleton" style={{ height: 60, borderRadius: 10 }} />
              <div className="skeleton" style={{ height: 32, marginTop: 12, borderRadius: 10 }} />
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && results.length === 0 && scanned !== null && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', padding: '60px 0', gap: 14, opacity: 0.6
        }}>
          <Telescope size={36} style={{ opacity: 0.3 }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 16, fontWeight: 600 }}>No signals found</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
              Try lowering the minimum confidence or changing the engine
            </div>
          </div>
        </div>
      )}

      {/* Initial empty state */}
      {!loading && results.length === 0 && scanned === null && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', padding: '80px 0', gap: 16, opacity: 0.55
        }}>
          <div style={{
            width: 72, height: 72, borderRadius: 22,
            background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)'
          }}>
            <Telescope size={30} />
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 18, fontWeight: 700 }}>Ready to Scan</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 6, maxWidth: 340, lineHeight: 1.6 }}>
              Configure your filters above and click <strong>Scan Universe</strong> to find actionable signals across 141 NSE stocks.
            </div>
          </div>
          <button className="btn-primary" onClick={scan} style={{ marginTop: 4 }}>
            <Search size={14} /> Scan Now
          </button>
        </div>
      )}

      {/* Results */}
      {!loading && results.length > 0 && (
        <>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: 14
          }}>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{results.length}</span> signals
              {scanned && <span> from {scanned} stocks scanned</span>}
              {signal && <span> · {signal} only</span>}
              {sector && <span> · {sector} sector</span>}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--green)', fontWeight: 600 }}>
                {results.filter(r => r.signal === 'BUY').length} BUY
              </span>
              <span style={{ color: 'var(--border-strong)' }}>·</span>
              <span style={{ fontSize: 11, color: 'var(--red)', fontWeight: 600 }}>
                {results.filter(r => r.signal === 'SELL').length} SELL
              </span>
            </div>
          </div>

          <div className="scanner-grid">
            {results.map((sig, i) => (
              <SignalCard
                key={sig.symbol + i}
                sig={sig}
                isApproved={approved.has(sig.symbol)}
                onApprove={() => approveSignal(sig)}
                animDelay={i * 40}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── Signal Card ────────────────────────────────────────────────────
function SignalCard({ sig, isApproved, onApprove, animDelay }) {
  const isBuy = sig.signal === 'BUY';

  return (
    <div
      className={`scanner-card ${isBuy ? 'buy' : 'sell'} animate-fade-in`}
      style={{ animationDelay: `${animDelay}ms` }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <div>
          <div className="scanner-symbol">
            {sig.symbol?.replace('.NS', '').replace('.BO', '')}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 5, flexWrap: 'wrap' }}>
            <span className="tag">{sig.weekly_trend || 'NSE'}</span>
            {sig.sector && <span className="tag">{sig.sector}</span>}
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <span className={`badge badge-${sig.signal?.toLowerCase()}`}>
            {isBuy
              ? <><TrendingUp size={10} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />{sig.signal}</>
              : <><TrendingDown size={10} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />{sig.signal}</>
            }
          </span>
          <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6,
            color: isBuy ? 'var(--green)' : 'var(--red)',
            fontVariantNumeric: 'tabular-nums'
          }}>
            {sig.confidence?.toFixed(0)}%
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            confidence
          </div>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="confidence-bar">
        <div
          className={`confidence-fill ${isBuy ? 'buy' : 'sell'}`}
          style={{ width: `${sig.confidence || 0}%` }}
        />
      </div>

      {/* Levels */}
      <div className="scanner-levels">
        <div>
          <div className="scanner-level-label">Entry</div>
          <div className="scanner-level-value" style={{ color: 'var(--text-primary)' }}>
            ₹{sig.entry_price?.toFixed(1)}
          </div>
        </div>
        <div>
          <div className="scanner-level-label">Stop Loss</div>
          <div className="scanner-level-value" style={{ color: 'var(--red)' }}>
            ₹{sig.stop_loss?.toFixed(1)}
          </div>
        </div>
        <div>
          <div className="scanner-level-label">Target</div>
          <div className="scanner-level-value" style={{ color: 'var(--green)' }}>
            ₹{sig.target?.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Action */}
      {isApproved ? (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
          padding: '9px', borderRadius: 12,
          background: 'var(--green-dim)', color: 'var(--green)',
          fontSize: 12, fontWeight: 600
        }}>
          <CheckCircle size={13} /> Signal Approved
        </div>
      ) : (
        <button
          className="btn-primary"
          onClick={onApprove}
          style={{
            width: '100%', justifyContent: 'center', fontSize: 12,
            background: isBuy
              ? 'linear-gradient(135deg, #10b981, #059669)'
              : 'linear-gradient(135deg, #f43f5e, #e11d48)',
            boxShadow: isBuy
              ? '0 4px 14px rgba(16,185,129,0.3)'
              : '0 4px 14px rgba(244,63,94,0.3)',
          }}
        >
          Approve Signal
        </button>
      )}
    </div>
  );
}
