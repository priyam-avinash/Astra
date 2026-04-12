import React, { useState, useEffect, useCallback } from 'react';
import {
  TrendingUp, TrendingDown, RefreshCw, Globe, IndianRupee,
  Activity, AlertTriangle, ChevronRight, Bitcoin, Zap
} from 'lucide-react';

/* ── STYLES ── */
const S = {
  page: {
    height: '100%', width: '100%', display: 'flex', flexDirection: 'column',
    paddingTop: 64, paddingLeft: 24, paddingRight: 24, paddingBottom: 32,
    background: '#0f1118', color: '#e1e4ea',
    fontFamily: "'Inter', -apple-system, sans-serif",
    overflowY: 'auto', boxSizing: 'border-box',
  },
  inner: { maxWidth: 1200, width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 },
  toggle: { display: 'flex', background: '#1e222d', borderRadius: 10, padding: 4, gap: 4 },
  toggleBtn: (active) => ({
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '8px 20px', borderRadius: 8, border: 'none', cursor: 'pointer',
    fontWeight: 600, fontSize: 13, fontFamily: "'Inter', sans-serif",
    background: active ? '#2962ff' : 'transparent',
    color: active ? '#fff' : '#787b86', transition: 'all .2s',
  }),
  card: { borderRadius: 12, border: '1px solid #2a2e39', background: '#1e222d', padding: 20 },
  tickerStrip: {
    display: 'flex', gap: 12, overflowX: 'auto',
    paddingBottom: 4,
    scrollbarWidth: 'none',
  },
  tickerCard: (change) => ({
    flexShrink: 0, minWidth: 160, padding: '14px 18px',
    borderRadius: 12, border: `1px solid ${change >= 0 ? '#26a69a33' : '#ef535033'}`,
    background: change >= 0 ? 'rgba(38,166,154,.06)' : 'rgba(239,83,80,.06)',
    cursor: 'pointer', transition: 'all .2s',
  }),
  signalBadge: (s) => ({
    display: 'inline-flex', padding: '4px 12px', borderRadius: 5,
    fontWeight: 700, fontSize: 12,
    background: s === 'BUY' ? 'rgba(38,166,154,.15)' : s === 'SELL' ? 'rgba(239,83,80,.15)' : 'rgba(120,123,134,.15)',
    color: s === 'BUY' ? '#26a69a' : s === 'SELL' ? '#ef5350' : '#787b86',
    border: `1px solid ${s === 'BUY' ? '#26a69a55' : s === 'SELL' ? '#ef535055' : '#787b8655'}`,
  }),
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
};

/* ── FEAR & GREED GAUGE ── */
function FearGreedGauge({ value, label }) {
  const safe = isNaN(value) ? 50 : Math.min(Math.max(value, 0), 100);
  const angle = -90 + (safe / 100) * 180;
  let color;
  if (safe <= 25) color = '#ef5350';
  else if (safe <= 45) color = '#f77c80';
  else if (safe >= 75) color = '#26a69a';
  else if (safe >= 55) color = '#4dd0c8';
  else color = '#ffc107';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <div style={{ position: 'relative', width: 160, height: 80 }}>
        <svg viewBox="0 0 200 100" style={{ width: '100%', height: '100%' }}>
          <path d="M 20 95 A 80 80 0 0 1 180 95" fill="none" stroke="#2a2e39" strokeWidth="10" strokeLinecap="round"/>
          <path d="M 20 95 A 80 80 0 0 1 56 28" fill="none" stroke="#ef5350" strokeWidth="10" strokeLinecap="round" opacity=".8"/>
          <path d="M 56 28 A 80 80 0 0 1 100 15" fill="none" stroke="#ffc107" strokeWidth="10" strokeLinecap="round" opacity=".8"/>
          <path d="M 100 15 A 80 80 0 0 1 144 28" fill="none" stroke="#ffc107" strokeWidth="10" strokeLinecap="round" opacity=".5"/>
          <path d="M 144 28 A 80 80 0 0 1 180 95" fill="none" stroke="#26a69a" strokeWidth="10" strokeLinecap="round" opacity=".8"/>
          <g transform={`rotate(${angle} 100 95)`}>
            <line x1="100" y1="95" x2="100" y2="25" stroke={color} strokeWidth="2.5" strokeLinecap="round"/>
            <circle cx="100" cy="95" r="5" fill={color}/>
            <circle cx="100" cy="95" r="2" fill="#131722"/>
          </g>
        </svg>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color }}>{safe}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color, textTransform: 'uppercase', letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 11, color: '#787b86', marginTop: 2 }}>Fear & Greed Index</div>
    </div>
  );
}

/* ── SIGNAL CARD ── */
function SignalCard({ asset, onAnalyze }) {
  const isPositive = asset.change_24h >= 0;
  const fmt = (v) => v == null ? '—' : Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  const currency = asset.currency === 'INR' ? '₹' : '$';

  return (
    <div
      style={{ ...S.card, cursor: 'pointer', transition: 'border-color .2s, transform .15s' }}
      onClick={() => onAnalyze(asset.symbol)}
      onMouseEnter={e => { e.currentTarget.style.borderColor = '#2962ff55'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = '#2a2e39'; e.currentTarget.style.transform = 'none'; }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: '#fff' }}>{asset.name || asset.symbol}</div>
          <div style={{ fontSize: 11, color: '#787b86', marginTop: 2 }}>
            {asset.symbol} • <span style={{ textTransform: 'capitalize', color: asset.risk_class === 'major' ? '#2962ff' : asset.risk_class === 'meme' ? '#ffc107' : '#787b86' }}>{asset.risk_class}</span>
          </div>
        </div>
        <span style={S.signalBadge(asset.signal || 'HOLD')}>{asset.signal || 'HOLD'}</span>
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#fff', letterSpacing: -0.5 }}>
        {currency}{fmt(asset.price)}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 4, fontSize: 13, fontWeight: 600, color: isPositive ? '#26a69a' : '#ef5350' }}>
        {isPositive ? <TrendingUp size={14}/> : <TrendingDown size={14}/>}
        {isPositive ? '+' : ''}{asset.change_24h?.toFixed(2)}%
        <span style={{ color: '#787b86', fontWeight: 400, marginLeft: 4 }}>24h</span>
      </div>
      {asset.confidence != null && (
        <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#787b86' }}>
          <span>AI Confidence</span>
          <span style={{ color: '#fff', fontWeight: 600 }}>{asset.confidence?.toFixed(1)}%</span>
        </div>
      )}
      <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#787b86' }}>
        <ChevronRight size={12}/> Click to Deep Analyze
      </div>
    </div>
  );
}

/* ── MAIN COMPONENT ── */
export default function CryptoView({ onNavigateToAnalysis }) {
  const [market, setMarket] = useState('international');
  const [watchlist, setWatchlist] = useState([]);
  const [fearGreed, setFearGreed] = useState({ value: 50, label: 'Neutral' });
  const [btcDom, setBtcDom] = useState(50);
  const [regime, setRegime] = useState({ text: 'Loading…', color: '#787b86' });
  const [loading, setLoading] = useState(false);
  const [signalMap, setSignalMap] = useState({});

  const API = 'http://localhost:8000';
  const headers = () => {
    const t = localStorage.getItem('token');
    return t ? { Authorization: `Bearer ${t}` } : {};
  };

  const fetchAll = useCallback(async (mkt) => {
    setLoading(true);
    try {
      // Fetch watchlist prices
      const wRes = await fetch(`${API}/api/crypto/watchlist?market=${mkt}`, { headers: headers() });
      if (wRes.ok) {
        const wData = await wRes.json();
        setWatchlist(wData.assets || []);
      }

      // Fetch Fear & Greed (only for international)
      if (mkt === 'international') {
        const fgRes = await fetch(`${API}/api/crypto/fear-greed`, { headers: headers() });
        if (fgRes.ok) {
          const fgData = await fgRes.json();
          setFearGreed(fgData.fear_greed || { value: 50, label: 'Neutral' });
          setBtcDom(fgData.btc_dominance || 50);
          // Determine market regime from F&G
          const v = fgData.fear_greed?.value || 50;
          if (v < 25) setRegime({ text: 'Extreme Fear — Possible Reversal Zone', color: '#ef5350' });
          else if (v < 45) setRegime({ text: 'Fear — Cautious Market', color: '#f77c80' });
          else if (v > 75) setRegime({ text: 'Extreme Greed — Reversal Risk', color: '#ff9800' });
          else if (v > 55) setRegime({ text: 'Greed — Momentum Favourable', color: '#26a69a' });
          else setRegime({ text: 'Neutral Market Conditions', color: '#787b86' });
        }
      } else {
        setFearGreed({ value: 50, label: 'Neutral' });
        setRegime({ text: 'Indian Market — INR Pairs Active', color: '#2962ff' });
      }
    } catch (e) {
      console.error('CryptoView fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch quick signals for all watchlist assets
  const fetchQuickSignals = useCallback(async (assets, mkt) => {
    const promises = assets.slice(0, 5).map(async (a) => {
      try {
        const res = await fetch(
          `${API}/api/crypto/analyze/${a.symbol}?market=${mkt}&engine=astra_crypto`,
          { headers: headers() }
        );
        if (res.ok) {
          const d = await res.json();
          return [a.symbol, { signal: d.signal, confidence: d.confidence, regime: d.market_regime }];
        }
      } catch { }
      return [a.symbol, { signal: 'HOLD', confidence: 50 }];
    });
    const results = await Promise.all(promises);
    setSignalMap(Object.fromEntries(results));
  }, []);

  useEffect(() => {
    fetchAll(market);
  }, [market, fetchAll]);

  useEffect(() => {
    if (watchlist.length > 0) {
      fetchQuickSignals(watchlist, market);
    }
  }, [watchlist, market, fetchQuickSignals]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(() => fetchAll(market), 60000);
    return () => clearInterval(interval);
  }, [market, fetchAll]);

  const handleAnalyze = (symbol) => {
    if (onNavigateToAnalysis) {
      onNavigateToAnalysis(symbol, market);
    }
  };

  // Enrich watchlist with signal data
  const enriched = watchlist.map(a => ({
    ...a,
    ...(signalMap[a.symbol] || {}),
  }));

  const currency = market === 'indian' ? '₹' : '$';

  return (
    <div style={S.page}>
      <div style={S.inner}>

        {/* ── HEADER ── */}
        <div style={S.header}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 10 }}>
              <Bitcoin size={26} color="#f7931a"/> ASTRA.CRYPTO
            </h1>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: '#787b86' }}>
              Live cryptocurrency signals — International & Indian markets
            </p>
          </div>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            {/* Market Toggle */}
            <div style={S.toggle}>
              <button style={S.toggleBtn(market === 'international')} onClick={() => setMarket('international')}>
                <Globe size={14}/> International (USD)
              </button>
              <button style={S.toggleBtn(market === 'indian')} onClick={() => setMarket('indian')}>
                <IndianRupee size={14}/> Indian (INR)
              </button>
            </div>
            <button
              onClick={() => fetchAll(market)}
              style={{ padding: '8px 14px', borderRadius: 8, border: '1px solid #2a2e39', background: '#1e222d', color: '#787b86', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}
            >
              <RefreshCw size={14} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }}/> Refresh
            </button>
          </div>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>

        {/* ── TICKER STRIP ── */}
        <div>
          <div style={{ fontSize: 12, color: '#787b86', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Live Prices
          </div>
          <div style={S.tickerStrip}>
            {loading && watchlist.length === 0
              ? Array(5).fill(0).map((_, i) => (
                  <div key={i} style={{ ...S.tickerCard(0), width: 160, opacity: 0.4 }}>
                    <div style={{ height: 12, background: '#2a2e39', borderRadius: 4, marginBottom: 8 }}/>
                    <div style={{ height: 20, background: '#2a2e39', borderRadius: 4 }}/>
                  </div>
                ))
              : enriched.map(a => (
                  <div
                    key={a.symbol}
                    style={S.tickerCard(a.change_24h)}
                    onClick={() => handleAnalyze(a.symbol)}
                  >
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#fff', marginBottom: 4 }}>{a.symbol}</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>
                      {currency}{Number(a.price || 0).toLocaleString('en-US', { maximumFractionDigits: 2 })}
                    </div>
                    <div style={{ fontSize: 12, color: a.change_24h >= 0 ? '#26a69a' : '#ef5350', marginTop: 2, fontWeight: 600 }}>
                      {a.change_24h >= 0 ? '▲' : '▼'} {Math.abs(a.change_24h || 0).toFixed(2)}%
                    </div>
                    <span style={{ ...S.signalBadge(a.signal || 'HOLD'), fontSize: 10, padding: '2px 8px', marginTop: 6, display: 'inline-flex' }}>
                      {a.signal || '…'}
                    </span>
                  </div>
                ))
            }
          </div>
        </div>

        {/* ── MACRO PANEL ── */}
        <div style={S.grid2}>
          {/* Fear & Greed */}
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
            <div style={{ alignSelf: 'flex-start' }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#fff' }}>Market Sentiment</div>
              <div style={{ fontSize: 12, color: '#787b86' }}>Alternative.me Fear & Greed Index</div>
            </div>
            <FearGreedGauge value={fearGreed.value} label={fearGreed.label}/>
          </div>

          {/* BTC Dominance + Regime */}
          <div style={S.card}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fff', marginBottom: 16 }}>Market Context</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {market === 'international' && (
                <div>
                  <div style={{ fontSize: 12, color: '#787b86', marginBottom: 4 }}>BTC Dominance</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ flex: 1, height: 6, background: '#2a2e39', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${btcDom}%`, height: '100%', background: '#f7931a', borderRadius: 3, transition: 'width .5s' }}/>
                    </div>
                    <span style={{ fontSize: 14, fontWeight: 700, color: '#f7931a', minWidth: 48 }}>{btcDom.toFixed(1)}%</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#787b86', marginTop: 4 }}>
                    {btcDom > 58 ? '⚠️ High dominance — altcoin BUYs suppressed' : '✅ Altcoin season conditions favourable'}
                  </div>
                </div>
              )}
              <div style={{ padding: '12px 16px', borderRadius: 8, background: '#131722', border: `1px solid ${regime.color}33` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={14} color={regime.color}/>
                  <span style={{ fontSize: 13, fontWeight: 600, color: regime.color }}>{regime.text}</span>
                </div>
              </div>
              {fearGreed.value < 25 && (
                <div style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(239,83,80,.1)', border: '1px solid #ef535033', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                  <AlertTriangle size={14} color="#ef5350" style={{ marginTop: 1, flexShrink: 0 }}/>
                  <span style={{ fontSize: 12, color: '#f77c80' }}>
                    Extreme Fear detected. ASTRA.CRYPTO will block new BUY signals until sentiment recovers above 30.
                  </span>
                </div>
              )}
              {fearGreed.value > 85 && (
                <div style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(255,152,0,.1)', border: '1px solid #ff980033', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                  <AlertTriangle size={14} color="#ff9800" style={{ marginTop: 1, flexShrink: 0 }}/>
                  <span style={{ fontSize: 12, color: '#ffb74d' }}>
                    Extreme Greed detected. High reversal risk — ASTRA.CRYPTO applying stricter entry filters.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── SIGNAL CARDS GRID ── */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>
                ASTRA.CRYPTO Signals
              </div>
              <div style={{ fontSize: 12, color: '#787b86', marginTop: 2 }}>
                {market === 'international' ? 'International USD pairs' : 'Indian INR pairs'} — Click any card to run deep analysis
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <span style={{ ...S.signalBadge('BUY'), fontSize: 11 }}>BUY</span>
              <span style={{ ...S.signalBadge('SELL'), fontSize: 11 }}>SELL</span>
              <span style={{ ...S.signalBadge('HOLD'), fontSize: 11 }}>HOLD</span>
            </div>
          </div>
          <div style={S.grid3}>
            {enriched.length === 0 && loading
              ? Array(6).fill(0).map((_, i) => (
                  <div key={i} style={{ ...S.card, opacity: 0.4, minHeight: 140 }}>
                    <div style={{ height: 14, background: '#2a2e39', borderRadius: 4, marginBottom: 12 }}/>
                    <div style={{ height: 24, background: '#2a2e39', borderRadius: 4, marginBottom: 8 }}/>
                    <div style={{ height: 12, background: '#2a2e39', borderRadius: 4, width: '60%' }}/>
                  </div>
                ))
              : enriched.map(a => (
                  <SignalCard key={a.symbol} asset={a} onAnalyze={handleAnalyze}/>
                ))
            }
          </div>
        </div>

        {/* ── ENGINE INFO ── */}
        <div style={{ ...S.card, background: 'rgba(41,98,255,.06)', borderColor: '#2962ff33' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
            <Zap size={20} color="#2962ff" style={{ marginTop: 2, flexShrink: 0 }}/>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#2962ff', marginBottom: 6 }}>
                ASTRA.CRYPTO Engine — How It Works
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 32px', fontSize: 12, color: '#a0a4ab', lineHeight: 1.7 }}>
                <span>📈 <strong style={{ color: '#e1e4ea' }}>Trending market (ADX &gt;25):</strong> Donchian Channel breakout + Volume surge</span>
                <span>📉 <strong style={{ color: '#e1e4ea' }}>Ranging market (ADX &lt;20):</strong> Bollinger Band mean reversion</span>
                <span>🛡️ <strong style={{ color: '#e1e4ea' }}>Macro filters:</strong> Fear &amp; Greed + BTC Dominance + Risk class</span>
                <span>⚡ <strong style={{ color: '#e1e4ea' }}>ATR-adaptive SL/TP:</strong> Major 2×/4.5× · Altcoin 2.8×/5.5× · Meme 3.5×/7×</span>
                <span>🌏 <strong style={{ color: '#e1e4ea' }}>Data source:</strong> CCXT Binance (intl) · yfinance .INR (India)</span>
                <span>🤖 <strong style={{ color: '#e1e4ea' }}>ML mode:</strong> Bidirectional LSTM trained on BTC+ETH+BNB+SOL</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
