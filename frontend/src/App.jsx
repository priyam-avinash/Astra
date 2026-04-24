import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart3, Brain, Zap, Briefcase, History, Settings,
  Menu, X, Bell, User, LogOut, Activity, LayoutDashboard,
  DollarSign, Bitcoin, Telescope, TrendingUp, TrendingDown,
  Package, ChevronRight, Search, RefreshCw, AlertCircle,
  ArrowUpRight, ArrowDownRight, Minus
} from 'lucide-react';
import AIPredictionsView  from './AIPredictionsView';
import AutoModeView       from './AutoModeView';
import PortfolioView      from './PortfolioView';
import HistoryView        from './HistoryView';
import AnalysisView       from './AnalysisView';
import ActiveTradesView   from './ActiveTradesView';
import ManualTradeView    from './ManualTradeView';
import SettingsView       from './SettingsView';
import ScannerView        from './ScannerView';
import CryptoView         from './CryptoView';

// ── Helpers ────────────────────────────────────────────────────────
function isMarketOpen() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  const day = ist.getDay();
  const h = ist.getHours(), m = ist.getMinutes();
  const mins = h * 60 + m;
  return day >= 1 && day <= 5 && mins >= 555 && mins <= 930; // 09:15–15:30
}

const MARKET_INDICES = [
  { id: 'NIFTY',    name: 'NIFTY 50',   symbol: '^NSEI',     price: '24,537',  chg: '+0.82%', up: true  },
  { id: 'BANKNIFTY',name: 'BANK NIFTY', symbol: '^NSEBANK',  price: '52,314',  chg: '+1.14%', up: true  },
  { id: 'MIDCAP',   name: 'MIDCAP',     symbol: '^CNXMDCP',  price: '11,208',  chg: '-0.31%', up: false },
  { id: 'IT',       name: 'IT INDEX',   symbol: '^CNXIT',    price: '38,921',  chg: '+0.56%', up: true  },
  { id: 'PHARMA',   name: 'PHARMA',     symbol: '^CNXPHARMA',price: '21,745',  chg: '-0.18%', up: false },
];

const NAV_GROUPS = [
  {
    label: 'Overview',
    items: [
      { id: 'Dashboard',    name: 'Dashboard',     icon: <LayoutDashboard size={16} /> },
    ]
  },
  {
    label: 'Markets',
    items: [
      { id: 'Deep Analysis',name: 'Deep Analysis', icon: <BarChart3      size={16} /> },
      { id: 'Crypto',       name: 'Crypto',        icon: <Bitcoin        size={16} /> },
      { id: 'Scanner',      name: 'Universe Scan', icon: <Telescope      size={16} /> },
      { id: 'Commodities',  name: 'Commodities',   icon: <Package        size={16} /> },
    ]
  },
  {
    label: 'AI Engine',
    items: [
      { id: 'AI Predictions',name: 'AI Predictions',icon: <Brain         size={16} /> },
      { id: 'Auto Mode',    name: 'Auto Mode',     icon: <Zap            size={16} /> },
    ]
  },
  {
    label: 'Trading',
    items: [
      { id: 'Active Trades',name: 'Active Trades', icon: <Activity       size={16} />, badge: true },
      { id: 'Manual Order', name: 'Manual Order',  icon: <DollarSign     size={16} /> },
      { id: 'My Portfolio', name: 'My Portfolio',  icon: <Briefcase      size={16} /> },
      { id: 'Trade History',name: 'Trade History', icon: <History        size={16} /> },
    ]
  },
  {
    label: 'System',
    items: [
      { id: 'Settings',     name: 'Settings',      icon: <Settings       size={16} /> },
    ]
  },
];

// ── Main App ──────────────────────────────────────────────────────
export default function App() {
  const [isLoggedIn, setIsLoggedIn]     = useState(true);
  const [activeTab, setActiveTab]       = useState('Dashboard');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [positions, setPositions]       = useState([]);
  const [history, setHistory]           = useState([]);
  const [queue, setQueue]               = useState([]);
  const [marketOpen, setMarketOpen]     = useState(isMarketOpen());

  // Crypto deep-link state
  const [cryptoSymbol, setCryptoSymbol] = useState(null);
  const [cryptoMarket, setCryptoMarket] = useState('international');

  // Quick-scan modal
  const [scanModal, setScanModal]       = useState(false);
  const [scanResults, setScanResults]   = useState([]);
  const [scanLoading, setScanLoading]   = useState(false);

  // Kill switch state
  const [halted, setHalted]             = useState(false);

  // Market open timer
  useEffect(() => {
    const t = setInterval(() => setMarketOpen(isMarketOpen()), 60_000);
    return () => clearInterval(t);
  }, []);

  // Fetch initial halt status on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/emergency/status')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setHalted(d.halted); })
      .catch(() => {});
  }, []);

  const toggleHalt = async () => {
    const endpoint = halted ? '/api/emergency/resume' : '/api/emergency/halt';
    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, { method: 'POST' });
      if (res.ok) { const d = await res.json(); setHalted(d.halted); }
    } catch (_) {}
  };

  // Positions / history polling
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/positions');
        if (res.ok) { const d = await res.json(); setPositions(d.positions || []); }
      } catch (_) {}
    };
    const fetchOther = async () => {
      try {
        const [hRes, qRes] = await Promise.all([
          fetch('http://localhost:8000/api/history'),
          fetch('http://localhost:8000/api/signals'),
        ]);
        if (hRes.ok) { const d = await hRes.json(); setHistory(d.history || []); }
        if (qRes.ok) { const d = await qRes.json(); setQueue(d.queue || []); }
      } catch (_) {}
    };
    fetchPositions(); fetchOther();
    const p = setInterval(fetchPositions, 5000);
    const o = setInterval(fetchOther, 10_000);
    return () => { clearInterval(p); clearInterval(o); };
  }, []);

  const handleLogout = () => { localStorage.removeItem('token'); setIsLoggedIn(false); };

  // Quick scan
  const runQuickScan = useCallback(async () => {
    setScanModal(true);
    setScanLoading(true);
    setScanResults([]);
    try {
      const res = await fetch('http://localhost:8000/api/scan/universe?engine=astra_ai&top_k=12&min_confidence=58');
      if (res.ok) { const d = await res.json(); setScanResults(d.signals || []); }
    } catch (_) { setScanResults([]); }
    setScanLoading(false);
  }, []);

  const livePnL = positions.reduce((s, p) => s + (p.pnl || 0), 0);

  // ── Navigation
  const goToAnalysis = (symbol, market) => {
    setCryptoSymbol(symbol); setCryptoMarket(market); setActiveTab('Deep Analysis');
  };

  // ── Content router
  const renderContent = () => {
    switch (activeTab) {
      case 'Deep Analysis':
        return <AnalysisView initialSymbol={cryptoSymbol} initialMarket={cryptoMarket} />;
      case 'Crypto':
        return <CryptoView onNavigateToAnalysis={(sym, mkt) => goToAnalysis(sym, mkt)} />;
      case 'Scanner':
        return <ScannerView />;
      case 'Commodities':
        return <ComingSoon title="Commodities" subtitle="Gold · Silver · Crude · Natural Gas · Copper" icon={<Package size={32} />} />;
      case 'AI Predictions':
        return <AIPredictionsView queue={queue} setQueue={setQueue} />;
      case 'Auto Mode':
        return <AutoModeView queue={queue} setQueue={setQueue} history={history} setHistory={setHistory} />;
      case 'Active Trades':
        return <ActiveTradesView positions={positions} setPositions={setPositions} />;
      case 'Manual Order':
        return <ManualTradeView />;
      case 'My Portfolio':
        return <PortfolioView />;
      case 'Trade History':
        return <HistoryView history={history} />;
      case 'Settings':
        return <SettingsView />;
      default:
        return <Dashboard
          positions={positions}
          queue={queue}
          livePnL={livePnL}
          onNavigate={setActiveTab}
          onIndexClick={goToAnalysis}
          onQuickScan={runQuickScan}
        />;
    }
  };

  return (
    <div className="app-container">

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">A</div>
            <div className="logo-text-wrapper">
              <span className="logo-text">ASTRA</span>
              <span className="logo-subtitle">AI Trading Platform</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_GROUPS.map(group => (
            <div key={group.label}>
              <div className="nav-section-label">{group.label}</div>
              {group.items.map(item => (
                <button
                  key={item.id}
                  className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(item.id)}
                >
                  {item.icon}
                  <span className="nav-text">{item.name}</span>
                  {item.badge && positions.length > 0 && (
                    <span className="nav-badge">{positions.length}</span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="nav-item" onClick={handleLogout}
            style={{ color: 'var(--red)', width: '100%', justifyContent: 'flex-start' }}>
            <LogOut size={16} />
            <span className="nav-text">Sign Out</span>
          </button>
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────────────── */}
      <main className="main-content">

        {/* Header */}
        <header className="main-header">
          <button className="icon-button" onClick={() => setSidebarOpen(v => !v)}>
            {isSidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <span className="header-breadcrumb" style={{ marginLeft: 4 }}>{activeTab}</span>

          <div className="header-actions">
            {/* Emergency Halt / Resume */}
            <button
              onClick={toggleHalt}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                background: 'transparent', cursor: 'pointer',
                border: `1px solid ${halted ? 'var(--green)' : 'var(--red)'}`,
                color: halted ? 'var(--green)' : 'var(--red)',
                borderRadius: 'var(--radius-md)',
                padding: '5px 11px', fontSize: 12, fontWeight: 600,
                transition: 'all 0.2s',
              }}
              title={halted ? 'Resume trading' : 'Emergency halt — stop all new orders'}
            >
              {halted ? '▶ Resume' : '⏹ Halt'}
            </button>

            {/* Quick Scan CTA */}
            <button className="btn-secondary" onClick={runQuickScan} style={{ fontSize: '12px', padding: '6px 14px' }}>
              <Search size={13} /> Quick Scan
            </button>

            {/* Market status */}
            <div className={`market-status-chip ${marketOpen ? 'open' : ''}`}>
              <span className={`status-dot ${marketOpen ? '' : 'closed'}`} />
              NSE {marketOpen ? 'OPEN' : 'CLOSED'}
            </div>

            {/* Notification */}
            <button className="icon-button" style={{ position: 'relative' }}>
              <Bell size={16} />
              {queue.length > 0 && (
                <span style={{
                  position: 'absolute', top: 4, right: 4,
                  width: 7, height: 7, borderRadius: '50%',
                  background: 'var(--red)', border: '1.5px solid var(--bg-surface)'
                }} />
              )}
            </button>

            {/* User */}
            <div className="user-profile">
              <div className="user-avatar">AT</div>
              <div className="user-info">
                <span className="user-name">Alpha Trader</span>
                <span className="user-role" style={{ color: 'var(--accent)' }}>Paper Trading</span>
              </div>
            </div>
          </div>
        </header>

        {/* Halt banner */}
        {halted && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '10px 24px', fontSize: 13, fontWeight: 500,
            background: 'var(--red-dim)', color: 'var(--red)',
            borderBottom: '1px solid rgba(244,63,94,0.2)',
          }}>
            ⛔ <strong>Trading halted</strong> — no new orders will be placed.
            <button onClick={toggleHalt} style={{
              marginLeft: 8, background: 'none', border: '1px solid var(--green)',
              color: 'var(--green)', borderRadius: 8, padding: '2px 10px',
              fontSize: 12, fontWeight: 600, cursor: 'pointer',
            }}>
              ▶ Resume
            </button>
          </div>
        )}

        {/* Content */}
        <div className="view-container">
          {renderContent()}
        </div>
      </main>

      {/* ── Quick Scan Modal ────────────────────────────────── */}
      {scanModal && (
        <QuickScanModal
          results={scanResults}
          loading={scanLoading}
          onClose={() => setScanModal(false)}
          onApprove={() => {}}
          onGoToScanner={() => { setScanModal(false); setActiveTab('Scanner'); }}
        />
      )}
    </div>
  );
}

// ── Dashboard View ─────────────────────────────────────────────────
function Dashboard({ positions, queue, livePnL, onNavigate, onIndexClick, onQuickScan }) {
  return (
    <div className="dashboard-home animate-fade-in">

      {/* Metrics */}
      <div className="metrics-grid">
        <MetricCard
          icon={<Briefcase size={16} />}
          title="Portfolio Value"
          value="₹12,45,670"
          change="+2.4% today"
          positive
        />
        <MetricCard
          icon={livePnL >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
          title="Live P&L"
          value={`${livePnL >= 0 ? '+' : ''}₹${Math.abs(livePnL).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
          change="Unrealized"
          positive={livePnL >= 0}
          negative={livePnL < 0}
        />
        <MetricCard
          icon={<Activity size={16} />}
          title="Open Positions"
          value={`${positions.length}`}
          change="Auto-monitored"
        />
        <MetricCard
          icon={<Brain size={16} />}
          title="Pending Signals"
          value={`${queue.length}`}
          change="Awaiting approval"
          onClick={() => onNavigate('AI Predictions')}
        />
      </div>

      {/* Market Pulse */}
      <div className="card" style={{ padding: '16px 20px', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Market Pulse
          </span>
          <button className="btn-ghost" onClick={onQuickScan} style={{ fontSize: 11 }}>
            <Search size={11} /> Quick Scan
          </button>
        </div>
        <div className="market-pulse-strip">
          {MARKET_INDICES.map(idx => (
            <div
              key={idx.id}
              className="pulse-chip"
              onClick={() => onIndexClick(idx.symbol, 'indian')}
            >
              <span className="pulse-chip-name">{idx.name}</span>
              <span className="pulse-chip-price">{idx.price}</span>
              <span className={`pulse-chip-change ${idx.up ? 'up' : 'down'}`}>
                {idx.up ? '▲' : '▼'} {idx.chg}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 16 }}>

        {/* Active Trades */}
        <div className="card" style={{ overflow: 'hidden' }}>
          <div className="panel-header">
            <span className="panel-title">Active Trades</span>
            <button className="text-btn" onClick={() => onNavigate('Active Trades')}>
              View all <ChevronRight size={12} style={{ verticalAlign: 'middle' }} />
            </button>
          </div>
          <table className="mini-table">
            <thead>
              <tr>
                <th>Asset</th><th>Direction</th><th>Entry</th>
                <th>LTP</th><th style={{ textAlign: 'right' }}>P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td colSpan="5" style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-tertiary)' }}>
                    <div style={{ marginBottom: 6 }}><Activity size={20} style={{ opacity: 0.3 }} /></div>
                    No active trades
                  </td>
                </tr>
              ) : (
                positions.slice(0, 6).map(p => (
                  <tr key={p.id}>
                    <td><strong>{p.asset}</strong></td>
                    <td>
                      <span className={`badge badge-${p.direction?.toLowerCase()}`}>
                        {p.direction}
                      </span>
                    </td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>₹{(p.entry_price || 0).toFixed(1)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>₹{(p.current_price || 0).toFixed(1)}</td>
                    <td style={{
                      textAlign: 'right', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                      color: (p.pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'
                    }}>
                      {(p.pnl || 0) >= 0 ? '+' : ''}₹{(p.pnl || 0).toFixed(0)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* AI Signal Feed */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="panel-header">
            <span className="panel-title">AI Signal Feed</span>
            <button className="text-btn" onClick={() => onNavigate('AI Predictions')}>
              Explore <ChevronRight size={12} style={{ verticalAlign: 'middle' }} />
            </button>
          </div>
          <div style={{ flex: 1, padding: '12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {queue.length === 0 ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '24px 0', color: 'var(--text-tertiary)' }}>
                <Brain size={24} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 12 }}>No signals pending</span>
              </div>
            ) : (
              queue.slice(0, 4).map(q => (
                <div key={q.id} className="signal-mini-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 14 }}>{q.asset}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>
                        {q.type} · {q.age}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span className={`badge badge-${q.signal?.toLowerCase()}`}>{q.signal}</span>
                      <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
                        {q.confidence}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="confidence-bar" style={{ marginTop: 8 }}>
                    <div
                      className={`confidence-fill ${q.signal === 'BUY' ? 'buy' : 'sell'}`}
                      style={{ width: `${q.confidence || 0}%` }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
          <div style={{ padding: '12px', borderTop: '1px solid var(--border-subtle)' }}>
            <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}
              onClick={() => onNavigate('AI Predictions')}>
              <Brain size={14} /> View All Predictions
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Metric Card ────────────────────────────────────────────────────
function MetricCard({ icon, title, value, change, positive, negative, onClick }) {
  return (
    <div
      className="card metric-card"
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      <div className="metric-icon" style={{
        background: positive ? 'var(--green-dim)' : negative ? 'var(--red-dim)' : 'var(--accent-dim)',
        color: positive ? 'var(--green)' : negative ? 'var(--red)' : 'var(--accent)',
      }}>
        {icon}
      </div>
      <div className="metric-title">{title}</div>
      <div className={`metric-value ${positive ? 'positive' : negative ? 'negative' : ''}`}>{value}</div>
      <div className={`metric-change ${positive ? 'positive' : negative ? 'negative' : ''}`}>{change}</div>
    </div>
  );
}

// ── Quick Scan Modal ───────────────────────────────────────────────
function QuickScanModal({ results, loading, onClose, onApprove, onGoToScanner }) {
  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ maxWidth: 620 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700 }}>Universe Quick Scan</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>
              ASTRA.AI · 141 NSE stocks · Top signals by confidence
            </div>
          </div>
          <button className="icon-button" onClick={onClose}><X size={18} /></button>
        </div>

        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[...Array(4)].map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 64, borderRadius: 14 }} />
            ))}
            <div style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-secondary)', marginTop: 8 }}>
              Scanning 141 stocks…
            </div>
          </div>
        ) : results.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-tertiary)' }}>
            <AlertCircle size={24} style={{ opacity: 0.4, marginBottom: 8, display: 'block', margin: '0 auto 8px' }} />
            <div>No high-confidence signals found right now.</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: '60vh', overflowY: 'auto' }}>
            {results.map((sig, i) => (
              <div key={i} className="card" style={{ padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontWeight: 700, fontSize: 15 }}>{sig.symbol?.replace('.NS','')}</span>
                    <span className={`badge badge-${sig.signal?.toLowerCase()}`}>{sig.signal}</span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    Entry ₹{sig.entry_price?.toFixed(1)} · SL ₹{sig.stop_loss?.toFixed(1)} · Target ₹{sig.target?.toFixed(1)}
                  </div>
                  <div className="confidence-bar" style={{ marginTop: 6, width: 140 }}>
                    <div className={`confidence-fill ${sig.signal === 'BUY' ? 'buy' : 'sell'}`}
                      style={{ width: `${sig.confidence || 0}%` }} />
                  </div>
                </div>
                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: sig.signal === 'BUY' ? 'var(--green)' : 'var(--red)' }}>
                    {sig.confidence?.toFixed(0)}%
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    confidence
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 10, marginTop: 18, justifyContent: 'flex-end' }}>
          <button className="btn-secondary" onClick={onClose}>Close</button>
          <button className="btn-primary" onClick={onGoToScanner}>
            <Telescope size={13} /> Open Full Scanner
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Coming Soon Placeholder ────────────────────────────────────────
function ComingSoon({ title, subtitle, icon }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: '60vh', gap: 16, opacity: 0.6
    }}>
      <div style={{
        width: 64, height: 64, borderRadius: 20,
        background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--accent)'
      }}>{icon}</div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 20, fontWeight: 700 }}>{title}</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>{subtitle}</div>
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 8 }}>
          In development — coming in the next sprint
        </div>
      </div>
    </div>
  );
}
