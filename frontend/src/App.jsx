import React, { useState, useEffect } from 'react';
import { 
  BarChart3, Brain, Zap, Briefcase, History, Settings, 
  Menu, X, Bell, User, LogOut, Activity, LayoutDashboard, DollarSign
} from 'lucide-react';
import AIPredictionsView from './AIPredictionsView';
import AutoModeView from './AutoModeView';
import PortfolioView from './PortfolioView';
import HistoryView from './HistoryView';
import AnalysisView from './AnalysisView';
import ActiveTradesView from './ActiveTradesView';
import ManualTradeView from './ManualTradeView';
import SettingsView from './SettingsView';
import LoginView from './LoginView';
import RegisterView from './RegisterView';

export default function App() {
  // Bypassing login for now as requested
  const [isLoggedIn, setIsLoggedIn] = useState(true); 
  const [authView, setAuthView] = useState('login'); 
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [positions, setPositions] = useState([]);
  const [history, setHistory] = useState([]);
  const [queue, setQueue] = useState([]);

  useEffect(() => {
    const fetchPositions = async () => {
      if (!isLoggedIn) return;
      try {
        const token = localStorage.getItem('token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        const res = await fetch('http://localhost:8000/api/positions', { headers });
        
        if (res.status === 401 && token) {
          // handleLogout(); // Disabled for bypass mode
          return;
        }
        const data = await res.json();
        setPositions(data.positions || []);
      } catch (e) {}
    };

    const fetchHistoryAndQueue = async () => {
      if (!isLoggedIn) return;
      try {
        const hRes = await fetch('http://localhost:8000/api/history');
        if (hRes.ok) {
          const hData = await hRes.json();
          setHistory(hData.history || []);
        }

        const qRes = await fetch('http://localhost:8000/api/signals');
        if (qRes.ok) {
          const qData = await qRes.json();
          setQueue(qData.queue || []);
        }
      } catch (e) {
        console.error("Failed to fetch history or queue:", e);
      }
    };

    fetchPositions();
    fetchHistoryAndQueue();
    const posInterval = setInterval(fetchPositions, 5000);
    const hostInterval = setInterval(fetchHistoryAndQueue, 10000);
    return () => {
      clearInterval(posInterval);
      clearInterval(hostInterval);
    };
  }, [isLoggedIn]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsLoggedIn(false);
    setAuthView('login');
  };

  const menuItems = [
    { id: 'Dashboard', name: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { id: 'Deep Analysis', name: 'Deep Analysis', icon: <BarChart3 size={20} /> },
    { id: 'AI Predictions', name: 'AI Predictions', icon: <Brain size={20} /> },
    { id: 'Auto Mode', name: 'Auto Mode', icon: <Zap size={20} /> },
    { id: 'Active Trades', name: 'Active Trades', icon: <Activity size={20} />, badge: positions.length > 0 ? positions.length : null },
    { id: 'Manual Order', name: 'Manual Order', icon: <DollarSign size={20} /> },
    { id: 'My Portfolio', name: 'My Portfolio', icon: <Briefcase size={20} /> },
    { id: 'Trade History', name: 'Trade History', icon: <History size={20} /> },
    { id: 'Settings', name: 'Settings', icon: <Settings size={20} /> },
  ];

  /* Authenticated View Logic - Disabled for Bypass
  if (!isLoggedIn) {
    return authView === 'login' ? (
      <LoginView onLogin={() => setIsLoggedIn(true)} onSwitchToRegister={() => setAuthView('register')} />
    ) : (
      <RegisterView onRegister={() => setAuthView('login')} onSwitchToLogin={() => setAuthView('login')} />
    );
  }
  */

  const renderContent = () => {
    switch (activeTab) {
      case 'Deep Analysis': return <AnalysisView />;
      case 'AI Predictions': return <AIPredictionsView queue={queue} setQueue={setQueue} />;
      case 'Auto Mode': return <AutoModeView queue={queue} setQueue={setQueue} history={history} setHistory={setHistory} />;
      case 'Active Trades': return <ActiveTradesView positions={positions} setPositions={setPositions} />;
      case 'Manual Order': return <ManualTradeView />;
      case 'My Portfolio': return <PortfolioView />;
      case 'Trade History': return <HistoryView history={history} />;
      case 'Settings': return <SettingsView />;
      default: return (
        <div className="animate-fade-in dashboard-home">
          <div className="topbar">
            <div>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Market Overview</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Real-time performance and AI signal monitoring.</p>
            </div>
            <div className="glass-panel" style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="status-indicator"></span>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>MARKET OPEN</span>
            </div>
          </div>
          
          <div className="metrics-grid">
            <div className="glass-panel metric-card">
              <span className="metric-title">Portfolio Value</span>
              <span className="metric-value">₹12,45,670.00</span>
              <span className="metric-change positive">+2.4% today</span>
            </div>
            <div className="glass-panel metric-card">
              <span className="metric-title">Live P&L</span>
              <span className={`metric-value ${positions.reduce((s,p)=>s+p.pnl,0) >= 0 ? 'positive' : 'negative'}`}>
                ₹{positions.reduce((s,p)=>s+p.pnl,0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              </span>
              <span className="metric-change">Unrealized profit</span>
            </div>
            <div className="glass-panel metric-card">
              <span className="metric-title">Open Positions</span>
              <span className="metric-value">{positions.length} Trades</span>
              <span className="metric-change">System Monitoring</span>
            </div>
          </div>

          <div className="dashboard-grid" style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '24px', marginTop: '24px' }}>
            {/* Active Trades Snapshot */}
            <div className="glass-panel" style={{ overflow: 'hidden' }}>
              <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--panel-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>Active Trades Snapshot</h3>
                <button className="text-btn" onClick={() => setActiveTab('Active Trades')}>View All</button>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table className="mini-table">
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Qty</th>
                      <th>LTP</th>
                      <th style={{ textAlign: 'right' }}>P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.length === 0 ? (
                      <tr><td colSpan="4" style={{ textAlign: 'center', padding: '32px', color: 'var(--text-secondary)' }}>No active trades</td></tr>
                    ) : (
                      positions.slice(0, 5).map(p => (
                        <tr key={p.id}>
                          <td><strong>{p.asset}</strong></td>
                          <td>{p.quantity}</td>
                          <td>₹{p.current_price.toFixed(1)}</td>
                          <td style={{ textAlign: 'right', color: p.pnl >= 0 ? 'var(--color-success)' : 'var(--color-danger)', fontWeight: 700 }}>
                            {p.pnl >= 0 ? '+' : ''}₹{p.pnl.toFixed(0)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* AI Signal Feed */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--panel-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>Recommended AI Signals</h3>
                <button className="text-btn" onClick={() => setActiveTab('AI Predictions')}>Explore</button>
              </div>
              <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {queue.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-secondary)' }}>No fresh signals available</div>
                ) : (
                  queue.slice(0, 3).map(q => (
                    <div key={q.id} className="signal-mini-card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{q.asset}</div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{q.type} • {q.age}</div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span style={{ 
                            fontSize: '0.75rem', fontWeight: 800, 
                            color: q.signal === 'BUY' ? 'var(--color-success)' : 'var(--color-danger)',
                            background: q.signal === 'BUY' ? 'hsla(142, 70%, 50%, 0.1)' : 'hsla(348, 80%, 55%, 0.1)',
                            padding: '2px 8px', borderRadius: '4px'
                          }}>{q.signal}</span>
                          <div style={{ fontSize: '0.7rem', marginTop: 4 }}>Confidence: {q.confidence}%</div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
                <div style={{ marginTop: 'auto', padding: '12px', borderTop: '1px solid var(--panel-border)', textAlign: 'center' }}>
                  <button className="btn-primary" style={{ width: '100%' }} onClick={() => setActiveTab('AI Predictions')}>
                    <Brain size={16} /> View All Predictions
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="app-container">
      <aside className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">
              <Activity size={20} strokeWidth={3} />
            </div>
            <div className="logo-text-wrapper">
              <span className="logo-text">ASTRA</span>
              <span className="logo-subtitle">Automated Stock Trading & Research Assistant</span>
            </div>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              <span className="nav-text">{item.name}</span>
              {item.badge && <span className="nav-badge">{item.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="nav-item" onClick={handleLogout} style={{ color: '#ef4444', border: 'none', background: 'none', width: '100%', cursor: 'pointer' }}>
            <LogOut size={20} />
            <span className="nav-text">Sign Out</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="main-header">
          <button className="icon-button" onClick={() => setSidebarOpen(!isSidebarOpen)}>
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          
          <div style={{ flex: 1 }} />
          
          <div className="header-actions">
            <button className="icon-button"><Bell size={20} /></button>
            <div className="user-profile">
              <div className="user-avatar"><User size={20} /></div>
              <div className="user-info">
                <span className="user-name">Alpha Trader</span>
                <span className="user-role">Live Account</span>
              </div>
            </div>
          </div>
        </header>

        <div className="view-container">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}
