import React, { useState, useEffect, useCallback } from 'react';
import {
  Check, X, AlertCircle, Zap, Settings2, ChevronDown, ChevronUp,
  IndianRupee, Shield, TrendingUp, TrendingDown, Info, Lock
} from 'lucide-react';

// ── Persistence helpers ─────────────────────────────────────────────
const STORAGE_KEY = 'astra_automode_settings';

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (_) {}
  return {
    totalCapital:     100000,   // ₹1,00,000 default
    perTradePct:      10,       // 10% per trade
    maxConcurrent:    5,        // up to 5 open at once
    autoExecute:      false,
    riskMode:         'normal', // 'conservative' | 'normal' | 'aggressive'
  };
}

function saveSettings(settings) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); } catch (_) {}
}

// ── Risk mode presets ───────────────────────────────────────────────
const RISK_PRESETS = {
  conservative: { perTradePct: 5,  maxConcurrent: 3,  label: 'Conservative', color: 'var(--accent)',  desc: '5%/trade · max 3 open' },
  normal:       { perTradePct: 10, maxConcurrent: 5,  label: 'Balanced',     color: 'var(--yellow)',  desc: '10%/trade · max 5 open' },
  aggressive:   { perTradePct: 20, maxConcurrent: 8,  label: 'Aggressive',   color: 'var(--red)',     desc: '20%/trade · max 8 open' },
};

// ── Format helpers ──────────────────────────────────────────────────
const fmt = (n) => `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
const fmtK = (n) => n >= 100000 ? `₹${(n/100000).toFixed(1)}L` : n >= 1000 ? `₹${(n/1000).toFixed(0)}K` : `₹${n}`;

export default function AutoModeView({ queue, setQueue, history, setHistory }) {
  const [settings, setSettings]         = useState(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [localCapital, setLocalCapital] = useState('');
  const [executingId, setExecutingId]   = useState(null);
  const [toastMsg, setToastMsg]         = useState(null);

  // Sync localCapital string input with settings
  useEffect(() => {
    setLocalCapital(String(settings.totalCapital));
  }, [settings.totalCapital]);

  // Persist whenever settings change
  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const updateSettings = useCallback((patch) => {
    setSettings(prev => ({ ...prev, ...patch }));
  }, []);

  const applyRiskPreset = (mode) => {
    const preset = RISK_PRESETS[mode];
    updateSettings({ riskMode: mode, perTradePct: preset.perTradePct, maxConcurrent: preset.maxConcurrent });
  };

  // ── Capital math ─────────────────────────────────────────────────
  const perTradeAmount     = Math.floor(settings.totalCapital * settings.perTradePct / 100);
  const maxCommitted       = perTradeAmount * settings.maxConcurrent;
  const committedPct       = Math.min(100, (maxCommitted / settings.totalCapital) * 100);
  const availableCapital   = settings.totalCapital - maxCommitted;

  // Entry price — never fall back to targetPrice (they are different values)
  const entryPrice = (trade) => trade.entryPrice || trade.price || 0;

  // Calculate quantity for a given trade
  const calcQty = (trade) => {
    const price = entryPrice(trade);
    if (!price || price <= 0) return 0;
    return Math.floor(perTradeAmount / price);
  };

  const calcValue = (trade) => {
    const qty = calcQty(trade);
    return qty * entryPrice(trade);
  };

  // ── Toast ────────────────────────────────────────────────────────
  const showToast = (msg, type = 'success') => {
    setToastMsg({ msg, type });
    setTimeout(() => setToastMsg(null), 3000);
  };

  // ── Action handler ───────────────────────────────────────────────
  const handleAction = async (id, action) => {
    const trade = queue.find(q => q.id === id);
    setQueue(queue.filter(q => q.id !== id));

    if (action === 'approve' && trade) {
      const qty   = calcQty(trade);
      const price = trade.entryPrice || trade.targetPrice || trade.price || 0;

      if (qty < 1) {
        showToast(`Allocation too small to buy even 1 share of ${trade.asset} at ₹${price?.toFixed(0)}`, 'error');
        setQueue(prev => [trade, ...prev]); // put it back
        return;
      }

      setExecutingId(id);
      const ep = entryPrice(trade);

      // Optimistic history entry
      const optimisticEntry = {
        id:     `TRD-${Date.now()}`,
        asset:  `${trade.asset} (${trade.type || 'Equity'})`,
        action: trade.signal,
        price:  ep,
        qty,
        pnl:    0,
        time:   new Date().toLocaleString('en-IN'),
        status: 'Executed (AI Signal)',
      };
      setHistory(prev => [optimisticEntry, ...prev]);

      try {
        const res = await fetch('http://localhost:8000/api/execute', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id:           trade.id,
            asset:        trade.asset,
            action:       trade.signal,
            quantity:     qty,
            price:        ep,
            target_price: trade.targetPrice,
            stop_loss:    trade.stopLoss,
          }),
        });
        const result = await res.json();
        if (result.order_id) {
          setHistory(prev => prev.map(h =>
            h.id === optimisticEntry.id ? { ...h, id: result.order_id } : h
          ));
          showToast(`${trade.signal} ${qty} × ${trade.asset} executed — ${fmt(qty * price)}`);
        } else {
          showToast(result.detail || 'Execution failed', 'error');
        }
      } catch (err) {
        showToast('Network error — trade recorded locally', 'error');
      } finally {
        setExecutingId(null);
      }
    } else if (action === 'reject') {
      showToast(`${trade?.asset} signal rejected`);
    }
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: 900, position: 'relative' }}>

      {/* ── Toast ──────────────────────────────────────────────── */}
      {toastMsg && (
        <div style={{
          position: 'fixed', top: 24, right: 28, zIndex: 2000,
          background: toastMsg.type === 'error' ? 'var(--red-dim)' : 'var(--green-dim)',
          border: `1px solid ${toastMsg.type === 'error' ? 'rgba(244,63,94,0.3)' : 'rgba(16,216,122,0.3)'}`,
          color: toastMsg.type === 'error' ? 'var(--red)' : 'var(--green)',
          padding: '12px 18px', borderRadius: 'var(--radius-md)',
          fontSize: 13, fontWeight: 500,
          boxShadow: 'var(--shadow-lg)',
          animation: 'fadeUp 0.25s var(--ease-out)',
          maxWidth: 360,
        }}>
          {toastMsg.msg}
        </div>
      )}

      {/* ── Page header ─────────────────────────────────────────── */}
      <div className="topbar">
        <div>
          <h2 className="page-title">Auto Mode</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 4, fontSize: 13 }}>
            Human-in-the-loop approval for AI-generated trades.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            className="btn-secondary"
            onClick={() => setSettingsOpen(v => !v)}
            style={{ gap: 6 }}
          >
            <Settings2 size={14} />
            Capital Settings
            {settingsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
          <button
            className="btn-primary"
            onClick={() => updateSettings({ autoExecute: !settings.autoExecute })}
            style={{
              background: settings.autoExecute
                ? 'linear-gradient(135deg, #f43f5e, #e11d48)'
                : 'linear-gradient(135deg, #3b82f6, #6366f1)',
              boxShadow: settings.autoExecute
                ? '0 4px 14px rgba(244,63,94,0.3)'
                : '0 4px 14px rgba(59,130,246,0.3)',
            }}
          >
            <Zap size={14} />
            {settings.autoExecute ? 'Disable Auto-Execute' : 'Enable Auto-Execute'}
          </button>
        </div>
      </div>

      {/* ── Capital Settings Panel ───────────────────────────────── */}
      {settingsOpen && (
        <div className="card" style={{ padding: '24px', marginBottom: 20, animation: 'fadeUp 0.25s var(--ease-out)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
            <Shield size={15} style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: 13, fontWeight: 700 }}>Auto-Mode Capital Configuration</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, marginBottom: 24 }}>

            {/* Total Capital */}
            <div>
              <label style={labelStyle}>Total Auto-Mode Capital</label>
              <div style={{ position: 'relative' }}>
                <span style={{
                  position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--text-secondary)', fontSize: 14, fontWeight: 600, pointerEvents: 'none'
                }}>₹</span>
                <input
                  type="number"
                  className="input-field"
                  style={{ paddingLeft: 28, fontVariantNumeric: 'tabular-nums', fontWeight: 600, fontSize: 15 }}
                  value={localCapital}
                  min="1000"
                  step="1000"
                  onChange={e => setLocalCapital(e.target.value)}
                  onBlur={() => {
                    const v = Math.max(1000, parseInt(localCapital) || 100000);
                    setLocalCapital(String(v));
                    updateSettings({ totalCapital: v });
                  }}
                />
              </div>
              <p style={hintStyle}>Capital ring-fenced for auto-trading only</p>
            </div>

            {/* Per-trade % */}
            <div>
              <label style={labelStyle}>
                Per-Trade Allocation — <span style={{ color: 'var(--accent)' }}>{settings.perTradePct}%</span>
                &nbsp;·&nbsp;
                <span style={{ color: 'var(--green)', fontWeight: 700 }}>{fmtK(perTradeAmount)}</span>
              </label>
              <input
                type="range" min="1" max="50" step="1"
                value={settings.perTradePct}
                onChange={e => updateSettings({ perTradePct: +e.target.value, riskMode: 'custom' })}
                style={{ width: '100%', accentColor: 'var(--accent)', marginTop: 10, cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span style={hintStyle}>1%</span>
                <span style={hintStyle}>50%</span>
              </div>
            </div>

            {/* Max concurrent */}
            <div>
              <label style={labelStyle}>
                Max Concurrent Positions — <span style={{ color: 'var(--accent)' }}>{settings.maxConcurrent}</span>
              </label>
              <input
                type="range" min="1" max="20" step="1"
                value={settings.maxConcurrent}
                onChange={e => updateSettings({ maxConcurrent: +e.target.value, riskMode: 'custom' })}
                style={{ width: '100%', accentColor: 'var(--accent)', marginTop: 10, cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span style={hintStyle}>1</span>
                <span style={hintStyle}>20</span>
              </div>
            </div>
          </div>

          {/* Risk presets */}
          <div style={{ marginBottom: 24 }}>
            <label style={labelStyle}>Risk Preset</label>
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              {Object.entries(RISK_PRESETS).map(([key, preset]) => (
                <button
                  key={key}
                  onClick={() => applyRiskPreset(key)}
                  style={{
                    flex: 1,
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    border: `1px solid ${settings.riskMode === key ? preset.color : 'var(--border-default)'}`,
                    background: settings.riskMode === key
                      ? `color-mix(in srgb, ${preset.color} 12%, transparent)`
                      : 'var(--bg-elevated)',
                    color: settings.riskMode === key ? preset.color : 'var(--text-secondary)',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all var(--duration-fast)',
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 3 }}>{preset.label}</div>
                  <div style={{ fontSize: 11, opacity: 0.7 }}>{preset.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Capital utilisation bar */}
          <div style={{
            background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)',
            padding: '16px 18px', border: '1px solid var(--border-subtle)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
                Capital Utilisation (max exposure)
              </span>
              <span style={{ fontSize: 12, fontWeight: 700, color: committedPct > 80 ? 'var(--red)' : 'var(--text-primary)' }}>
                {committedPct.toFixed(0)}%
              </span>
            </div>
            <div style={{ height: 6, background: 'var(--bg-overlay)', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 99,
                width: `${committedPct}%`,
                background: committedPct > 80
                  ? 'linear-gradient(90deg, var(--yellow), var(--red))'
                  : 'linear-gradient(90deg, var(--accent), var(--green))',
                transition: 'width 0.4s var(--ease-out)',
              }} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 12 }}>
              <CapStat label="Total Pool"     value={fmt(settings.totalCapital)} />
              <CapStat label="Max Committed"  value={fmt(maxCommitted)}          color="var(--yellow)" />
              <CapStat label="Always Free"    value={fmt(Math.max(0, availableCapital))} color="var(--green)" />
            </div>
          </div>
        </div>
      )}

      {/* ── Auto-execute warning ─────────────────────────────────── */}
      {settings.autoExecute && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, padding: '13px 16px',
          background: 'var(--red-dim)', border: '1px solid rgba(244,63,94,0.25)',
          borderRadius: 'var(--radius-md)', marginBottom: 16,
          color: 'var(--red)', fontSize: 13,
        }}>
          <AlertCircle size={16} />
          <span>
            <strong>Auto-Execute is ON.</strong> Trades matching your criteria will execute immediately —
            {fmt(perTradeAmount)} per trade, up to {settings.maxConcurrent} concurrent positions.
          </span>
        </div>
      )}

      {/* ── Capital quick-stat strip (always visible) ────────────── */}
      {!settingsOpen && (
        <div style={{ display: 'flex', gap: 10, marginBottom: 18 }}>
          <QuickStat icon={<IndianRupee size={12} />} label="Auto Capital"    value={fmtK(settings.totalCapital)} />
          <QuickStat icon={<TrendingUp  size={12} />} label="Per Trade"       value={`${settings.perTradePct}% · ${fmtK(perTradeAmount)}`} />
          <QuickStat icon={<Shield      size={12} />} label="Max Positions"   value={String(settings.maxConcurrent)} />
          <QuickStat
            icon={<Zap size={12} />}
            label="Mode"
            value={settings.autoExecute ? 'Auto-Execute ON' : 'Manual Approval'}
            valueColor={settings.autoExecute ? 'var(--red)' : 'var(--green)'}
          />
        </div>
      )}

      {/* ── Trade queue ──────────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {queue.length === 0 ? (
          <div className="card" style={{
            padding: '56px 32px', textAlign: 'center',
            color: 'var(--text-tertiary)', display: 'flex',
            flexDirection: 'column', alignItems: 'center', gap: 12
          }}>
            <Zap size={28} style={{ opacity: 0.25 }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-secondary)' }}>
              Queue is empty
            </div>
            <div style={{ fontSize: 13 }}>
              No pending trades. The AI engines are scanning the market — new signals appear here.
            </div>
          </div>
        ) : (
          queue.map(trade => {
            const qty      = calcQty(trade);
            const value    = calcValue(trade);
            const price    = entryPrice(trade);   // entry price only — never targetPrice
            const isBuy    = trade.signal === 'BUY';
            const isExec   = executingId === trade.id;
            const tooSmall = qty < 1;

            return (
              <div
                key={trade.id}
                className="card"
                style={{
                  padding: '20px 22px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 16,
                  borderLeft: `3px solid ${isBuy ? 'var(--green)' : 'var(--red)'}`,
                  opacity: isExec ? 0.7 : 1,
                  transition: 'opacity 0.2s',
                }}
              >
                {/* Top row: signal badge + asset + confidence */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span className={`badge badge-${trade.signal?.toLowerCase()}`}>
                      {isBuy
                        ? <><TrendingUp size={10} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />{trade.signal}</>
                        : <><TrendingDown size={10} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />{trade.signal}</>
                      }
                    </span>
                    <span style={{ fontSize: 17, fontWeight: 700, letterSpacing: '-0.3px' }}>{trade.asset}</span>
                    {trade.type && (
                      <span className="tag">{trade.type}</span>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>AI Confidence</span>
                    <span style={{
                      fontSize: 15, fontWeight: 700,
                      color: isBuy ? 'var(--green)' : 'var(--red)'
                    }}>
                      {trade.confidence}%
                    </span>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="confidence-bar" style={{ height: 4 }}>
                  <div
                    className={`confidence-fill ${isBuy ? 'buy' : 'sell'}`}
                    style={{ width: `${trade.confidence || 0}%` }}
                  />
                </div>

                {/* Agent verdict badges */}
                {(trade.risk_verdict || trade.debate_verdict || trade.news_sentiment) && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    {trade.risk_verdict && (() => {
                      const rv = trade.risk_verdict;
                      const rvColor = rv === 'GREEN' ? '#26a69a' : rv === 'AMBER' ? '#ffa726' : '#ef5350';
                      const rvBg   = rv === 'GREEN' ? 'rgba(38,166,154,.12)' : rv === 'AMBER' ? 'rgba(255,167,38,.12)' : 'rgba(239,83,80,.12)';
                      return (
                        <span title="Portfolio Manager risk verdict" style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: rvBg, color: rvColor, border: `1px solid ${rvColor}33`, letterSpacing: '0.05em' }}>
                          ⚖ {rv}
                        </span>
                      );
                    })()}
                    {trade.debate_verdict && (() => {
                      const dv = trade.debate_verdict;
                      const dvColor = dv === 'CONFIRMED' ? '#26a69a' : dv === 'DOWNGRADED' ? '#ffa726' : '#ef5350';
                      const dvBg   = dv === 'CONFIRMED' ? 'rgba(38,166,154,.12)' : dv === 'DOWNGRADED' ? 'rgba(255,167,38,.12)' : 'rgba(239,83,80,.12)';
                      return (
                        <span title="Bull/Bear debate verdict" style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: dvBg, color: dvColor, border: `1px solid ${dvColor}33`, letterSpacing: '0.05em' }}>
                          🗣 {dv}
                        </span>
                      );
                    })()}
                    {trade.news_sentiment && (() => {
                      const ns = trade.news_sentiment;
                      const nsColor = ns === 'POSITIVE' ? '#26a69a' : ns === 'NEGATIVE' ? '#ef5350' : '#787b86';
                      const nsBg   = ns === 'POSITIVE' ? 'rgba(38,166,154,.12)' : ns === 'NEGATIVE' ? 'rgba(239,83,80,.12)' : 'rgba(120,123,134,.12)';
                      return (
                        <span title="News sentiment" style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: nsBg, color: nsColor, border: `1px solid ${nsColor}33`, letterSpacing: '0.05em' }}>
                          📰 {ns}
                        </span>
                      );
                    })()}
                    {trade.gate_reason && (
                      <span title={`Gated by ${trade.gated_by}`} style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(239,83,80,.12)', color: '#ef5350', border: '1px solid rgba(239,83,80,.2)', letterSpacing: '0.05em', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        🚫 {trade.gate_reason}
                      </span>
                    )}
                  </div>
                )}

                {/* Trade levels + allocation breakdown */}
                {(() => {
                  const tp  = Number(trade.targetPrice) || 0;
                  const sl  = Number(trade.stopLoss)    || 0;
                  const upPct   = price > 0 && tp  > 0 ? (isBuy ? (tp - price) / price * 100 : (price - tp) / price * 100) : null;
                  const downPct = price > 0 && sl  > 0 ? (isBuy ? (price - sl) / price * 100 : (sl - price) / price * 100) : null;
                  const rr = upPct && downPct && downPct > 0 ? (upPct / downPct).toFixed(1) : null;
                  return (
                    <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-tertiary)' }}>
                      {upPct   !== null && <span style={{ color: 'var(--green)', fontWeight: 600 }}>▲ +{upPct.toFixed(1)}% potential</span>}
                      {downPct !== null && <span style={{ color: 'var(--red)',   fontWeight: 600 }}>▼ -{downPct.toFixed(1)}% risk</span>}
                      {rr      !== null && <span style={{ color: 'var(--text-secondary)' }}>· R:R {rr}×</span>}
                    </div>
                  );
                })()}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr) 1.4fr', gap: 12 }}>
                  <LevelBox label="Entry Price"  value={price    ? `₹${price.toFixed(2)}`          : '—'} />
                  <LevelBox label="Target"       value={trade.targetPrice ? `₹${Number(trade.targetPrice).toFixed(2)}` : '—'} color="var(--green)" />
                  <LevelBox label="Stop Loss"    value={trade.stopLoss    ? `₹${Number(trade.stopLoss).toFixed(2)}`    : '—'} color="var(--red)" />

                  {/* Allocation box — highlighted */}
                  <div style={{
                    background: tooSmall ? 'var(--red-dim)' : 'var(--accent-dim)',
                    border: `1px solid ${tooSmall ? 'rgba(244,63,94,0.2)' : 'rgba(59,130,246,0.2)'}`,
                    borderRadius: 'var(--radius-md)', padding: '10px 14px',
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: tooSmall ? 'var(--red)' : 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
                      {tooSmall ? '⚠ Too Small' : 'Your Allocation'}
                    </div>
                    {tooSmall ? (
                      <div style={{ fontSize: 12, color: 'var(--red)' }}>
                        {fmtK(perTradeAmount)} can't buy 1 share at ₹{price?.toFixed(0)}
                      </div>
                    ) : (
                      <>
                        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                          {qty.toLocaleString()} shares
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                          {fmt(value)} of {fmtK(perTradeAmount)} budget
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
                          {settings.perTradePct}% of {fmtK(settings.totalCapital)} pool
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Age */}
                {trade.age && (
                  <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                    Generated {trade.age}
                  </div>
                )}

                {/* Action buttons */}
                <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 2 }}>
                  <button
                    className="btn-secondary"
                    onClick={() => handleAction(trade.id, 'reject')}
                    disabled={isExec}
                    style={{ gap: 6 }}
                  >
                    <X size={13} /> Reject
                  </button>
                  <button
                    className="btn-primary"
                    onClick={() => handleAction(trade.id, 'approve')}
                    disabled={isExec || tooSmall}
                    style={{
                      gap: 6, minWidth: 180, justifyContent: 'center',
                      background: tooSmall
                        ? 'var(--bg-overlay)'
                        : isBuy
                          ? 'linear-gradient(135deg, #10b981, #059669)'
                          : 'linear-gradient(135deg, #f43f5e, #e11d48)',
                      boxShadow: tooSmall ? 'none' : isBuy
                        ? '0 4px 14px rgba(16,185,129,0.3)'
                        : '0 4px 14px rgba(244,63,94,0.3)',
                      cursor: tooSmall ? 'not-allowed' : 'pointer',
                      opacity: isExec ? 0.6 : 1,
                    }}
                  >
                    {isExec ? (
                      <>Executing…</>
                    ) : tooSmall ? (
                      <><Lock size={13} /> Increase Capital</>
                    ) : (
                      <><Check size={13} /> {isBuy ? 'Buy' : 'Sell'} {qty} × {trade.asset} · {fmt(value)}</>
                    )}
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────────
function LevelBox({ label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)',
      padding: '10px 14px', border: '1px solid var(--border-subtle)'
    }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, color: color || 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
    </div>
  );
}

function QuickStat({ icon, label, value, valueColor }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px',
      background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-md)', flex: 1,
    }}>
      <span style={{ color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center' }}>{icon}</span>
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', lineHeight: 1 }}>
          {label}
        </div>
        <div style={{ fontSize: 13, fontWeight: 700, color: valueColor || 'var(--text-primary)', marginTop: 3, fontVariantNumeric: 'tabular-nums' }}>
          {value}
        </div>
      </div>
    </div>
  );
}

function CapStat({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: color || 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
    </div>
  );
}

// ── CSS helpers ─────────────────────────────────────────────────────
const labelStyle = {
  fontSize: 11, fontWeight: 600, color: 'var(--text-tertiary)',
  textTransform: 'uppercase', letterSpacing: '0.08em',
  display: 'block', marginBottom: 8,
};

const hintStyle = {
  fontSize: 11, color: 'var(--text-tertiary)', marginTop: 5,
};
