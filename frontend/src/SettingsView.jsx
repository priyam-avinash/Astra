import React, { useState, useEffect, useCallback } from 'react';
import { API_URL } from './config';

import {
  Settings, Eye, EyeOff, Save, RefreshCw, CheckCircle, XCircle,
  AlertTriangle, Brain, Zap, ChevronDown, ChevronUp, Info, Database, Activity
} from 'lucide-react';
import BrokersPanel from './BrokersPanel';

const API = API_URL;

const S = {
  page:      { height: '100%', width: '100%', display: 'flex', flexDirection: 'column', paddingTop: 64, paddingLeft: 24, paddingRight: 24, paddingBottom: 32, background: '#0f1118', color: '#e1e4ea', fontFamily: "'Inter', -apple-system, sans-serif", overflowY: 'auto', boxSizing: 'border-box' },
  inner:     { maxWidth: 760, margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: 24 },
  heading:   { fontSize: 22, fontWeight: 700, color: '#fff', marginBottom: 4 },
  sub:       { fontSize: 13, color: '#6b7280' },
  card:      { background: '#1a1d2e', border: '1px solid #2a2e39', borderRadius: 12, padding: '20px 24px' },
  cardTitle: { fontSize: 13, fontWeight: 700, color: '#9ca3af', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 },
  label:     { fontSize: 14, color: '#d1d5db', fontWeight: 500 },
  hint:      { fontSize: 12, color: '#6b7280', marginTop: 2 },
  input:     { background: '#0f1118', border: '1px solid #2a2e39', borderRadius: 8, padding: '9px 12px', color: '#e1e4ea', fontSize: 13, fontFamily: 'monospace', outline: 'none', width: '100%', boxSizing: 'border-box' },
  select:    { background: '#0f1118', border: '1px solid #2a2e39', borderRadius: 8, padding: '9px 12px', color: '#e1e4ea', fontSize: 13, outline: 'none', cursor: 'pointer', minWidth: 280 },
  btn:       { padding: '10px 24px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 14, display: 'flex', alignItems: 'center', gap: 8, transition: 'background .2s' },
  divider:   { borderTop: '1px solid #2a2e39', margin: '16px 0' },
};

const badge = (color) => ({
  display: 'inline-flex', alignItems: 'center', gap: 4,
  padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
  background: color === 'green' ? 'rgba(38,166,154,.15)' : color === 'red' ? 'rgba(239,83,80,.15)' : 'rgba(255,193,7,.15)',
  color: color === 'green' ? '#26a69a' : color === 'red' ? '#ef5350' : '#ffc107',
});

function Toggle({ value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      style={{ width: 42, height: 24, borderRadius: 12, background: value ? '#2962ff' : '#2a2e39', cursor: 'pointer', border: 'none', position: 'relative', transition: 'background .2s', flexShrink: 0 }}
    >
      <div style={{ position: 'absolute', top: 3, left: value ? 21 : 3, width: 18, height: 18, borderRadius: '50%', background: '#fff', transition: 'left .2s' }} />
    </button>
  );
}

function PasswordField({ value, onChange, placeholder }) {
  const [show, setShow] = useState(false);
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <input
        type={show ? 'text' : 'password'}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{ ...S.input, paddingRight: 38 }}
      />
      <button
        type="button"
        onClick={() => setShow(!show)}
        style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', padding: 0 }}
      >
        {show ? <EyeOff size={14} /> : <Eye size={14} />}
      </button>
    </div>
  );
}

function ProviderStatus({ status }) {
  if (!status) return null;
  const { active_provider, anthropic_configured, groq_configured } = status;
  const color = active_provider === 'anthropic' ? 'green' : active_provider === 'groq' ? 'yellow' : 'red';
  const label = active_provider === 'anthropic' ? '🟢 Claude (Anthropic)' : active_provider === 'groq' ? '🟡 Groq / Llama 3' : '🔴 No Provider — LLM Disabled';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', background: '#0f1118', borderRadius: 8, border: '1px solid #2a2e39', marginBottom: 16 }}>
      <span style={badge(color)}>{label}</span>
      <span style={{ fontSize: 12, color: '#6b7280' }}>
        {anthropic_configured ? '✓ Anthropic' : '✗ Anthropic'} · {groq_configured ? '✓ Groq' : '✗ Groq'}
      </span>
    </div>
  );
}

function InfoBox({ children, color = 'blue' }) {
  const bg     = color === 'amber' ? 'rgba(255,193,7,.07)' : 'rgba(41,98,255,.07)';
  const border = color === 'amber' ? 'rgba(255,193,7,.25)' : 'rgba(41,98,255,.25)';
  const icon   = color === 'amber' ? <AlertTriangle size={13} color="#ffc107" /> : <Info size={13} color="#2962ff" />;
  return (
    <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: '10px 14px', fontSize: 13, color: '#9ca3af', marginBottom: 16, display: 'flex', gap: 10, lineHeight: 1.6 }}>
      <span style={{ flexShrink: 0, marginTop: 2 }}>{icon}</span>
      <span>{children}</span>
    </div>
  );
}

function FieldRow({ label, hint, children }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={S.label}>{label}</div>
          {hint && <div style={S.hint}>{hint}</div>}
        </div>
        <div style={{ flexShrink: 0 }}>{children}</div>
      </div>
    </div>
  );
}

function SectionCard({ title, icon, open, onToggle, children }) {
  return (
    <div style={S.card}>
      <button
        onClick={onToggle}
        style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginBottom: open ? 16 : 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 700, color: '#9ca3af', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {icon}{title}
        </span>
        {open ? <ChevronUp size={14} color="#6b7280" /> : <ChevronDown size={14} color="#6b7280" />}
      </button>
      {open && children}
    </div>
  );
}

function AgentTierCard({ tier, color, title, desc, alwaysOn, alwaysDebate, onToggleAlways }) {
  return (
    <div style={{ background: '#0f1118', border: '1px solid #2a2e39', borderRadius: 8, padding: '12px 14px', marginBottom: 10 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 10, fontWeight: 800, color, background: `${color}22`, padding: '2px 7px', borderRadius: 4 }}>{tier}</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#e1e4ea' }}>{title}</span>
            {alwaysOn && <span style={{ fontSize: 10, color: '#26a69a', background: 'rgba(38,166,154,.15)', padding: '2px 6px', borderRadius: 4 }}>Always On</span>}
          </div>
          <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.6 }}>{desc}</div>
        </div>
        {onToggleAlways !== undefined && (
          <div style={{ flexShrink: 0, textAlign: 'right' }}>
            <div style={{ fontSize: 10, color: '#6b7280', marginBottom: 4 }}>Always run</div>
            <Toggle value={!!alwaysDebate} onChange={onToggleAlways} />
          </div>
        )}
      </div>
    </div>
  );
}


// ── Market data provider health ────────────────────────────────────────────
const PROVIDER_LABELS = {
  yahoo: 'Yahoo Finance', yfinance: 'yfinance (Yahoo)', nse: 'NSE India', binance: 'Binance (crypto)',
  dhan: 'Dhan HQ', upstox: 'Upstox', twelve_data: 'Twelve Data', alpha_vantage: 'Alpha Vantage', yahoo_fundamentals: 'Fundamentals (Yahoo)',
};
const STATUS_COLOR = { ok: 'green', idle: 'yellow', cooldown: 'red', error: 'red', not_configured: 'grey' };

function DataHealthCard({ open, onToggle }) {
  const [health, setHealth] = useState(null);
  const [probing, setProbing] = useState(false);
  const [err, setErr] = useState('');

  const fetchHealth = useCallback(async (probe = false) => {
    setErr('');
    if (probe) setProbing(true);
    try {
      const tok = localStorage.getItem('astra_token');
      const r = await fetch(`${API}/api/data/health${probe ? '?probe=true' : ''}`,
        { headers: tok ? { Authorization: `Bearer ${tok}` } : {} });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setHealth(await r.json());
    } catch (e) { setErr(`Backend unreachable (${e.message}). Is it running on :8000?`); }
    setProbing(false);
  }, []);

  useEffect(() => { fetchHealth(false); const t = setInterval(() => fetchHealth(false), 30000); return () => clearInterval(t); }, [fetchHealth]);

  const probe = health?.probe || {};
  const probeFor = (name) => Object.entries(probe).filter(([k]) => k.startsWith(name + ':'));

  return (
    <SectionCard title="Market Data Providers" icon={<Activity size={14} />} open={open} onToggle={onToggle}>
      <InfoBox>
        Free providers (Yahoo, NSE, Binance) need <strong>no key</strong>. Keys below are optional extras.
        Every trade in ASTRA is <strong>paper-only</strong>. No real orders are ever sent, even with Dhan configured.
      </InfoBox>
      {err && <div style={{ ...badge('red'), marginBottom: 12 }}><XCircle size={12} /> {err}</div>}
      {health && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
          {Object.entries(health.providers).map(([name, p]) => {
            const c = STATUS_COLOR[p.status] || 'yellow';
            const probes = probeFor(name);
            return (
              <div key={name} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, padding: '8px 0', borderBottom: '1px solid #232636' }}>
                <div style={{ width: 170, fontSize: 13, color: '#d1d5db', fontWeight: 600 }}>{PROVIDER_LABELS[name] || name}</div>
                <span style={c === 'grey' ? { ...badge('yellow'), background: 'rgba(107,114,128,.15)', color: '#9ca3af' } : badge(c)}>
                  {p.status.replace('_', ' ')}
                </span>
                <div style={{ flex: 1, fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>
                  {p.ok > 0 && <span style={{ color: '#26a69a' }}>{p.ok} ok </span>}
                  {p.failed > 0 && <span>{p.failed} failed </span>}
                  {p.last_error && p.status !== 'ok' && <div style={{ color: '#ef9a9a', wordBreak: 'break-word' }}>{p.last_error}</div>}
                  {p.cooldown_until && <div>retrying after {p.cooldown_until.slice(11)}</div>}
                  {probes.map(([k, v]) => (
                    <div key={k} style={{ color: v.ok ? '#26a69a' : '#ef9a9a' }}>
                      {k.split(':')[1]}: {v.ok ? `✓ ${v.bars} bars · last ${v.last_close} (${v.last_date}) · ${v.secs}s` : `✗ ${v.detail}`}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          <div style={{ fontSize: 12, color: '#6b7280' }}>
            Chrome TLS impersonation (curl_cffi): {health.curl_cffi ? <span style={{ color: '#26a69a' }}>on</span> : <span style={{ color: '#ef5350' }}>missing (pip install curl_cffi)</span>}
            {' · '}Dhan live feed: {health.dhan_feed?.live ? 'streaming' : (health.dhan_feed?.running ? 'connecting' : 'off')}
          </div>
        </div>
      )}
      <button onClick={() => fetchHealth(true)} disabled={probing}
        style={{ ...S.btn, background: probing ? '#1e2235' : '#2962ff', color: '#fff', alignSelf: 'flex-start' }}>
        <RefreshCw size={14} /> {probing ? 'Testing providers…' : 'Test all providers now'}
      </button>
    </SectionCard>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────
export default function SettingsView() {
  const [form, setForm]       = useState({});
  const [providerStatus, setProviderStatus] = useState(null);
  const [saving, setSaving]   = useState(false);
  const [saveMsg, setSaveMsg] = useState('');
  const [loading, setLoading] = useState(true);
  const [exp, setExp]         = useState({ data: true, dataKeys: false, keys: true, agents: true, exec: true });
  const [keySources, setKeySources] = useState({});

  const token   = localStorage.getItem('astra_token');
  const headers = { ...(token ? { Authorization: `Bearer ${token}` } : {}), 'Content-Type': 'application/json' };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/settings`, { headers });
      if (res.ok) {
        const d = await res.json();
        setProviderStatus(d.provider_status || null);
        setKeySources(d.key_sources || {});
        setForm({
          twelve_data_key:        d.twelve_data_key    || '',
          alpha_vantage_key:      d.alpha_vantage_key  || '',
          dhan_client_id:         d.dhan_client_id     || '',
          dhan_access_token:      d.dhan_access_token  || '',
          llm_enabled:            d.llm_enabled === 'true' || d.llm_enabled === true,
          anthropic_api_key:      d.anthropic_api_key  || '',
          groq_api_key:           d.groq_api_key       || '',
          auto_execute_mode:      d.auto_execute_mode  || 'advisory',
          auto_execute_threshold: d.auto_execute_threshold || 'GREEN',
          always_debate:          d.always_debate === 'true' || d.always_debate === true,
          max_debate_rounds:      parseInt(d.max_debate_rounds  || '1'),
          max_risk_rounds:        parseInt(d.max_risk_rounds    || '1'),
        });
      }
    } catch (_) {}
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, []);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));
  const tog = (k) => setExp(e => ({ ...e, [k]: !e[k] }));

  const save = async () => {
    setSaving(true); setSaveMsg('');
    try {
      const payload = { ...form, llm_enabled: String(form.llm_enabled), always_debate: String(form.always_debate) };
      const res = await fetch(`${API}/api/settings`, { method: 'PUT', headers, body: JSON.stringify({ settings: payload }) });
      const d = await res.json();
      if (res.ok) {
        setSaveMsg(`✓ ${d.updated?.length || 0} settings saved`);
        load();
      } else { setSaveMsg('✗ Save failed'); }
    } catch { setSaveMsg('✗ Connection error'); }
    setSaving(false);
    setTimeout(() => setSaveMsg(''), 4000);
  };

  if (loading) return (
    <div style={{ ...S.page, alignItems: 'center', justifyContent: 'center' }}>
      <RefreshCw size={24} color="#2962ff" />
    </div>
  );

  return (
    <div style={S.page}>
      <div style={S.inner}>

        {/* Header */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
            <Settings size={22} color="#2962ff" />
            <div style={S.heading}>Settings</div>
          </div>
          <div style={S.sub}>Configure ASTRA's AI agents, auto-execution, and API keys</div>
        </div>

        {/* ── Market data ───────────────────────────────── */}
        <DataHealthCard open={exp.data} onToggle={() => tog('data')} />

        <SectionCard title="Market Data API Keys (optional)" icon={<Database size={14} />} open={exp.dataKeys} onToggle={() => tog('dataKeys')}>
          <InfoBox>
            Keys saved here override <code>backend/.env</code> and take effect immediately (no restart).
            {' '}Twelve Data's free plan covers US stocks and crypto. Indian (NSE) symbols need a paid plan.
            {' '}Dhan needs a <strong>Data API</strong> subscription, and access tokens expire. Dhan is used for <strong>market data only</strong>.
          </InfoBox>
          {[
            ['twelve_data_key', 'Twelve Data API key', 'twelvedata.com: 800 requests/day free', false],
            ['alpha_vantage_key', 'Alpha Vantage API key', 'alphavantage.co: 25 requests/day free, daily bars only', false],
            ['dhan_client_id', 'Dhan client ID', 'From web.dhan.co → My Profile → DhanHQ Trading APIs', true],
            ['dhan_access_token', 'Dhan access token', 'Regenerate when it expires', false],
          ].map(([k, label, hint, plain]) => (
            <FieldRow key={k} label={label} hint={`${hint}${keySources[k] === 'env' ? ' · currently loaded from .env' : ''}`}>
              <div style={{ width: 340 }}>
                {plain
                  ? <input style={S.input} value={form[k] || ''} onChange={e => set(k, e.target.value)} placeholder="optional" />
                  : <PasswordField value={form[k] || ''} onChange={v => set(k, v)} placeholder={keySources[k] === 'env' ? '(set in .env)' : 'optional'} />}
              </div>
            </FieldRow>
          ))}
        </SectionCard>

        {/* ── LLM API Keys ─────────────────────────────── */}
        <SectionCard title="LLM API Keys" icon={<Brain size={14} />} open={exp.keys} onToggle={() => tog('keys')}>
          <ProviderStatus status={providerStatus} />
          <InfoBox>
            Enter your <strong>Anthropic API key</strong> to use Claude (highest quality).
            Without it, ASTRA falls back to <strong>Groq</strong> (free — Llama 3.3-70B, 14,400 req/day).
            All ML engines work regardless of LLM availability.
          </InfoBox>

          <FieldRow label="Anthropic API Key" hint="Preferred provider — Claude Haiku for analysts, Claude Sonnet for Portfolio Manager">
            <div style={{ width: 340 }}>
              <PasswordField value={form.anthropic_api_key || ''} onChange={v => set('anthropic_api_key', v)} placeholder="sk-ant-api03-..." />
            </div>
          </FieldRow>

          <FieldRow label="Groq API Key" hint="Free fallback — get key at console.groq.com · No credit card needed">
            <div style={{ width: 340 }}>
              <PasswordField value={form.groq_api_key || ''} onChange={v => set('groq_api_key', v)} placeholder="gsk_..." />
            </div>
          </FieldRow>

          <div style={{ fontSize: 12, color: '#6b7280' }}>
            {'Groq free tier: '}
            <a href="https://console.groq.com" target="_blank" rel="noreferrer" style={{ color: '#2962ff' }}>console.groq.com</a>
            {' · Anthropic: '}
            <a href="https://console.anthropic.com" target="_blank" rel="noreferrer" style={{ color: '#2962ff' }}>console.anthropic.com</a>
          </div>
        </SectionCard>

        {/* ── Multi-Agent Mode ──────────────────────────── */}
        <SectionCard title="Multi-Agent Pipeline" icon={<Brain size={14} />} open={exp.agents} onToggle={() => tog('agents')}>
          <FieldRow
            label="Enable LLM Agent Pipeline"
            hint="Master switch — when off, ASTRA runs ML-only (fast, no API cost)"
          >
            <Toggle value={!!form.llm_enabled} onChange={v => set('llm_enabled', v)} />
          </FieldRow>

          {form.llm_enabled && (
            <>
              <div style={S.divider} />
              <div style={{ fontSize: 12, color: '#9ca3af', fontWeight: 600, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Agent Tiers</div>

              <AgentTierCard tier="Tier 1" color="#2962ff" alwaysOn
                title="News Analyst + Fundamentals Scorer"
                desc="Fetches news via Alpha Vantage (no extra key needed). LLM classifies as POSITIVE/NEGATIVE/NEUTRAL/MIXED. Rules-based fundamentals scoring (A–F). Negative news gates BUY signals below 65% confidence."
              />
              <AgentTierCard tier="Tier 2" color="#ffc107"
                title="Bull / Bear Debate → Research Manager"
                desc="Two LLM personas argue for and against the trade. A Research Manager arbitrates and delivers CONFIRMED / DOWNGRADED / REJECTED verdict. Runs in confidence grey-zone (45–72%)."
                alwaysDebate={form.always_debate}
                onToggleAlways={() => set('always_debate', !form.always_debate)}
              />
              <AgentTierCard tier="Tier 3" color="#26a69a"
                title="Risk Debate + Portfolio Manager (Final Authority)"
                desc="Aggressive, Conservative, and Neutral risk managers debate. The Portfolio Manager synthesises all context — past lessons, open positions, news, fundamentals — and delivers GREEN / AMBER / RED with recommended position size. RED = trade vetoed."
              />
              <AgentTierCard tier="Tier 4" color="#9c27b0" alwaysOn
                title="Reflector — Continuous Learning"
                desc="After every position closes, an LLM reflects on the outcome and writes 2-4 sentences of analysis. Key lessons are stored in ASTRA's memory and injected into future agent prompts for the same ticker."
              />

              <div style={S.divider} />

              <FieldRow label="Debate Rounds (Tier 2)" hint="1 round = 2 messages (Bull→Bear). More rounds = deeper analysis, slower.">
                <select value={form.max_debate_rounds} onChange={e => set('max_debate_rounds', parseInt(e.target.value))} style={S.select}>
                  <option value={1}>1 round — recommended</option>
                  <option value={2}>2 rounds — deeper</option>
                  <option value={3}>3 rounds — maximum</option>
                </select>
              </FieldRow>

              <FieldRow label="Risk Debate Rounds (Tier 3)" hint="1 round = 3 messages (Aggressive→Conservative→Neutral).">
                <select value={form.max_risk_rounds} onChange={e => set('max_risk_rounds', parseInt(e.target.value))} style={S.select}>
                  <option value={1}>1 round — recommended</option>
                  <option value={2}>2 rounds</option>
                  <option value={3}>3 rounds</option>
                </select>
              </FieldRow>
            </>
          )}
        </SectionCard>

        {/* ── Auto-Execute ──────────────────────────────── */}
        <SectionCard title="Auto-Execute Mode" icon={<Zap size={14} />} open={exp.exec} onToggle={() => tog('exec')}>
          <InfoBox color="amber">
            <strong>Advisory</strong> (default): Portfolio Manager shows verdict but you always click Execute manually.{' '}
            <strong>Auto</strong>: ASTRA paper-trades automatically when the Portfolio Manager verdict passes the threshold.
            The header kill-switch always overrides.
          </InfoBox>

          <FieldRow label="Global Execution Mode" hint="Per-trade override available in Deep Analysis panel">
            <select value={form.auto_execute_mode} onChange={e => set('auto_execute_mode', e.target.value)} style={S.select}>
              <option value="advisory">Advisory — you approve every trade</option>
              <option value="auto">Auto — executes when verdict passes threshold</option>
            </select>
          </FieldRow>

          {form.auto_execute_mode === 'auto' && (
            <FieldRow label="Auto-Execute Threshold" hint="GREEN=high conviction only. GREEN+AMBER=also executes at reduced size.">
              <select value={form.auto_execute_threshold} onChange={e => set('auto_execute_threshold', e.target.value)} style={S.select}>
                <option value="GREEN">GREEN only (most conservative)</option>
                <option value="AMBER_GREEN">GREEN + AMBER (more trades, reduced size on AMBER)</option>
              </select>
            </FieldRow>
          )}
        </SectionCard>

        {/* ── Broker Connections ───────────────────────── */}
        <BrokersPanel />

        {/* ── Save ─────────────────────────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button style={{ ...S.btn, background: '#2962ff', color: '#fff' }} onClick={save} disabled={saving}>
            {saving ? <RefreshCw size={14} /> : <Save size={14} />}
            {saving ? 'Saving…' : 'Save Settings'}
          </button>
          {saveMsg && (
            <span style={{ fontSize: 13, fontWeight: 600, color: saveMsg.startsWith('✓') ? '#26a69a' : '#ef5350' }}>
              {saveMsg}
            </span>
          )}
        </div>

      </div>
    </div>
  );
}
