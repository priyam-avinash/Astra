/**
 * AgentPanel — Reusable multi-agent analysis display component.
 *
 * Receives the enriched signal payload (from /api/analyze/{symbol}/enrich)
 * and renders tiered agent results:
 *   - LLM mode toggle (per-symbol override)
 *   - Auto-execute override (advisory / auto)
 *   - Tier 1: News sentiment card + Fundamentals grade
 *   - Tier 2: Bull/Bear Debate card with expandable arguments
 *   - Tier 3: Risk Debate + Portfolio Manager decision card
 *   - Gate banner (if signal was suppressed by an agent)
 *   - Continuous learning: past lessons for this ticker
 */

import React, { useState } from 'react';
import {
  Brain, Newspaper, TrendingUp, TrendingDown, Shield,
  ChevronDown, ChevronUp, Zap, AlertTriangle, CheckCircle,
  XCircle, BookOpen, Loader, RefreshCw
} from 'lucide-react';

// ── Palette helpers ──────────────────────────────────────────────────────
const VERDICT_COLOR = {
  CONFIRMED:  '#26a69a',
  DOWNGRADED: '#ffc107',
  REJECTED:   '#ef5350',
};
const RISK_COLOR = {
  GREEN: '#26a69a',
  AMBER: '#ffc107',
  RED:   '#ef5350',
};
const SENT_COLOR = {
  POSITIVE: '#26a69a',
  NEGATIVE: '#ef5350',
  NEUTRAL:  '#9ca3af',
  MIXED:    '#ffc107',
};
const GRADE_COLOR = { A: '#26a69a', B: '#66bb6a', C: '#ffc107', D: '#ff7043', F: '#ef5350' };

const S = {
  panel:     { background: '#1a1d2e', border: '1px solid #2a2e39', borderRadius: 12, overflow: 'hidden' },
  hdr:       { padding: '14px 18px', borderBottom: '1px solid #2a2e39', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  hdrTitle:  { fontSize: 13, fontWeight: 700, color: '#e1e4ea', display: 'flex', alignItems: 'center', gap: 8 },
  body:      { padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 },
  tag:       (color) => ({ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 9px', borderRadius: 20, fontSize: 11, fontWeight: 700, background: `${color}22`, color }),
  row:       { display: 'flex', alignItems: 'flex-start', gap: 12 },
  label:     { fontSize: 11, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 },
  value:     { fontSize: 13, color: '#e1e4ea', lineHeight: 1.5 },
  subcard:   { background: '#0f1118', border: '1px solid #2a2e39', borderRadius: 8, padding: '12px 14px' },
  expandBtn: { background: 'none', border: 'none', cursor: 'pointer', color: '#6b7280', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, padding: 0, marginTop: 6 },
  divider:   { borderTop: '1px solid #2a2e39' },
  bullet:    { fontSize: 12, color: '#9ca3af', lineHeight: 1.6 },
};

// ── Sub-cards ─────────────────────────────────────────────────────────────

function NewsCard({ news }) {
  if (!news) return null;
  const color = SENT_COLOR[news.sentiment] || '#9ca3af';
  return (
    <div style={S.subcard}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 700, color: '#9ca3af' }}>
          <Newspaper size={12} /> News Analyst
        </div>
        <span style={S.tag(color)}>{news.sentiment}</span>
      </div>
      <div style={{ fontSize: 13, color: '#d1d5db', lineHeight: 1.6, marginBottom: news.key_events?.length ? 10 : 0 }}>
        {news.summary}
      </div>
      {news.key_events?.length > 0 && (
        <ul style={{ margin: 0, paddingLeft: 16 }}>
          {news.key_events.map((e, i) => (
            <li key={i} style={S.bullet}>{e}</li>
          ))}
        </ul>
      )}
      {news.should_gate && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#ef5350', display: 'flex', alignItems: 'center', gap: 4 }}>
          <AlertTriangle size={11} /> BUY signal suppressed by negative news
        </div>
      )}
    </div>
  );
}

function FundamentalsCard({ fund }) {
  if (!fund) return null;
  const color = GRADE_COLOR[fund.grade] || '#9ca3af';
  return (
    <div style={S.subcard}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#9ca3af' }}>📊 Fundamentals</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 80, height: 5, background: '#2a2e39', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ width: `${fund.score}%`, height: '100%', background: color, borderRadius: 3, transition: 'width .5s' }} />
          </div>
          <span style={{ ...S.tag(color), fontSize: 13, padding: '2px 10px' }}>{fund.grade}</span>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#9ca3af', lineHeight: 1.5 }}>{fund.message}</div>
      {fund.gate_long && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#ff7043', display: 'flex', alignItems: 'center', gap: 4 }}>
          <AlertTriangle size={11} /> Multi-day BUY gated — weak fundamentals
        </div>
      )}
    </div>
  );
}

function DebateCard({ debate }) {
  const [showBull, setShowBull]   = useState(false);
  const [showBear, setShowBear]   = useState(false);
  if (!debate) return null;
  const color = VERDICT_COLOR[debate.verdict] || '#9ca3af';
  return (
    <div style={S.subcard}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#9ca3af', display: 'flex', alignItems: 'center', gap: 6 }}>
          <TrendingUp size={12} color="#26a69a" /> Bull
          <span style={{ color: '#2a2e39' }}>⟷</span>
          <TrendingDown size={12} color="#ef5350" /> Bear Debate
        </div>
        <span style={S.tag(color)}>{debate.verdict}</span>
      </div>

      <div style={{ fontSize: 13, color: '#d1d5db', lineHeight: 1.5, marginBottom: 8 }}>
        {debate.verdict_reasoning}
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12, color: '#9ca3af' }}>
          Position multiplier: <strong style={{ color: debate.suggested_position_multiplier > 0 ? '#26a69a' : '#ef5350' }}>
            {(debate.suggested_position_multiplier * 100).toFixed(0)}%
          </strong>
        </span>
        {debate.source === 'rule_based' && (
          <span style={{ fontSize: 11, color: '#ffc107' }}>⚠ Rule-based (LLM unavailable)</span>
        )}
      </div>

      {debate.bull_argument && debate.bull_argument !== 'LLM unavailable — rule-based assessment used.' && (
        <>
          <button style={{ ...S.expandBtn, color: '#26a69a' }} onClick={() => setShowBull(!showBull)}>
            <TrendingUp size={11} /> Bull argument {showBull ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
          {showBull && (
            <div style={{ marginTop: 8, padding: '8px 12px', background: 'rgba(38,166,154,.08)', borderRadius: 6, fontSize: 12, color: '#a7f3d0', lineHeight: 1.6, borderLeft: '2px solid #26a69a' }}>
              {debate.bull_argument}
            </div>
          )}
          <button style={{ ...S.expandBtn, color: '#ef5350' }} onClick={() => setShowBear(!showBear)}>
            <TrendingDown size={11} /> Bear argument {showBear ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
          {showBear && (
            <div style={{ marginTop: 8, padding: '8px 12px', background: 'rgba(239,83,80,.08)', borderRadius: 6, fontSize: 12, color: '#fca5a5', lineHeight: 1.6, borderLeft: '2px solid #ef5350' }}>
              {debate.bear_argument}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function PortfolioCard({ portfolio, onExecute, autoExecute }) {
  const [showAgg, setShowAgg]   = useState(false);
  const [showCon, setShowCon]   = useState(false);
  const [showNeu, setShowNeu]   = useState(false);
  if (!portfolio) return null;
  const color = RISK_COLOR[portfolio.risk_verdict] || '#9ca3af';
  const icon  = portfolio.risk_verdict === 'GREEN'
    ? <CheckCircle size={13} color="#26a69a" />
    : portfolio.risk_verdict === 'RED'
      ? <XCircle size={13} color="#ef5350" />
      : <AlertTriangle size={13} color="#ffc107" />;

  return (
    <div style={{ ...S.subcard, border: `1px solid ${color}44` }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 700, color: '#9ca3af' }}>
          <Shield size={12} /> Portfolio Manager
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {icon}
          <span style={{ ...S.tag(color), fontSize: 12 }}>{portfolio.risk_verdict}</span>
        </div>
      </div>

      {/* Executive Summary */}
      <div style={{ fontSize: 13, color: '#d1d5db', lineHeight: 1.6, marginBottom: 10 }}>
        {portfolio.executive_summary}
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 10 }}>
        <div>
          <div style={S.label}>Position Size</div>
          <div style={{ fontSize: 16, fontWeight: 700, color }}>{portfolio.recommended_position_pct?.toFixed(0)}%</div>
        </div>
        {portfolio.price_target && (
          <div>
            <div style={S.label}>PM Target</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#26a69a' }}>₹{Number(portfolio.price_target).toFixed(2)}</div>
          </div>
        )}
        {portfolio.stop_loss && (
          <div>
            <div style={S.label}>PM Stop Loss</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#ef5350' }}>₹{Number(portfolio.stop_loss).toFixed(2)}</div>
          </div>
        )}
        <div>
          <div style={S.label}>Time Horizon</div>
          <div style={{ fontSize: 13, color: '#e1e4ea' }}>{portfolio.time_horizon}</div>
        </div>
      </div>

      {/* Investment thesis */}
      {portfolio.investment_thesis && (
        <div style={{ fontSize: 12, color: '#9ca3af', lineHeight: 1.6, marginBottom: 10, borderLeft: '2px solid #2a2e39', paddingLeft: 10 }}>
          {portfolio.investment_thesis}
        </div>
      )}

      {/* Key risks */}
      {portfolio.key_risks && (
        <div style={{ fontSize: 12, color: '#ff7043', lineHeight: 1.5, marginBottom: 10, display: 'flex', gap: 6 }}>
          <AlertTriangle size={11} style={{ flexShrink: 0, marginTop: 2 }} />
          {portfolio.key_risks}
        </div>
      )}

      {/* Debate transcripts (expandable) */}
      {portfolio.aggressive_view && portfolio.aggressive_view !== 'LLM unavailable.' && (
        <div style={{ borderTop: '1px solid #2a2e39', paddingTop: 10, marginTop: 4 }}>
          <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Risk Committee Debate</div>
          {[
            { key: 'agg',  label: 'Aggressive', text: portfolio.aggressive_view,   color: '#ef5350', open: showAgg, toggle: () => setShowAgg(!showAgg) },
            { key: 'con',  label: 'Conservative', text: portfolio.conservative_view, color: '#26a69a', open: showCon, toggle: () => setShowCon(!showCon) },
            { key: 'neu',  label: 'Neutral',    text: portfolio.neutral_view,      color: '#ffc107', open: showNeu, toggle: () => setShowNeu(!showNeu) },
          ].map(({ label, text, color: c, open, toggle }) => text && (
            <div key={label}>
              <button style={{ ...S.expandBtn, color: c }} onClick={toggle}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: c, display: 'inline-block' }} />
                {label} {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              </button>
              {open && (
                <div style={{ margin: '6px 0 8px', padding: '8px 12px', background: `${c}11`, borderRadius: 6, fontSize: 12, color: '#d1d5db', lineHeight: 1.6, borderLeft: `2px solid ${c}` }}>
                  {text}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Execute button */}
      {portfolio.risk_verdict !== 'RED' && onExecute && (
        <div style={{ borderTop: '1px solid #2a2e39', paddingTop: 12, marginTop: 8 }}>
          {portfolio.auto_execute ? (
            <div style={{ fontSize: 12, color: '#26a69a', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Zap size={12} /> Auto-executing paper trade…
            </div>
          ) : (
            <button
              onClick={onExecute}
              style={{ padding: '9px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: 13, background: color, color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <Zap size={13} /> Execute Paper Trade
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function GateBanner({ gated_by, gate_reason }) {
  if (!gated_by) return null;
  return (
    <div style={{ background: 'rgba(239,83,80,.1)', border: '1px solid rgba(239,83,80,.3)', borderRadius: 8, padding: '10px 14px', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
      <XCircle size={16} color="#ef5350" style={{ flexShrink: 0, marginTop: 1 }} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#ef5350', marginBottom: 3 }}>
          Signal Suppressed by {gated_by.replace('_', ' ')}
        </div>
        <div style={{ fontSize: 12, color: '#fca5a5', lineHeight: 1.5 }}>{gate_reason}</div>
      </div>
    </div>
  );
}

function LessonsCard({ lessons }) {
  const [open, setOpen] = useState(false);
  if (!lessons?.length) return null;
  return (
    <div style={S.subcard}>
      <button
        style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 0 }}
        onClick={() => setOpen(!open)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 700, color: '#9c27b0' }}>
          <BookOpen size={12} /> ASTRA Memory ({lessons.length} lesson{lessons.length !== 1 ? 's' : ''})
        </div>
        {open ? <ChevronUp size={12} color="#6b7280" /> : <ChevronDown size={12} color="#6b7280" />}
      </button>
      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {lessons.map((l, i) => (
            <div key={i} style={{ fontSize: 12, color: '#c4b5fd', lineHeight: 1.6, padding: '7px 10px', background: 'rgba(156,39,176,.08)', borderRadius: 6, borderLeft: '2px solid #9c27b0' }}>
              {l.content || l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main AgentPanel export ────────────────────────────────────────────────
export default function AgentPanel({
  symbol,
  enrichedData,        // enriched signal payload from /api/analyze/{symbol}/enrich
  loading,             // bool — showing spinner while pipeline runs
  llmEnabled,          // bool — current per-symbol LLM toggle state
  onToggleLLM,         // fn() — toggle per-symbol LLM
  autoExecOverride,    // "advisory" | "auto"
  onToggleAutoExec,    // fn()
  onExecuteTrade,      // fn(enrichedData) — called when user clicks Execute
  onRunAnalysis,       // fn() — trigger pipeline from parent
  lessons,             // [] from /api/agent/memories/{symbol}
}) {
  const data = enrichedData || {};

  return (
    <div style={S.panel}>
      {/* Panel Header with toggles */}
      <div style={S.hdr}>
        <div style={S.hdrTitle}>
          <Brain size={14} color="#2962ff" /> AI Agent Analysis
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          {/* Auto-exec override */}
          {llmEnabled && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#6b7280' }}>
              <Zap size={12} />
              <span>Auto</span>
              <button
                onClick={onToggleAutoExec}
                style={{ width: 34, height: 19, borderRadius: 10, background: autoExecOverride === 'auto' ? '#ffc107' : '#2a2e39', cursor: 'pointer', border: 'none', position: 'relative', transition: 'background .2s' }}
              >
                <div style={{ position: 'absolute', top: 2, left: autoExecOverride === 'auto' ? 17 : 2, width: 15, height: 15, borderRadius: '50%', background: '#fff', transition: 'left .2s' }} />
              </button>
            </div>
          )}
          {/* LLM toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#6b7280' }}>
            <Brain size={12} />
            <span>LLM</span>
            <button
              onClick={onToggleLLM}
              style={{ width: 34, height: 19, borderRadius: 10, background: llmEnabled ? '#2962ff' : '#2a2e39', cursor: 'pointer', border: 'none', position: 'relative', transition: 'background .2s' }}
            >
              <div style={{ position: 'absolute', top: 2, left: llmEnabled ? 17 : 2, width: 15, height: 15, borderRadius: '50%', background: '#fff', transition: 'left .2s' }} />
            </button>
          </div>
        </div>
      </div>

      {/* Body */}
      <div style={S.body}>
        {!llmEnabled && (
          <div style={{ fontSize: 13, color: '#6b7280', textAlign: 'center', padding: '12px 0' }}>
            Enable LLM above to run agent analysis for this symbol.<br />
            <span style={{ fontSize: 12 }}>Global LLM can be enabled in Settings.</span>
          </div>
        )}

        {llmEnabled && loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: '20px 0', color: '#6b7280' }}>
            <RefreshCw size={22} color="#2962ff" style={{ animation: 'spin 1s linear infinite' }} />
            <div style={{ fontSize: 13 }}>Running agent pipeline…</div>
            <div style={{ fontSize: 12 }}>News → Fundamentals → Debate → Risk → Portfolio Manager</div>
          </div>
        )}

        {llmEnabled && !loading && data.llm_enriched && (
          <>
            {/* Gate banner (if signal was suppressed) */}
            {data.gated_by && (
              <GateBanner gated_by={data.gated_by} gate_reason={data.gate_reason} />
            )}

            {/* Tier 1 */}
            <NewsCard news={data.news} />
            <FundamentalsCard fund={data.fundamentals} />

            {/* Tier 2 */}
            {data.debate && <DebateCard debate={data.debate} />}

            {/* Tier 3 */}
            {data.portfolio && (
              <PortfolioCard
                portfolio={data.portfolio}
                onExecute={onExecuteTrade ? () => onExecuteTrade(data) : null}
                autoExecute={data.auto_execute}
              />
            )}

            {/* Pipeline timing */}
            {data.agent_pipeline_ms > 0 && (
              <div style={{ fontSize: 11, color: '#4b5563', textAlign: 'right' }}>
                Pipeline completed in {data.agent_pipeline_ms}ms · Provider: {data.portfolio?.source || '—'}
              </div>
            )}

            {/* Memory / lessons */}
            {lessons?.length > 0 && <LessonsCard lessons={lessons} />}
          </>
        )}

        {llmEnabled && !loading && !data.llm_enriched && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: '20px 0' }}>
            <div style={{ fontSize: 13, color: '#6b7280', textAlign: 'center' }}>
              Run the multi-agent pipeline for <strong style={{ color: '#e1e4ea' }}>{symbol}</strong>.<br/>
              <span style={{ fontSize: 12 }}>News → Fundamentals → Bull/Bear Debate → Risk → Portfolio Manager</span>
            </div>
            <button
              onClick={onRunAnalysis}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '9px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', background: '#2962ff', color: '#fff', fontWeight: 700, fontSize: 13, fontFamily: "'Inter', sans-serif", transition: 'background .2s' }}
              onMouseEnter={e => e.currentTarget.style.background = '#1e4fd8'}
              onMouseLeave={e => e.currentTarget.style.background = '#2962ff'}
            >
              <Brain size={14} /> Run Agent Analysis
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
