import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Search, X, BarChart2 } from 'lucide-react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';

const INDICATORS = [
  { id: 'sma9',   name: 'SMA (9)',   cat: 'Moving Averages', color: '#4caf50' },
  { id: 'sma20',  name: 'SMA (20)',  cat: 'Moving Averages', color: '#2962ff' },
  { id: 'sma50',  name: 'SMA (50)',  cat: 'Moving Averages', color: '#e91e63' },
  { id: 'sma100', name: 'SMA (100)', cat: 'Moving Averages', color: '#ff5722' },
  { id: 'sma200', name: 'SMA (200)', cat: 'Moving Averages', color: '#ffc107' },
  { id: 'ema9',   name: 'EMA (9)',   cat: 'Moving Averages', color: '#8bc34a' },
  { id: 'ema12',  name: 'EMA (12)',  cat: 'Moving Averages', color: '#00bcd4' },
  { id: 'ema21',  name: 'EMA (21)',  cat: 'Moving Averages', color: '#03a9f4' },
  { id: 'ema26',  name: 'EMA (26)',  cat: 'Moving Averages', color: '#9c27b0' },
  { id: 'ema50',  name: 'EMA (50)',  cat: 'Moving Averages', color: '#673ab7' },
  { id: 'bb',     name: 'Bollinger Bands (20,2)', cat: 'Volatility', color: '#607d8b' },
  { id: 'kc',     name: 'Keltner Channel (20)', cat: 'Volatility', color: '#795548' },
  { id: 'atr',    name: 'ATR (14)',  cat: 'Volatility', color: '#ff9800' },
  { id: 'vwap',   name: 'VWAP',     cat: 'Volume', color: '#ff9800' },
  { id: 'vol',    name: 'Volume',    cat: 'Volume', color: '#26a69a' },
  { id: 'obv',    name: 'On Balance Volume', cat: 'Volume', color: '#ab47bc' },
  { id: 'macd_line', name: 'MACD Line', cat: 'Oscillators', color: '#2196f3' },
  { id: 'rsi_line', name: 'RSI (14)', cat: 'Oscillators', color: '#ff5722' },
  { id: 'supertrend', name: 'Supertrend', cat: 'Strategies', color: '#4caf50' },
  { id: 'ichimoku_conv', name: 'Ichimoku Conversion', cat: 'Strategies', color: '#2962ff' },
  { id: 'ichimoku_base', name: 'Ichimoku Base',       cat: 'Strategies', color: '#e91e63' },
];

const TF = [
  { label: '5m',  interval: '5m',  period: '5d' },
  { label: '15m', interval: '15m', period: '5d' },
  { label: '1H',  interval: '1h',  period: '1mo' },
  { label: '1D',  interval: '1d',  period: '6mo' },
  { label: '5D',  interval: '5d',  period: '2y' },
  { label: '1M',  interval: '1mo', period: '5y' },
];

function computeSMA(data, period) {
  const r = [];
  for (let i = period - 1; i < data.length; i++) {
    let s = 0; for (let j = i - period + 1; j <= i; j++) s += data[j].close;
    r.push({ time: data[i].time, value: s / period });
  }
  return r;
}
function computeEMA(data, period) {
  const k = 2 / (period + 1), r = []; let prev = null;
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;
    if (prev === null) { let s = 0; for (let j = i - period + 1; j <= i; j++) s += data[j].close; prev = s / period; }
    else { prev = data[i].close * k + prev * (1 - k); }
    r.push({ time: data[i].time, value: prev });
  }
  return r;
}
function computeBB(data, period, mult) {
  const u = [], l = [], m = [];
  for (let i = period - 1; i < data.length; i++) {
    let s = 0; for (let j = i - period + 1; j <= i; j++) s += data[j].close;
    const mean = s / period; let sq = 0;
    for (let j = i - period + 1; j <= i; j++) sq += (data[j].close - mean) ** 2;
    const std = Math.sqrt(sq / period);
    m.push({ time: data[i].time, value: mean });
    u.push({ time: data[i].time, value: mean + mult * std });
    l.push({ time: data[i].time, value: mean - mult * std });
  }
  return { upper: u, lower: l, mid: m };
}
function computeATR(data, period) {
  const r = [];
  for (let i = 1; i < data.length; i++) {
    const tr = Math.max(data[i].high - data[i].low, Math.abs(data[i].high - data[i-1].close), Math.abs(data[i].low - data[i-1].close));
    if (i >= period) {
      let s = 0; for (let j = i - period + 1; j <= i; j++) {
        const t = Math.max(data[j].high - data[j].low, j > 0 ? Math.abs(data[j].high - data[j-1].close) : 0, j > 0 ? Math.abs(data[j].low - data[j-1].close) : 0);
        s += t;
      }
      r.push({ time: data[i].time, value: s / period });
    }
  }
  return r;
}
function computeRSI(data, period) {
  const r = []; let avgGain = 0, avgLoss = 0;
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i-1].close;
    if (i <= period) { if (change > 0) avgGain += change; else avgLoss += Math.abs(change); }
    if (i === period) { avgGain /= period; avgLoss /= period; }
    if (i > period) { const g = change > 0 ? change : 0; const l = change < 0 ? Math.abs(change) : 0; avgGain = (avgGain * (period - 1) + g) / period; avgLoss = (avgLoss * (period - 1) + l) / period; }
    if (i >= period) { const rs = avgLoss === 0 ? 100 : avgGain / avgLoss; r.push({ time: data[i].time, value: 100 - (100 / (1 + rs)) }); }
  }
  return r;
}
function computeOBV(data) {
  const r = []; let obv = 0;
  for (let i = 0; i < data.length; i++) {
    if (i > 0) { if (data[i].close > data[i-1].close) obv += data[i].volume || 0; else if (data[i].close < data[i-1].close) obv -= data[i].volume || 0; }
    r.push({ time: data[i].time, value: obv });
  }
  return r;
}

export default function FullChartModal({ data, symbol, onClose, onLoadInterval }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const [activeIndicators, setActiveIndicators] = useState(['sma200', 'vol']);
  const [showPanel, setShowPanel] = useState(false);
  const [search, setSearch] = useState('');
  const [activeTF, setActiveTF] = useState('1D');
  const [loadingTF, setLoadingTF] = useState(false);

  const sorted = useMemo(() => {
    if (!data?.chartData) return [];
    const used = new Set();
    return [...data.chartData].sort((a, b) => (a.time > b.time ? 1 : -1)).filter(d => { if (used.has(d.time)) return false; used.add(d.time); return true; });
  }, [data]);

  useEffect(() => {
    if (!sorted.length || !containerRef.current) return;
    if (chartRef.current) { try { chartRef.current.remove(); } catch {} }

    const chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#131722' }, textColor: 'rgba(255,255,255,.6)' },
      grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#2a2e39' },
      timeScale: { borderColor: '#2a2e39', timeVisible: activeTF !== '1D' && activeTF !== '5D' && activeTF !== '1M' },
      autoSize: true,
    });
    chartRef.current = chart;
    const cs = chart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' });
    cs.setData(sorted.map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })));

    const ai = activeIndicators;
    if (ai.includes('vol')) { const vs = chart.addHistogramSeries({ color: '#26a69a', priceFormat: { type: 'volume' }, priceScaleId: '', scaleMargins: { top: 0.85, bottom: 0 } }); vs.setData(sorted.map(d => ({ time: d.time, value: d.volume || 0, color: d.close >= d.open ? 'rgba(38,166,154,.3)' : 'rgba(239,83,80,.3)' }))); }

    const smaConf = { sma9: 9, sma20: 20, sma50: 50, sma100: 100, sma200: 200 };
    const smaColors = { sma9: '#4caf50', sma20: '#2962ff', sma50: '#e91e63', sma100: '#ff5722', sma200: '#ffc107' };
    Object.entries(smaConf).forEach(([id, p]) => { if (ai.includes(id)) { const s = chart.addLineSeries({ color: smaColors[id], lineWidth: 1.5, crosshairMarkerVisible: false }); s.setData(computeSMA(sorted, p)); } });

    const emaConf = { ema9: 9, ema12: 12, ema21: 21, ema26: 26, ema50: 50 };
    const emaColors = { ema9: '#8bc34a', ema12: '#00bcd4', ema21: '#03a9f4', ema26: '#9c27b0', ema50: '#673ab7' };
    Object.entries(emaConf).forEach(([id, p]) => { if (ai.includes(id)) { const s = chart.addLineSeries({ color: emaColors[id], lineWidth: 1.5, crosshairMarkerVisible: false }); s.setData(computeEMA(sorted, p)); } });

    if (ai.includes('bb')) { const bb = computeBB(sorted, 20, 2); chart.addLineSeries({ color: '#607d8b', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false }).setData(bb.upper); chart.addLineSeries({ color: '#607d8b', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false }).setData(bb.lower); chart.addLineSeries({ color: '#90a4ae', lineWidth: 1, crosshairMarkerVisible: false }).setData(bb.mid); }
    if (ai.includes('atr')) { const s = chart.addLineSeries({ color: '#ff9800', lineWidth: 1.5, priceScaleId: 'atr', crosshairMarkerVisible: false }); s.setData(computeATR(sorted, 14)); }
    if (ai.includes('rsi_line')) { const s = chart.addLineSeries({ color: '#ff5722', lineWidth: 1.5, priceScaleId: 'rsi', crosshairMarkerVisible: false }); s.setData(computeRSI(sorted, 14)); }
    if (ai.includes('obv')) { const s = chart.addLineSeries({ color: '#ab47bc', lineWidth: 1.5, priceScaleId: 'obv', crosshairMarkerVisible: false }); s.setData(computeOBV(sorted)); }
    if (ai.includes('vwap')) { let cumVP = 0, cumV = 0; const vd = sorted.map(d => { cumVP += ((d.high + d.low + d.close) / 3) * (d.volume || 1); cumV += (d.volume || 1); return { time: d.time, value: cumVP / cumV }; }); chart.addLineSeries({ color: '#ff9800', lineWidth: 1.5, crosshairMarkerVisible: false }).setData(vd); }
    if (ai.includes('macd_line')) { const e12 = computeEMA(sorted, 12), e26 = computeEMA(sorted, 26); const macd = []; for (let i = 0; i < e12.length; i++) { const m26 = e26.find(x => x.time === e12[i].time); if (m26) macd.push({ time: e12[i].time, value: e12[i].value - m26.value }); } chart.addLineSeries({ color: '#2196f3', lineWidth: 1.5, priceScaleId: 'macd', crosshairMarkerVisible: false }).setData(macd); }
    if (ai.includes('supertrend')) { const atrD = computeATR(sorted, 10); const st = []; atrD.forEach((a, i) => { const d = sorted.find(x => x.time === a.time); if (d) st.push({ time: a.time, value: (d.high + d.low) / 2 + 3 * a.value }); }); chart.addLineSeries({ color: '#4caf50', lineWidth: 1.5, crosshairMarkerVisible: false }).setData(st); }
    if (ai.includes('ichimoku_conv')) { chart.addLineSeries({ color: '#2962ff', lineWidth: 1, crosshairMarkerVisible: false }).setData(computeSMA(sorted, 9)); }
    if (ai.includes('ichimoku_base')) { chart.addLineSeries({ color: '#e91e63', lineWidth: 1, crosshairMarkerVisible: false }).setData(computeSMA(sorted, 26)); }
    if (ai.includes('kc')) { const atrD = computeATR(sorted, 20); const ema20 = computeEMA(sorted, 20); const ku = [], kl = []; ema20.forEach(e => { const a = atrD.find(x => x.time === e.time); if (a) { ku.push({ time: e.time, value: e.value + 1.5 * a.value }); kl.push({ time: e.time, value: e.value - 1.5 * a.value }); } }); chart.addLineSeries({ color: '#795548', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false }).setData(ku); chart.addLineSeries({ color: '#795548', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false }).setData(kl); }

    chart.timeScale().fitContent();
    return () => { try { chart.remove(); } catch {} };
  }, [sorted, activeIndicators, activeTF]);

  const handleTF = async (tf) => {
    setActiveTF(tf.label); setLoadingTF(true);
    try { await onLoadInterval(tf.interval, tf.period); } finally { setLoadingTF(false); }
  };

  const toggle = (id) => setActiveIndicators(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);
  const filtered = INDICATORS.filter(i => i.name.toLowerCase().includes(search.toLowerCase()) || i.cat.toLowerCase().includes(search.toLowerCase()));
  const grouped = {}; filtered.forEach(i => { if (!grouped[i.cat]) grouped[i.cat] = []; grouped[i.cat].push(i); });
  const last = sorted.length ? sorted[sorted.length - 1] : null;

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 9999, background: '#131722', display: 'flex', flexDirection: 'column', fontFamily: "'Inter', sans-serif" }}>
      {/* TOOLBAR */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px', background: '#1e222d', borderBottom: '1px solid #2a2e39', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 4, background: '#2a2e39' }}>
          <span style={{ color: '#fff', fontWeight: 700, fontSize: 14 }}>{symbol || data?.asset}</span>
        </div>
        <div style={{ display: 'flex', gap: 2, marginLeft: 8 }}>
          {TF.map(tf => (
            <button key={tf.label} onClick={() => handleTF(tf)} style={{ padding: '4px 10px', borderRadius: 4, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600, background: activeTF === tf.label ? '#2962ff' : 'transparent', color: activeTF === tf.label ? '#fff' : '#787b86', fontFamily: "'Inter', sans-serif" }}>{tf.label}</button>
          ))}
        </div>
        <div style={{ width: 1, height: 24, background: '#2a2e39', margin: '0 4px' }} />
        <button onClick={() => setShowPanel(!showPanel)} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 4, border: '1px solid #2a2e39', background: showPanel ? '#2962ff22' : 'transparent', color: showPanel ? '#2962ff' : '#787b86', cursor: 'pointer', fontSize: 13, fontWeight: 500, fontFamily: "'Inter', sans-serif" }}>
          <BarChart2 size={14}/> Indicators
        </button>
        {loadingTF && <span style={{ color: '#787b86', fontSize: 12, animation: 'pulse 1s infinite' }}>Loading…</span>}
        {last && (
          <div style={{ display: 'flex', gap: 12, marginLeft: 'auto', fontSize: 12, color: '#787b86' }}>
            <span>O <span style={{ color: '#e1e4ea' }}>{last.open}</span></span>
            <span>H <span style={{ color: '#e1e4ea' }}>{last.high}</span></span>
            <span>L <span style={{ color: '#e1e4ea' }}>{last.low}</span></span>
            <span>C <span style={{ color: last.close >= last.open ? '#26a69a' : '#ef5350' }}>{last.close}</span></span>
          </div>
        )}
        <button onClick={onClose} style={{ marginLeft: 12, padding: 6, borderRadius: 4, border: 'none', background: '#2a2e39', color: '#fff', cursor: 'pointer', display: 'flex' }}><X size={16}/></button>
      </div>
      {/* BODY */}
      <div style={{ flex: 1, display: 'flex', position: 'relative' }}>
        {showPanel && (
          <div style={{ width: 320, background: '#1e222d', borderRight: '1px solid #2a2e39', display: 'flex', flexDirection: 'column', overflow: 'hidden', flexShrink: 0 }}>
            <div style={{ padding: '14px 14px 10px', borderBottom: '1px solid #2a2e39' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ color: '#fff', fontWeight: 700, fontSize: 14 }}>Indicators, metrics, and strategies</span>
                <button onClick={() => setShowPanel(false)} style={{ background: 'none', border: 'none', color: '#787b86', cursor: 'pointer' }}><X size={14}/></button>
              </div>
              <div style={{ position: 'relative' }}>
                <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: '#787b86' }} />
                <input placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)} style={{ width: '100%', padding: '7px 10px 7px 30px', borderRadius: 6, border: '1px solid #2a2e39', background: '#131722', color: '#fff', fontSize: 12, fontFamily: "'Inter', sans-serif", outline: 'none', boxSizing: 'border-box' }} />
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '6px 0' }}>
              {Object.entries(grouped).map(([cat, items]) => (
                <div key={cat}>
                  <div style={{ padding: '6px 14px 2px', fontSize: 10, fontWeight: 700, color: '#787b86', textTransform: 'uppercase', letterSpacing: 1 }}>{cat}</div>
                  {items.map(ind => (
                    <button key={ind.id} onClick={() => toggle(ind.id)} style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '6px 14px', background: activeIndicators.includes(ind.id) ? '#2962ff15' : 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left', fontFamily: "'Inter', sans-serif", transition: 'background .15s' }}>
                      <div style={{ width: 4, height: 4, borderRadius: '50%', background: ind.color, flexShrink: 0 }} />
                      <span style={{ color: activeIndicators.includes(ind.id) ? '#2962ff' : '#e1e4ea', fontSize: 12, fontWeight: activeIndicators.includes(ind.id) ? 600 : 400 }}>{ind.name}</span>
                      {activeIndicators.includes(ind.id) && <span style={{ marginLeft: 'auto', fontSize: 10, color: '#2962ff', fontWeight: 600 }}>✓</span>}
                    </button>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ padding: '10px 14px', borderTop: '1px solid #2a2e39', display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {activeIndicators.map(id => {
                const ind = INDICATORS.find(i => i.id === id); if (!ind) return null;
                return <span key={id} style={{ display: 'inline-flex', alignItems: 'center', gap: 3, padding: '2px 6px', borderRadius: 3, background: '#2a2e39', fontSize: 10, color: ind.color, fontWeight: 600 }}>{ind.name}<span onClick={() => toggle(id)} style={{ cursor: 'pointer', color: '#787b86', marginLeft: 1 }}>×</span></span>;
              })}
            </div>
          </div>
        )}
        <div ref={containerRef} style={{ flex: 1 }} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 12px', background: '#1e222d', borderTop: '1px solid #2a2e39', fontSize: 11, color: '#787b86', flexShrink: 0 }}>
        <span>ASTRA SuperCharts • {symbol} • {activeTF}</span>
        <span>{new Date().toLocaleString()}</span>
      </div>
      <style>{`@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .4 } }`}</style>
    </div>
  );
}
