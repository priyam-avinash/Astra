import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Search, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Chart constants
// ─────────────────────────────────────────────────────────────────────────────
const W = 980, H_MAIN = 360, H_SUB = 90;
const PAD = { t: 12, r: 72, b: 28, l: 10 }; // Y-axis on RIGHT like Kite

const chartW = W - PAD.l - PAD.r;
const chartH = H_MAIN - PAD.t - PAD.b;

// ─────────────────────────────────────────────────────────────────────────────
// Scale helpers
// ─────────────────────────────────────────────────────────────────────────────
const sx = (i, n) => PAD.l + (i / (n - 1)) * chartW;
const sy = (v, lo, hi) => PAD.t + (1 - (v - lo) / (hi - lo)) * chartH;
const syS = (v, lo, hi, h) => (1 - (v - lo) / (hi - lo)) * (h - 20) + 10; // sub-panel

// ─────────────────────────────────────────────────────────────────────────────
// Indicator math
// ─────────────────────────────────────────────────────────────────────────────
const sma  = (arr, n) => arr.map((_, i) => i < n - 1 ? null : arr.slice(i-n+1,i+1).reduce((a,b)=>a+b)/n);
const ema  = (arr, n) => { const k=2/(n+1); let e=arr[0]; return arr.map((v,i)=>i===0?e:(e=v*k+e*(1-k))); };
const rsi14= (arr) => {
  if (arr.length < 15) return arr.map(() => null);
  const d = arr.map((v,i)=>i===0?0:v-arr[i-1]);
  let g=0,l=0; for(let i=1;i<=14;i++){g+=Math.max(d[i],0);l+=Math.max(-d[i],0);} g/=14;l/=14;
  const r = Array(15).fill(null);
  for(let i=15;i<arr.length;i++){g=(g*13+Math.max(d[i],0))/14;l=(l*13+Math.max(-d[i],0))/14;r.push(l===0?100:100-100/(1+g/l));}
  return r;
};
const macd = (arr) => {
  const f=ema(arr,12),s=ema(arr,26);
  const line=f.map((v,i)=>v-s[i]);
  const sig=[...Array(26).fill(null),...ema(line.slice(26),9)];
  return { line, sig, hist:line.map((v,i)=>sig[i]==null?null:v-sig[i]) };
};
const bb   = (arr, n=20) => {
  const mid=sma(arr,n);
  return { mid, up:mid.map((m,i)=>{if(!m)return null;const sl=arr.slice(i-n+1,i+1);return m+2*Math.sqrt(sl.reduce((s,v)=>s+(v-m)**2,0)/n);}),
           lo:mid.map((m,i)=>{if(!m)return null;const sl=arr.slice(i-n+1,i+1);return m-2*Math.sqrt(sl.reduce((s,v)=>s+(v-m)**2,0)/n);}) };
};

const atr = (candles, n=14) => {
  const tr = candles.map((c,i)=>{
    if(i===0) return c.h-c.l;
    return Math.max(c.h-c.l, Math.abs(c.h-candles[i-1].c), Math.abs(c.l-candles[i-1].c));
  });
  return sma(tr, n);
};

const stoch = (candles, n=14, d=3) => {
  const k = candles.map((_, i) => {
    if(i < n-1) return null;
    const slice = candles.slice(i-n+1, i+1);
    const low = Math.min(...slice.map(s=>s.l));
    const high = Math.max(...slice.map(s=>s.h));
    return ((candles[i].c - low) / (high - low)) * 100;
  });
  return { k, d: sma(k.filter(x=>x!=null), d) };
};

const vwap = (candles) => {
  let cumTPV = 0, cumVol = 0;
  return candles.map(c => {
    const tp = (c.h + c.l + c.c) / 3;
    cumTPV += tp * c.v;
    cumVol += c.v;
    return cumVol === 0 ? tp : cumTPV / cumVol;
  });
};

const supertrend = (candles, n=10, mult=3) => {
  const atrv = atr(candles, n);
  let up = [], dn = [], st = [], dir = 1;
  candles.forEach((c, i) => {
    if (i === 0 || !atrv[i]) { up.push(null); dn.push(null); st.push(null); return; }
    const basicUp = (c.h + c.l) / 2 + mult * atrv[i];
    const basicDn = (c.h + c.l) / 2 - mult * atrv[i];
    
    up.push(basicUp < up[i-1] || candles[i-1].c > up[i-1] ? basicUp : up[i-1]);
    dn.push(basicDn > dn[i-1] || candles[i-1].c < dn[i-1] ? basicDn : dn[i-1]);
    
    if (dir === 1) {
      if (candles[i].c < dn[i]) { dir = -1; st.push(up[i]); }
      else { st.push(dn[i]); }
    } else {
      if (candles[i].c > up[i]) { dir = 1; st.push(dn[i]); }
      else { st.push(up[i]); }
    }
  });
  return st;
};

// ─────────────────────────────────────────────────────────────────────────────
// SVG line helpers
// ─────────────────────────────────────────────────────────────────────────────
function Line({ series, lo, hi, color, sw=1.5, dash, subH }) {
  const segs = []; let cur = [];
  series.forEach((v,i) => {
    if (v!=null&&!isNaN(v)) { cur.push(`${sx(i,series.length)},${subH!=null?syS(v,lo,hi,subH):sy(v,lo,hi)}`); }
    else { if(cur.length>1) segs.push(cur.join(' ')); cur=[]; }
  });
  if (cur.length>1) segs.push(cur.join(' '));
  return <>{segs.map((p,k)=><polyline key={k} fill="none" stroke={color} strokeWidth={sw} strokeDasharray={dash?'4,3':undefined} points={p}/>)}</>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Right-side Y axis (Kite style)
// ─────────────────────────────────────────────────────────────────────────────
function RYAxis({ lo, hi, ltp, ltpColor }) {
  const ticks = Array.from({length:6},(_,i)=>lo+(i/5)*(hi-lo));
  return (
    <g>
      {ticks.map((v,i)=>{
        const y=sy(v,lo,hi);
        return <g key={i}>
          <line x1={PAD.l} x2={W-PAD.r} y1={y} y2={y} stroke="rgba(255,255,255,0.05)" strokeWidth={1}/>
          <text x={W-PAD.r+8} y={y+4} fontSize={10} fill="rgba(255,255,255,0.4)">{v>=1000?Math.round(v).toLocaleString('en-IN'):v.toFixed(1)}</text>
        </g>;
      })}
      {/* Live price label like Kite */}
      {ltp && <>
        <line x1={PAD.l} x2={W-PAD.r} y1={sy(ltp,lo,hi)} y2={sy(ltp,lo,hi)} stroke={ltpColor} strokeWidth={1} strokeDasharray="4,2" opacity={0.7}/>
        <rect x={W-PAD.r+6} y={sy(ltp,lo,hi)-9} width={58} height={18} rx={3} fill={ltpColor}/>
        <text x={W-PAD.r+10} y={sy(ltp,lo,hi)+4} fontSize={11} fill="#fff" fontWeight={700}>{ltp>=1000?Math.round(ltp).toLocaleString('en-IN'):ltp.toFixed(2)}</text>
      </>}
    </g>
  );
}

function XAxisLabels({ candles, total }) {
  const step = Math.max(1, Math.floor(total/8));
  return Array.from({length:Math.floor(total/step)+1},(_,i)=>{
    const idx=i*step; if(idx>=total) return null;
    const c = candles[idx];
    let label = `D-${total-1-idx}`;
    if (c?.t) {
      const d = new Date(c.t);
      const isIntraday = d.getHours() > 0 || d.getMinutes() > 0;
      label = isIntraday 
        ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
        : d.toLocaleDateString([], { day: '2-digit', month: 'short' });
    }
    return <text key={i} x={sx(idx,total)} y={H_MAIN-4} textAnchor="middle" fontSize={10} fill="rgba(255,255,255,0.3)">{label}</text>;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Indicator pill toggle
// ─────────────────────────────────────────────────────────────────────────────
function Pill({ label, color, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding:'3px 10px', borderRadius:14, border:'none', cursor:'pointer', fontSize:'0.72rem', fontWeight:700, transition:'all .15s',
      background: active?`${color}20`:'rgba(255,255,255,0.04)',
      color: active?color:'rgba(255,255,255,0.35)',
      outline: active?`1px solid ${color}50`:'1px solid transparent',
      display:'flex', alignItems:'center', gap:5
    }}>
      <span style={{width:6,height:6,borderRadius:'50%',background:active?color:'rgba(255,255,255,0.2)'}}/>
      {label}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
const SYMBOLS = {
  'Stocks':      ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','M&M.NS','WIPRO.NS','SBIN.NS','BHARTIARTL.NS','ICICIBANK.NS'],
  'F&O':         ['^NSEI','^NSEBANK','NIFTY_FIN_SERVICE.NS'],
  'Commodities': ['GC=F','CL=F','SI=F','HG=F'],
};
const TF_BARS = { 
  '1m':'1m', '5m':'5m', '15m':'15m', '1h':'1h', 
  '1D':1, '5D':5, '1M':22, '3M':66, '6M':132, '1Y':252 
};

// ─────────────────────────────────────────────────────────────────────────────
// Mapping for backend calls
const TF_MAP = {
  '1m':  { interval: '1m', period: '1d' },
  '5m':  { interval: '5m', period: '5d' },
  '15m': { interval: '15m', period: '1wk' },
  '1h':  { interval: '1h', period: '1mo' },
  '1D':  { interval: '1d', period: '6mo' }, // Default
  '5D':  { interval: '1d', period: '1y' },
  '1M':  { interval: '1d', period: '2y' },
  '3M':  { interval: '1d', period: '2y' },
  '6M':  { interval: '1d', period: '2y' },
  '1Y':  { interval: '1d', period: '5y' },
};

const C = { sma20:'#f59e0b', sma50:'#60a5fa', sma200:'#c084fc', ema12:'#34d399', ema26:'#fb923c' };
const SIG_C = s => s==='BUY'?'#22c55e':s==='SELL'?'#ef4444':'#f59e0b';

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function AnalysisView() {
  const [sym, setSym]       = useState('RELIANCE.NS');
  const [tab, setTab]       = useState('Stocks');
  const [tf, setTf]         = useState('1D'); // Default to daily now
  const [engine, setEngine] = useState('astra');
  const [loading, setLoad]  = useState(false);
  const [raw, setRaw]       = useState(null);
  const [error, setError]   = useState('');
  const [tooltip, setTip]   = useState(null);
  const [zoom, setZoom]     = useState(null); // number of bars, null = tf default
  const [ind, setInd] = useState({ 
    sma20:false, sma50:true, sma200:true, 
    ema12:false, ema26:false, 
    bb:false, rsi:true, macd:false, vol:true,
    vwap:false, atr:false, stoch:false, supertrend:false
  });
  const svgRef = useRef(null);
  const tog = k => setInd(p=>({...p,[k]:!p[k]}));

  const fetch_data = useCallback(async (s=sym, timeframe=tf) => {
    setLoad(true); setError(''); setTip(null);
    try {
      const token = localStorage.getItem('token');
      const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
      const { interval, period } = TF_MAP[timeframe] || TF_MAP['1D'];

      // 1. Start the analysis task with selected engine
      const r = await fetch(`http://localhost:8000/api/analyze/${s.toUpperCase()}?interval=${interval}&period=${period}&engine=${engine}`, { headers });
      if(!r.ok) throw new Error('Backend error — ensure the server is running and you are logged in');
      const startData = await r.json();
      
      if (startData.status === 'processing' && startData.task_id) {
        // 2. Poll for status
        let completed = false;
        let attempts = 0;
        const maxAttempts = 60; // 60 seconds timeout
        
        while (!completed && attempts < maxAttempts) {
          const sRes = await fetch(`http://localhost:8000/api/analyze/status/${startData.task_id}`, { headers });
          const sData = await sRes.json();
          
          if (sData.status === 'completed') {
            const d = sData.result;
            if(!d.ohlc?.length && !d.prices?.length) throw new Error('No price data for this symbol');
            setRaw(d); setZoom(null);
            completed = true;
          } else if (sData.status === 'failed') {
            throw new Error(sData.error || 'Analysis task failed');
          } else {
            // Still processing
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
          }
        }
        
        if (!completed) throw new Error('Analysis timed out. Please try again.');
      } else {
        // Fallback or unexpected response
        const d = startData.result || startData;
        if(!d.ohlc?.length && !d.prices?.length) throw new Error('No price data for this symbol');
        setRaw(d); setZoom(null);
      }
    } catch(e){ 
      console.error(e);
      setError(e.message); 
    }
    finally { setLoad(false); }
  }, [sym, engine, tf]);

  useEffect(()=>{ fetch_data(); }, [fetch_data]);

  // ── Build display OHLC slice based on timeframe + zoom ──────────────────────
  const candles = useMemo(()=>{
    if(!raw) return [];
    const src = raw.ohlc?.length
      ? raw.ohlc
      : (raw.prices||[]).map(c=>({o:c,h:c,l:c,c,v:0})); // fallback
    const bars = zoom ?? Math.min(TF_BARS[tf]??132, src.length);
    return src.slice(-Math.min(bars, src.length));
  },[raw,tf,zoom]);

  const closes = useMemo(()=>candles.map(c=>c.c),[candles]);
  const inds = useMemo(()=>{
    if(!closes.length) return null;
    return { 
      sma20:sma(closes,Math.min(20,closes.length)), 
      sma50:sma(closes,Math.min(50,closes.length)),
      sma200:sma(closes,Math.min(200,closes.length)), 
      ema12:ema(closes,12), 
      ema26:ema(closes,26),
      bb:bb(closes), 
      rsi:rsi14(closes), 
      vwap:vwap(candles),
      atr:atr(candles),
      stoch:stoch(candles),
      supertrend:supertrend(candles),
      ...macd(closes) 
    };
  },[closes, candles]);

  const allH = candles.length ? Math.max(...candles.map(c=>c.h)) * 1.012 : 1;
  const allL = candles.length ? Math.min(...candles.map(c=>c.l)) * 0.988 : 0;
  const ltp  = candles.length ? candles[candles.length-1].c : null;
  const ltpUp = candles.length && candles[candles.length-1].c >= candles[0].c;
  const ltpC  = ltpUp ? '#22c55e' : '#ef4444';
  const pct   = candles.length>1 ? ((candles[candles.length-1].c-candles[0].c)/candles[0].c*100).toFixed(2) : '0.00';

  const latest = candles.at(-1);
  const hover  = tooltip ? candles[tooltip.idx] : null;
  const disp   = hover || latest; // OHLC bar to show in header

  // ── Candlestick SVG renderer ─────────────────────────────────────────────────
  const renderCandles = () => {
    if(!candles.length) return null;
    const n = candles.length;
    const bw = Math.max(1, Math.min(12, chartW/n - 2));
    return candles.map((c,i)=>{
      const up = c.c >= c.o;
      const clr = up ? '#2CC17E' : '#E64646';
      const x = sx(i,n);
      const top    = sy(Math.max(c.o,c.c), allL, allH);
      const bot    = sy(Math.min(c.o,c.c), allL, allH);
      const bodyH  = Math.max(1, bot-top);
      const wickT  = sy(c.h, allL, allH);
      const wickB  = sy(c.l, allL, allH);
      return (
        <g key={i}>
          <line x1={x} x2={x} y1={wickT} y2={wickB} stroke={clr} strokeWidth={1}/>
          <rect x={x-bw/2} y={top} width={bw} height={bodyH}
            fill={up?'#2CC17E':'#E64646'} rx={bw>4?1:0}/>
        </g>
      );
    });
  };

  // ── Wheel / pinch zoom ────────────────────────────────────────────────────────
  const handleWheel = useCallback((e)=>{
    e.preventDefault();
    if(!raw) return;
    const total = (raw.ohlc||raw.prices||[]).length;
    const cur = zoom ?? Math.min(TF_BARS[tf]??132, total);
    const nxt = Math.round(cur * (e.deltaY>0?1.12:0.88));
    setZoom(Math.min(total, Math.max(5, nxt)));
  },[raw,zoom,tf]);

  useEffect(()=>{
    const el=svgRef.current; if(!el) return;
    el.addEventListener('wheel',handleWheel,{passive:false});
    return ()=>el.removeEventListener('wheel',handleWheel);
  },[handleWheel]);

  // Auto-refresh on engine change
  useEffect(() => {
    if (sym) fetch_data();
  }, [engine]);

  // ── Crosshair ────────────────────────────────────────────────────────────────
  const onMove = (e)=>{
    if(!candles.length||!svgRef.current) return;
    const rect=svgRef.current.getBoundingClientRect();
    const rx=(e.clientX-rect.left)*(W/rect.width);
    const idx=Math.min(candles.length-1,Math.max(0,Math.round((rx-PAD.l)/chartW*(candles.length-1))));
    setTip({idx, x:sx(idx,candles.length)});
  };

  // MACD range
  const macdVals = inds ? [...(inds.line||[]),...(inds.sig||[]),...(inds.hist||[])].filter(v=>v!=null&&!isNaN(v)) : [];
  const mMin = macdVals.length ? Math.min(...macdVals)*1.2 : -1;
  const mMax = macdVals.length ? Math.max(...macdVals)*1.2 : 1;

  const volMax = candles.length ? Math.max(...candles.map(c=>c.v||1)) : 1;

  const rsiVals = inds?.rsi?.filter(v=>v!=null) ?? [];
  const latestRsi = rsiVals.at(-1);

  return (
    <div className="animate-fade-in" style={{display:'flex',flexDirection:'column',gap:12}}>

      {/* ── Top bar: symbol + price ── */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',flexWrap:'wrap',gap:10}}>
        <div style={{display:'flex',flexDirection:'column',gap:4}}>
          <div style={{display:'flex',alignItems:'center',gap:12,flexWrap:'wrap'}}>
            <div style={{display:'flex',flexDirection:'column'}}>
               <h2 style={{margin:0,fontSize:'1.2rem',display:'flex',alignItems:'center',gap:8}}>
                {sym} <span style={{fontSize:'0.85rem',color:'rgba(255,255,255,0.4)',fontWeight:400}}>{raw?.company_name}</span>
               </h2>
            </div>
            {ltp && <>
              <span style={{fontSize:'1.5rem',fontWeight:700}}>{raw?.currency || (sym.endsWith('.NS') ? '₹' : '$')}{ltp.toLocaleString('en-IN')}</span>
              <span style={{color:ltpC,fontWeight:700,fontSize:'0.95rem'}}>{ltpUp?'▲':'▼'} {Math.abs(+pct)}%</span>
              {raw?.signal && (
                <span style={{padding:'2px 10px',borderRadius:12,fontSize:'0.75rem',fontWeight:700,background:`${SIG_C(raw.signal)}18`,color:SIG_C(raw.signal)}}>
                  {engine === 'astra' ? 'Astra 1.0' : engine === 'astra_ai' ? 'Astra.ai 1.0' : 'Astra.ml'}: {raw.signal} {raw.confidence}%
                </span>
              )}
            </>}
          </div>
          {/* OHLC header — like Kite */}
          {disp && (
            <div style={{display:'flex',gap:16,fontSize:'0.8rem',color:'rgba(255,255,255,0.5)'}}>
              <span>O <strong style={{color:'#fff'}}>{disp.o?.toFixed(2)}</strong></span>
              <span>H <strong style={{color:'#22c55e'}}>{disp.h?.toFixed(2)}</strong></span>
              <span>L <strong style={{color:'#ef4444'}}>{disp.l?.toFixed(2)}</strong></span>
              <span>C <strong style={{color:ltpC}}>{disp.c?.toFixed(2)}</strong></span>
              {disp.v>0 && <span>Vol <strong style={{color:'#94a3b8'}}>{(disp.v/1e6).toFixed(2)}M</strong></span>}
              {raw && <span>RSI <strong style={{color:latestRsi>70?'#ef4444':latestRsi<30?'#22c55e':'#f59e0b'}}>{latestRsi?.toFixed(1)??'—'}</strong></span>}
            </div>
          )}
        </div>

        {/* Controls */}
        <div style={{display:'flex',gap:12,alignItems:'center',flexWrap:'wrap'}}>
          {/* Engine Selector */}
          <div style={{display:'flex',background:'rgba(255,255,255,0.03)',borderRadius:10,padding:3,border:'1px solid rgba(255,255,255,0.08)'}}>
            <button onClick={()=>setEngine('astra')} style={{
              padding:'6px 14px', borderRadius:8, border:'none', cursor:'pointer', fontSize:'0.72rem', fontWeight:700, transition:'all .2s',
              background: engine === 'astra' ? 'rgba(255,255,255,0.08)' : 'transparent',
              color: engine === 'astra' ? '#60a5fa' : 'rgba(255,255,255,0.3)'
            }}>Astra 1.0 <span style={{fontSize:'0.6rem',opacity:0.6,marginLeft:4}}>SAFE</span></button>
            <button onClick={()=>setEngine('astra_ai')} style={{
              padding:'6px 14px', borderRadius:8, border:'none', cursor:'pointer', fontSize:'0.72rem', fontWeight:700, transition:'all .2s',
              background: engine === 'astra_ai' ? 'rgba(96,165,250,0.15)' : 'transparent',
              color: engine === 'astra_ai' ? '#c084fc' : 'rgba(255,255,255,0.3)'
            }}>Astra.ai 1.0 <span style={{fontSize:'0.6rem',opacity:0.6,marginLeft:4}}>AGGR</span></button>
            <button onClick={()=>setEngine('astra_ml')} style={{
              padding:'6px 14px', borderRadius:8, border:'none', cursor:'pointer', fontSize:'0.72rem', fontWeight:700, transition:'all .2s',
              background: engine === 'astra_ml' ? 'rgba(34,197,94,0.15)' : 'transparent',
              color: engine === 'astra_ml' ? '#2dd4bf' : 'rgba(255,255,255,0.3)'
            }}>Astra.ml <span style={{fontSize:'0.6rem',opacity:0.6,marginLeft:4}}>DEEP</span></button>
          </div>

          <div style={{display:'flex',background:'var(--panel-bg)',borderRadius:10,padding:4}}>
            {['Stocks','F&O','Commodities'].map(t=>(
              <button key={t} onClick={()=>{setTab(t);const s=SYMBOLS[t][0];setSym(s);fetch_data(s);}}
                style={{padding:'5px 12px',borderRadius:8,border:'none',fontWeight:600,cursor:'pointer',fontSize:'0.78rem',transition:'.2s',
                  background:tab===t?'hsla(210,100%,55%,.15)':'transparent',color:tab===t?'var(--color-primary)':'rgba(255,255,255,.38)'}}>{t}</button>
            ))}
          </div>
          <div style={{position:'relative'}}>
            <Search size={14} style={{position:'absolute',left:8,top:'50%',transform:'translateY(-50%)',color:'rgba(255,255,255,.3)'}}/>
            <input value={sym} onChange={e=>setSym(e.target.value)} onKeyDown={e=>e.key==='Enter'&&fetch_data()}
              placeholder="Symbol…" style={{background:'var(--panel-bg)',border:'1px solid rgba(255,255,255,.1)',borderRadius:9,padding:'6px 10px 6px 26px',color:'#fff',fontSize:'0.82rem',width:150,outline:'none'}}/>
          </div>
          <button className="btn-primary" onClick={()=>fetch_data()} disabled={loading} style={{padding:'7px 14px',fontSize:'0.82rem',display:'flex',alignItems:'center',gap:6}}>
            <RefreshCw size={13} style={{animation:loading?'spin 1s linear infinite':undefined}}/> {loading?'…':'Analyze'}
          </button>
        </div>
      </div>

      {/* ── Symbol chips + timeframe ── */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:8}}>
        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          {SYMBOLS[tab].map(s=>(
            <button key={s} onClick={()=>{setSym(s);fetch_data(s);}}
              style={{padding:'3px 11px',borderRadius:14,border:sym===s?'1px solid var(--color-primary)':'1px solid rgba(255,255,255,.1)',background:'transparent',color:sym===s?'var(--color-primary)':'rgba(255,255,255,.35)',fontSize:'0.75rem',cursor:'pointer',transition:'.15s'}}>
              {s}
            </button>
          ))}
        </div>

        <div style={{display:'flex',alignItems:'center',gap:8}}>
          {zoom && (
            <span style={{fontSize:'0.7rem',color:'#60a5fa',background:'hsla(210,100%,55%,.1)',padding:'3px 10px',borderRadius:8,display:'flex',alignItems:'center',gap:6}}>
              {zoom} bars
              <button onClick={()=>setZoom(null)} style={{background:'none',border:'none',color:'#60a5fa',cursor:'pointer',fontSize:'0.7rem',padding:0}}>✕</button>
            </span>
          )}
          <button onClick={()=>{if(!raw)return;const t=(raw.ohlc||raw.prices||[]).length;const c=zoom??Math.min(TF_BARS[tf]??132,t);setZoom(Math.min(t,Math.max(5,Math.round(c*1.25))));}} style={{background:'rgba(255,255,255,.05)',border:'1px solid rgba(255,255,255,.1)',color:'rgba(255,255,255,.5)',padding:'5px 8px',borderRadius:8,cursor:'pointer'}}><ZoomOut size={14}/></button>
          <button onClick={()=>{if(!raw)return;const t=(raw.ohlc||raw.prices||[]).length;const c=zoom??Math.min(TF_BARS[tf]??132,t);setZoom(Math.min(t,Math.max(5,Math.round(c*0.75))));}} style={{background:'rgba(255,255,255,.05)',border:'1px solid rgba(255,255,255,.1)',color:'rgba(255,255,255,.5)',padding:'5px 8px',borderRadius:8,cursor:'pointer'}}><ZoomIn size={14}/></button>
          <div style={{display:'flex',background:'var(--panel-bg)',borderRadius:10,padding:3}}>
            {Object.keys(TF_MAP).map(t=>(
              <button key={t} onClick={()=>{setTf(t);setZoom(null);fetch_data(sym, t);}}
                style={{padding:'4px 10px',borderRadius:8,border:'none',fontWeight:700,cursor:'pointer',fontSize:'0.74rem',transition:'.15s',
                  background:tf===t&&!zoom?'hsla(210,100%,55%,.2)':'transparent',color:tf===t&&!zoom?'var(--color-primary)':'rgba(255,255,255,.35)'}}>{t}</button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Indicator toolbar ── */}
      <div style={{display:'flex',gap:6,flexWrap:'wrap',padding:'7px 12px',background:'rgba(255,255,255,.02)',borderRadius:10,border:'1px solid rgba(255,255,255,.06)',alignItems:'center'}}>
        <span style={{fontSize:'0.66rem',color:'rgba(255,255,255,.25)',marginRight:2}}>ON CHART</span>
        <Pill label="SMA 20"  color={C.sma20}  active={ind.sma20}  onClick={()=>tog('sma20')}/>
        <Pill label="SMA 50"  color={C.sma50}  active={ind.sma50}  onClick={()=>tog('sma50')}/>
        <Pill label="SMA 200" color={C.sma200} active={ind.sma200} onClick={()=>tog('sma200')}/>
        <Pill label="EMA 12"  color={C.ema12}  active={ind.ema12}  onClick={()=>tog('ema12')}/>
        <Pill label="EMA 26"  color={C.ema26}  active={ind.ema26}  onClick={()=>tog('ema26')}/>
        <Pill label="VWAP"    color="#f472b6" active={ind.vwap}   onClick={()=>tog('vwap')}/>
        <Pill label="STR"     color="#10b981" active={ind.supertrend} onClick={()=>tog('supertrend')}/>
        <Pill label="BB (20)" color="#94a3b8"  active={ind.bb}     onClick={()=>tog('bb')}/>
        <span style={{width:1,background:'rgba(255,255,255,.07)',margin:'0 4px',alignSelf:'stretch'}}/>
        <span style={{fontSize:'0.66rem',color:'rgba(255,255,255,.25)',marginRight:2}}>PANELS</span>
        <Pill label="RSI"    color="#a78bfa" active={ind.rsi}  onClick={()=>tog('rsi')}/>
        <Pill label="MACD"   color="#38bdf8" active={ind.macd} onClick={()=>tog('macd')}/>
        <Pill label="ATR"    color="#fbbf24" active={ind.atr}  onClick={()=>tog('atr')}/>
        <Pill label="Stoch"  color="#2dd4bf" active={ind.stoch} onClick={()=>tog('stoch')}/>
        <Pill label="Volume" color="#64748b" active={ind.vol}  onClick={()=>tog('vol')}/>
      </div>

      {error && (
        <div style={{padding:'12px 16px',borderRadius:10,border:'1px solid rgba(239,68,68,.35)',background:'rgba(239,68,68,.06)',color:'#f87171',fontSize:'0.85rem'}}>⚠ {error}</div>
      )}

      {/* ── Main candlestick chart ── */}
      <div className="glass-panel" style={{padding:'12px 14px',position:'relative',userSelect:'none'}}>
        {loading && (
          <div style={{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:12,background:'rgba(8,10,18,.7)',zIndex:6,borderRadius:'inherit'}}>
            <div style={{width:34,height:34,borderRadius:'50%',border:'3px solid rgba(255,255,255,.08)',borderTopColor:'var(--color-primary)',animation:'spin 1s linear infinite'}}/>
            <p style={{color:'rgba(255,255,255,.4)',fontSize:'0.85rem'}}>Fetching {sym}…</p>
          </div>
        )}

        {/* Legend */}
        <div style={{display:'flex',gap:12,marginBottom:6,flexWrap:'wrap',fontSize:'0.68rem'}}>
          {ind.sma20  && <span style={{color:C.sma20 }}>─ SMA 20</span>}
          {ind.sma50  && <span style={{color:C.sma50 }}>─ SMA 50</span>}
          {ind.sma200 && <span style={{color:C.sma200}}>─ SMA 200</span>}
          {ind.ema12  && <span style={{color:C.ema12 }}>─ EMA 12</span>}
          {ind.ema26  && <span style={{color:C.ema26 }}>─ EMA 26</span>}
          {ind.bb     && <span style={{color:'#94a3b8'}}>- BB(20)</span>}
          <span style={{marginLeft:'auto',color:'rgba(255,255,255,.2)'}}>⌥ scroll/pinch to zoom</span>
        </div>

        <svg ref={svgRef} viewBox={`0 0 ${W} ${H_MAIN}`} width="100%" height={H_MAIN}
          onMouseMove={onMove} onMouseLeave={()=>setTip(null)} style={{cursor:'crosshair',display:'block'}}>

          {candles.length>1 && <RYAxis lo={allL} hi={allH} ltp={ltp} ltpColor={ltpC}/>}
          {candles.length>1 && <XAxisLabels total={candles.length} candles={candles}/>}

          {/* BB bands */}
          {ind.bb && inds && <>
            <Line series={inds.bb.up}  lo={allL} hi={allH} color="#64748b" sw={1} dash/>
            <Line series={inds.bb.lo}  lo={allL} hi={allH} color="#64748b" sw={1} dash/>
            <Line series={inds.bb.mid} lo={allL} hi={allH} color="#475569" sw={1}/>
          </>}

          {/* SMA/EMA/VWAP overlays */}
          {ind.sma20  && inds && <Line series={inds.sma20}  lo={allL} hi={allH} color={C.sma20}/>}
          {ind.sma50  && inds && <Line series={inds.sma50}  lo={allL} hi={allH} color={C.sma50}/>}
          {ind.sma200 && inds && <Line series={inds.sma200} lo={allL} hi={allH} color={C.sma200}/>}
          {ind.ema12  && inds && <Line series={inds.ema12}  lo={allL} hi={allH} color={C.ema12}/>}
          {ind.ema26  && inds && <Line series={inds.ema26}  lo={allL} hi={allH} color={C.ema26}/>}
          {ind.vwap   && inds && <Line series={inds.vwap}   lo={allL} hi={allH} color="#f472b6" sw={1}/>}
          {ind.supertrend && inds && <Line series={inds.supertrend} lo={allL} hi={allH} color="#10b981" sw={2}/>}

          {/* AI target / stoploss lines */}
          {raw?.target && <g>
            <line x1={PAD.l} x2={W-PAD.r} y1={sy(raw.target,allL,allH)} y2={sy(raw.target,allL,allH)} stroke="#22c55e" strokeWidth={1} strokeDasharray="6,3" opacity={0.5}/>
            <text x={PAD.l+4} y={sy(raw.target,allL,allH)-3} fontSize={9} fill="#22c55e" opacity={0.7}>TARGET {raw.target}</text>
          </g>}
          {raw?.stop_loss && <g>
            <line x1={PAD.l} x2={W-PAD.r} y1={sy(raw.stop_loss,allL,allH)} y2={sy(raw.stop_loss,allL,allH)} stroke="#ef4444" strokeWidth={1} strokeDasharray="6,3" opacity={0.5}/>
            <text x={PAD.l+4} y={sy(raw.stop_loss,allL,allH)-3} fontSize={9} fill="#ef4444" opacity={0.7}>STOP {raw.stop_loss}</text>
          </g>}

          {/* Candlesticks */}
          {renderCandles()}

          {/* Crosshair */}
          {tooltip && candles.length>1 && <>
            <line x1={tooltip.x} x2={tooltip.x} y1={PAD.t} y2={H_MAIN-PAD.b} stroke="rgba(255,255,255,.15)" strokeDasharray="4,3"/>
          </>}

          {/* Empty state */}
          {!candles.length && !loading && (
            <text x={W/2} y={H_MAIN/2} textAnchor="middle" fontSize={14} fill="rgba(255,255,255,.2)">Select a symbol and click Analyze</text>
          )}
        </svg>
      </div>

      {/* ── RSI sub-panel ── */}
      {ind.rsi && inds && (()=>{
        const rv=inds.rsi, lo=0, hi=100;
        return (
          <div className="glass-panel" style={{padding:'8px 14px'}}>
            <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
              <span style={{fontSize:'0.7rem',color:'#a78bfa',fontWeight:700}}>RSI (14)</span>
              <span style={{fontSize:'0.68rem',color:'rgba(255,255,255,.3)'}}>
                {tooltip?`${rv[tooltip.idx]?.toFixed(1)??'—'}`:`${latestRsi?.toFixed(1)??'—'}`} · OB&gt;70 OS&lt;30
              </span>
            </div>
            <svg viewBox={`0 0 ${W} ${H_SUB}`} width="100%" height={H_SUB} style={{display:'block'}}>
              <rect x={PAD.l} y={syS(70,lo,hi,H_SUB)} width={W-PAD.l-PAD.r} height={syS(30,lo,hi,H_SUB)-syS(70,lo,hi,H_SUB)} fill="rgba(167,139,250,0.04)"/>
              {[70,50,30].map(v=><g key={v}>
                <line x1={PAD.l} x2={W-PAD.r} y1={syS(v,lo,hi,H_SUB)} y2={syS(v,lo,hi,H_SUB)} stroke={v===50?'rgba(255,255,255,.05)':v===70?'rgba(239,68,68,.25)':'rgba(34,197,94,.25)'} strokeDasharray={v!==50?'4,3':undefined}/>
                <text x={W-PAD.r+8} y={syS(v,lo,hi,H_SUB)+4} fontSize={9} fill="rgba(255,255,255,.3)">{v}</text>
              </g>)}
              <Line series={rv} lo={lo} hi={hi} color="#a78bfa" sw={1.8} subH={H_SUB}/>
              {tooltip && rv[tooltip.idx]!=null && <circle cx={sx(tooltip.idx,rv.length)} cy={syS(rv[tooltip.idx],lo,hi,H_SUB)} r={3} fill="#a78bfa"/>}
            </svg>
          </div>
        );
      })()}

      {/* ── MACD sub-panel ── */}
      {ind.macd && inds && macdVals.length>0 && (()=>(
        <div className="glass-panel" style={{padding:'8px 14px'}}>
          <div style={{display:'flex',gap:10,marginBottom:4}}>
            <span style={{fontSize:'0.7rem',color:'#38bdf8',fontWeight:700}}>MACD (12,26,9)</span>
            <span style={{fontSize:'0.68rem',color:'#38bdf8'}}>─ MACD</span>
            <span style={{fontSize:'0.68rem',color:'#fb923c'}}>─ Signal</span>
            <span style={{fontSize:'0.68rem',color:'rgba(255,255,255,.3)'}}>█ Hist</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H_SUB}`} width="100%" height={H_SUB} style={{display:'block'}}>
            <line x1={PAD.l} x2={W-PAD.r} y1={syS(0,mMin,mMax,H_SUB)} y2={syS(0,mMin,mMax,H_SUB)} stroke="rgba(255,255,255,.1)"/>
            {inds.hist.map((v,i)=>{if(v==null||isNaN(v))return null;const x=sx(i,inds.hist.length);const z=syS(0,mMin,mMax,H_SUB);const y=syS(v,mMin,mMax,H_SUB);const bw=Math.max(1,(W-PAD.l-PAD.r)/inds.hist.length-1);return <rect key={i} x={x-bw/2} y={Math.min(y,z)} width={bw} height={Math.abs(z-y)} fill={v>=0?'rgba(34,197,94,.45)':'rgba(239,68,68,.45)'}/>;})}
            <Line series={inds.line} lo={mMin} hi={mMax} color="#38bdf8" sw={1.5} subH={H_SUB}/>
            <Line series={inds.sig}  lo={mMin} hi={mMax} color="#fb923c" sw={1.5} subH={H_SUB}/>
          </svg>
        </div>
      ))()}

      {/* ── ATR sub-panel ── */}
      {ind.atr && inds && inds.atr && (()=>(
        <div className="glass-panel" style={{padding:'8px 14px'}}>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
            <span style={{fontSize:'0.7rem',color:'#fbbf24',fontWeight:700}}>ATR (14)</span>
            <span style={{fontSize:'0.68rem',color:'rgba(255,255,255,.3)'}}>
              {tooltip?`${inds.atr[tooltip.idx]?.toFixed(2)??'—'}`:`${inds.atr.at(-1)?.toFixed(2)??'—'}`}
            </span>
          </div>
          <svg viewBox={`0 0 ${W} ${H_SUB}`} width="100%" height={H_SUB} style={{display:'block'}}>
            <Line series={inds.atr} lo={Math.min(...inds.atr.filter(x=>x!=null))*0.9} hi={Math.max(...inds.atr.filter(x=>x!=null))*1.1} color="#fbbf24" sw={1.5} subH={H_SUB}/>
          </svg>
        </div>
      ))()}

      {/* ── Stochastic sub-panel ── */}
      {ind.stoch && inds && inds.stoch && (()=>(
        <div className="glass-panel" style={{padding:'8px 14px'}}>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
            <span style={{fontSize:'0.7rem',color:'#2dd4bf',fontWeight:700}}>Stochastic (14, 3)</span>
            <span style={{fontSize:'0.68rem',color:'rgba(255,255,255,.3)'}}>
              %K: {tooltip?inds.stoch.k[tooltip.idx]?.toFixed(1):inds.stoch.k.at(-1)?.toFixed(1)} 
              %D: {tooltip?inds.stoch.d[tooltip.idx]?.toFixed(1):inds.stoch.d.at(-1)?.toFixed(1)}
            </span>
          </div>
          <svg viewBox={`0 0 ${W} ${H_SUB}`} width="100%" height={H_SUB} style={{display:'block'}}>
            <rect x={PAD.l} y={syS(80,0,100,H_SUB)} width={W-PAD.l-PAD.r} height={syS(20,0,100,H_SUB)-syS(80,0,100,H_SUB)} fill="rgba(45,212,191,0.04)"/>
            <Line series={inds.stoch.k} lo={0} hi={100} color="#2dd4bf" sw={1.5} subH={H_SUB}/>
            <Line series={inds.stoch.d} lo={0} hi={100} color="#fb7185" sw={1.5} subH={H_SUB} dash/>
          </svg>
        </div>
      ))()}

      {/* ── Volume sub-panel ── */}
      {ind.vol && candles.length>1 && (()=>(
        <div className="glass-panel" style={{padding:'8px 14px'}}>
          <span style={{fontSize:'0.7rem',color:'rgba(255,255,255,.35)',fontWeight:700}}>VOLUME</span>
          <svg viewBox={`0 0 ${W} ${H_SUB}`} width="100%" height={H_SUB} style={{display:'block',marginTop:4}}>
            {candles.map((c,i)=>{
              const x=sx(i,candles.length);const bw=Math.max(1,(W-PAD.l-PAD.r)/candles.length-1);
              const bh=((c.v||0)/volMax)*(H_SUB-20);const up=c.c>=c.o;
              return <rect key={i} x={x-bw/2} y={H_SUB-10-bh} width={bw} height={Math.max(1,bh)} fill={up?'rgba(34,197,94,.45)':'rgba(239,68,68,.45)'}/>;
            })}
          </svg>
        </div>
      ))()}

      {/* ── Stats strip ── */}
      {raw && (
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(120px,1fr))',gap:10}}>
          {[
            {l:'RSI (14)', v:raw.rsi, badge:raw.rsi>70?'Overbought':raw.rsi<30?'Oversold':'Neutral', bc:raw.rsi>70?'#ef4444':raw.rsi<30?'#22c55e':'#f59e0b'},
            {l:'MACD', v:raw.macd},
            {l:'Suggested Entry', v:raw.entry_price, u:raw.currency||'₹', c:'var(--color-primary)'},
            {l:'AI Target', v:raw.target, u:raw.currency||'₹', c:'#22c55e'},
            {l:'Stop Loss', v:raw.stop_loss, u:raw.currency||'₹', c:'#ef4444'},
            {l:'Confidence', v:raw.confidence, u:'%'},
          ].map(({l,v,u='',badge,bc,c})=>(
            <div key={l} className="glass-panel" style={{padding:'11px 14px'}}>
              <div style={{fontSize:'0.66rem',color:'rgba(255,255,255,.35)',marginBottom:5}}>{l}</div>
              <div style={{fontSize:'1rem',fontWeight:700,color:c||'#fff'}}>{v!=null?`${u === '₹' || u === '$' ? u : ''}${typeof v==='number'?v.toLocaleString('en-IN'):v}${u !== '₹' && u !== '$' ? u : ''}`:'—'}</div>
              {badge && <div style={{fontSize:'0.64rem',color:bc,marginTop:3,fontWeight:600}}>{badge}</div>}
            </div>
          ))}
        </div>
      )}

      {/* ── Signals Overview Table ── */}
      <SignalsOverview tab={tab} fetch_data={fetch_data} currentSym={sym} setSym={setSym} engine={engine}/>
    </div>
  );
}

function SignalsOverview({ tab, fetch_data, currentSym, setSym, engine }) {
  const [data, setData] = useState([]);
  const [loading, setLoad] = useState(false);

  useEffect(() => {
    const fetchBulk = async () => {
      setLoad(true);
      try {
        const token = localStorage.getItem('token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        const symbols = SYMBOLS[tab].join(',');
        const r = await fetch(`http://localhost:8000/api/analyze/bulk?symbols=${symbols}&engine=${engine}`, { headers });
        const d = await r.json();
        setData(d.results || []);
      } catch (e) {
        console.error("Bulk fetch failed", e);
      } finally {
        setLoad(false);
      }
    };
    fetchBulk();
  }, [tab, engine]);

  if (!data.length && !loading) return null;

  return (
    <div className="glass-panel" style={{padding:'16px',marginTop:12}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <h3 style={{margin:0,fontSize:'0.95rem',color:'rgba(255,255,255,0.7)'}}>Market Signals Overview: {tab}</h3>
        {loading && <div style={{width:16,height:16,borderRadius:'50%',border:'2px solid rgba(255,255,255,.1)',borderTopColor:'var(--color-primary)',animation:'spin 1s linear infinite'}}/>}
      </div>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',fontSize:'0.82rem',color:'rgba(255,255,255,0.8)'}}>
          <thead>
            <tr style={{borderBottom:'1px solid rgba(255,255,255,0.05)',textAlign:'left'}}>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>SYMBOL</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>SIGNAL</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>PRICE</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>ENTRY</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>TARGET</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>STOP</th>
              <th style={{padding:'10px 8px',color:'rgba(255,255,255,0.4)',fontWeight:500}}>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {data.map(s => (
              <tr key={s.symbol} style={{
                borderBottom:'1px solid rgba(255,255,255,0.03)',
                background: currentSym === s.symbol ? 'rgba(96,165,250,0.05)' : 'transparent',
                transition: '0.2s'
              }}>
                <td style={{padding:'12px 8px'}}>
                  <div style={{fontWeight:700,color:'#fff'}}>{s.symbol}</div>
                  <div style={{fontSize:'0.7rem',color:'rgba(255,255,255,0.4)'}}>{s.name}</div>
                </td>
                <td style={{padding:'12px 8px'}}>
                  <span style={{
                    padding:'2px 8px', borderRadius:6, fontSize:'0.75rem', fontWeight:700,
                    background: `${SIG_C(s.signal)}15`, color: SIG_C(s.signal)
                  }}>
                    {s.signal} {s.confidence}%
                  </span>
                </td>
                <td style={{padding:'12px 8px',fontWeight:600}}>{s.currency}{s.price.toLocaleString('en-IN')}</td>
                <td style={{padding:'12px 8px',color:'var(--color-primary)'}}>{s.currency}{s.entry.toLocaleString('en-IN')}</td>
                <td style={{padding:'12px 8px',color:'#22c55e'}}>{s.currency}{s.target.toLocaleString('en-IN')}</td>
                <td style={{padding:'12px 8px',color:'#ef4444'}}>{s.currency}{s.stop.toLocaleString('en-IN')}</td>
                <td style={{padding:'12px 8px'}}>
                  <button 
                    onClick={() => { setSym(s.symbol); fetch_data(s.symbol); window.scrollTo({top:0, behavior:'smooth'}); }}
                    style={{padding:'4px 10px',borderRadius:6,border:'1px solid rgba(255,255,255,0.1)',background:'rgba(255,255,255,0.05)',color:'#fff',fontSize:'0.75rem',cursor:'pointer'}}
                  >
                    View Chart
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
