import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_URL as WS_ORIGIN, API_URL } from '../config';

const WS_URL = `${WS_ORIGIN}/ws/prices`;
const RECONNECT_DELAY_MS = 5000;
const POLL_MS = 15000;
// Serverless hosts (Vercel) can't hold WebSockets: poll /api/quote instead.
// Also falls back to polling if the socket never opens after 2 tries.
const WS_DISABLED = import.meta.env.VITE_DISABLE_WS === '1';
const MAX_FAILED_CONNECTS = 2;

export default function useLivePrices() {
  const [prices, setPrices] = useState({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const mountedRef = useRef(true);
  const symbolsRef = useRef(new Set());
  const failedRef = useRef(0);
  const pollTimerRef = useRef(null);
  const pollingRef = useRef(false);

  const pollOnce = useCallback(async () => {
    const syms = [...symbolsRef.current];
    await Promise.all(syms.map(async (sym) => {
      try {
        const r = await fetch(`${API_URL}/api/quote/${encodeURIComponent(sym)}`);
        if (!r.ok) return;
        const q = await r.json();
        if (!mountedRef.current || !(q.price > 0)) return;
        setPrices(prev => ({ ...prev, [sym]: { price: q.price, source: q.source, ts: Date.now() / 1000, stale: q.stale } }));
      } catch (_) {}
    }));
  }, []);

  const startPolling = useCallback(() => {
    if (pollingRef.current || !mountedRef.current) return;
    pollingRef.current = true;
    const tick = async () => {
      if (!mountedRef.current) return;
      await pollOnce();
      pollTimerRef.current = setTimeout(tick, POLL_MS);
    };
    tick();
  }, [pollOnce]);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (WS_DISABLED || failedRef.current >= MAX_FAILED_CONNECTS) { startPolling(); return; }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        failedRef.current = -1000;   // socket works: never fall back to polling
        setConnected(true);
        symbolsRef.current.forEach(sym => ws.send(JSON.stringify({ action: 'subscribe', symbol: sym })));
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'snapshot') {
            // msg.prices: { [symbol]: { price, source, ts } }
            setPrices(msg.prices || {});
          } else if (msg.type === 'tick') {
            // msg: { symbol, price, source, ts }
            setPrices(prev => ({
              ...prev,
              [msg.symbol]: { price: msg.price, source: msg.source, ts: msg.ts },
            }));
          }
        } catch (_) {}
      };

      ws.onerror = () => {
        // onclose will fire after onerror, handle reconnect there
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnected(false);
        wsRef.current = null;
        failedRef.current += 1;
        // Schedule reconnect
        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, RECONNECT_DELAY_MS);
      };
    } catch (_) {
      // WebSocket constructor threw (e.g. invalid URL) — retry after delay
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, RECONNECT_DELAY_MS);
    }
  }, [startPolling]); // stable — uses refs for mounted/failed state

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      clearTimeout(pollTimerRef.current);
      pollingRef.current = false;
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const subscribe = useCallback((symbol) => {
    if (!symbol) return;
    const isNew = !symbolsRef.current.has(symbol);
    symbolsRef.current.add(symbol);
    if (pollingRef.current && isNew) pollOnce();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'subscribe', symbol }));
    }
  }, [pollOnce]);

  // isLive: WS is connected AND at least one symbol has source === 'dhan_ws'
  const isLive = connected && Object.values(prices).some(p => p.source === 'dhan_ws');

  return { prices, isLive, subscribe };
}
