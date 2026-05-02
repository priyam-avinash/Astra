import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = 'ws://localhost:8000/ws/prices';
const RECONNECT_DELAY_MS = 5000;

export default function useLivePrices() {
  const [prices, setPrices] = useState({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        setConnected(true);
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
  }, []); // no deps — connect is stable, uses ref for mounted check

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const subscribe = useCallback((symbol) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'subscribe', symbol }));
    }
  }, []);

  // isLive: WS is connected AND at least one symbol has source === 'dhan_ws'
  const isLive = connected && Object.values(prices).some(p => p.source === 'dhan_ws');

  return { prices, isLive, subscribe };
}
