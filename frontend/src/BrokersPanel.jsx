/**
 * Brokers Panel — used inside Settings.
 * Shows all registered broker plugins, their connection status, and a "Connect"
 * button that opens the OAuth flow when available.
 *
 * Driven by the /brokers and /brokers/{name}/login-url endpoints.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { Server, CheckCircle, XCircle, ExternalLink, RefreshCw } from 'lucide-react';
import { apiFetch } from './auth';


export default function BrokersPanel() {
  const [brokers, setBrokers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const r = await apiFetch('/brokers');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setBrokers(data.brokers || []);
    } catch (e) {
      setError(e.message || 'Failed to load brokers');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const openLogin = useCallback(async (name) => {
    try {
      const r = await apiFetch(`/brokers/${encodeURIComponent(name)}/login-url`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      if (!data.login_url) {
        alert(
          `${name} does not have an OAuth flow yet, or requires credentials in your ` +
          `.env first (e.g. KITE_API_KEY for Zerodha). Check the broker plugin docs.`
        );
        return;
      }
      // Open in a new tab/window
      window.open(data.login_url, '_blank', 'noopener,noreferrer');
    } catch (e) {
      alert(`Failed to start ${name} login: ${e.message}`);
    }
  }, []);

  return (
    <div style={{
      background: '#1a1d2e', border: '1px solid #2a2e39',
      borderRadius: 12, padding: '20px 24px',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16,
      }}>
        <div style={{
          fontSize: 13, fontWeight: 700, color: '#9ca3af',
          letterSpacing: '0.08em', textTransform: 'uppercase',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <Server size={14} /> Broker Connections
        </div>
        <button onClick={load} style={{
          background: 'transparent', border: '1px solid #2a2e39',
          borderRadius: 6, padding: '4px 8px', cursor: 'pointer',
          color: '#9ca3af', fontSize: 11,
          display: 'inline-flex', alignItems: 'center', gap: 4,
        }}>
          <RefreshCw size={11} /> Refresh
        </button>
      </div>

      {error && (
        <div style={{ fontSize: 13, color: '#ef5350', marginBottom: 12 }}>
          Error: {error}
        </div>
      )}

      {loading && (
        <div style={{ fontSize: 13, color: '#6b7280', padding: '12px 0' }}>
          Loading…
        </div>
      )}

      {!loading && brokers.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {brokers.map(b => (
            <div key={b.name} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 14px',
              background: '#0f1118',
              border: '1px solid #2a2e39',
              borderRadius: 8,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {b.available ? (
                  <CheckCircle size={16} color="#26a69a" />
                ) : (
                  <XCircle size={16} color="#6b7280" />
                )}
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#e1e4ea' }}>
                    {b.name}
                  </div>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    priority {b.priority} ·{' '}
                    {b.available ? (
                      <span style={{ color: '#26a69a' }}>connected</span>
                    ) : (
                      <span>not connected</span>
                    )}
                  </div>
                </div>
              </div>

              <button
                onClick={() => openLogin(b.name)}
                style={{
                  padding: '6px 12px', borderRadius: 6, cursor: 'pointer',
                  background: b.available ? '#2a2e39' : '#2962ff',
                  color: b.available ? '#9ca3af' : '#fff',
                  border: 'none', fontSize: 12, fontWeight: 600,
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                }}>
                <ExternalLink size={11} />
                {b.available ? 'Reconnect' : 'Connect'}
              </button>
            </div>
          ))}
        </div>
      )}

      <div style={{ fontSize: 11, color: '#6b7280', marginTop: 12, lineHeight: 1.5 }}>
        SEBI requires daily broker token refresh. Click "Reconnect" each trading
        morning to refresh tokens via OAuth. Higher-priority brokers (lower number)
        are tried first for data fetches.
      </div>
    </div>
  );
}
