import React from 'react';

export default function HistoryView({ history }) {
  return (
    <div className="animate-fade-in">
      <div className="topbar">
        <div>
          <h2>Trade History</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
            Immutable record of all AI-generated and manual executions.
          </p>
        </div>
        <button className="btn-primary" style={{ background: 'transparent', border: '1px solid var(--panel-border)' }}>
          Export CSV
        </button>
      </div>

      <div className="glass-panel" style={{ marginTop: '24px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'hsla(222, 47%, 8%, 0.5)', color: 'var(--text-secondary)' }}>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Execution Time</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Trade ID</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Asset</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Action</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Price</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Qty</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Net P&L</th>
              <th style={{ padding: '16px 24px', fontWeight: 500 }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {history.map((record) => (
              <tr key={record.id} style={{ borderBottom: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '16px 24px', color: 'var(--text-secondary)' }}>{record.time}</td>
                <td style={{ padding: '16px 24px', fontFamily: 'monospace' }}>{record.id}</td>
                <td style={{ padding: '16px 24px', fontWeight: 600 }}>{record.asset}</td>
                <td style={{ padding: '16px 24px' }}>
                  <span style={{ color: record.action === 'BUY' ? 'var(--color-success)' : 'var(--color-danger)', fontWeight: 600 }}>
                    {record.action}
                  </span>
                </td>
                <td style={{ padding: '16px 24px' }}>₹{record.price}</td>
                <td style={{ padding: '16px 24px' }}>{record.qty}</td>
                <td style={{ padding: '16px 24px', fontWeight: 700, color: record.pnl > 0 ? 'var(--color-success)' : 'var(--color-danger)' }}>
                  {record.pnl > 0 ? '+' : ''}₹{record.pnl}
                </td>
                <td style={{ padding: '16px 24px' }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--color-success)' }} />
                    {record.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
