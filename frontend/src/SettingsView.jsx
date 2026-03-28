import React from 'react';
import { User, Shield, Bell, CreditCard, LogOut, ExternalLink } from 'lucide-react';

export default function SettingsView() {
  const profile = {
    name: "Alpha Trader",
    email: "alpha.trader@astra.ai",
    plan: "Pro SaaS Terminal",
    joined: "March 2024",
    status: "Active (Live)"
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="topbar" style={{ marginBottom: '24px' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Profile & Settings</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Manage your account, API keys, and notification preferences.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '24px' }}>
        {/* Profile Card */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div style={{ 
            width: '80px', height: '80px', borderRadius: '50%', background: 'var(--color-primary-gradient)', 
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' 
          }}>
            <User size={40} />
          </div>
          <div style={{ flex: 1 }}>
            <h3 style={{ margin: '0 0 4px 0', fontSize: '1.2rem' }}>{profile.name}</h3>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{profile.email}</p>
            <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
              <span style={{ 
                fontSize: '0.7rem', px: '8px', py: '2px', background: 'rgba(59, 130, 246, 0.1)', 
                color: '#3b82f6', borderRadius: '4px', fontWeight: 700, padding: '2px 8px'
              }}>{profile.plan}</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Joined {profile.joined}</span>
            </div>
          </div>
          <button className="btn-secondary" style={{ padding: '8px 16px' }}>Edit Profile</button>
        </div>

        {/* Settings Groups */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          
          <div className="glass-panel" style={{ padding: '20px' }}>
            <h4 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <Shield size={18} color="var(--color-primary)" /> Security
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem' }}>Two-Factor Authentication</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--color-success)', fontWeight: 700 }}>ENABLED</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem' }}>API Key Management</span>
                <button className="text-btn" style={{ fontSize: '0.75rem' }}>Update Keys <ExternalLink size={12} /></button>
              </div>
              <button className="btn-secondary" style={{ width: '100%', marginTop: '8px', fontSize: '0.8rem' }}>Change Password</button>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '20px' }}>
            <h4 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <Bell size={18} color="var(--color-primary)" /> Notifications
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem' }}>Trade Execution Alerts</span>
                <input type="checkbox" defaultChecked />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem' }}>AI Signal Alerts</span>
                <input type="checkbox" defaultChecked />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem' }}>Daily Market Summary</span>
                <input type="checkbox" />
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '20px' }}>
            <h4 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <CreditCard size={18} color="var(--color-primary)" /> Subscription
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <p style={{ margin: 0, fontSize: '0.85rem' }}>Current: <strong>Pro Yearly</strong></p>
              <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Next billing date: April 15, 2025</p>
              <button className="btn-primary" style={{ width: '100%', marginTop: '12px', fontSize: '0.8rem' }}>Manage Subscription</button>
            </div>
          </div>

        </div>

        <div className="glass-panel" style={{ padding: '20px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#ef4444', fontSize: '1rem' }}>Danger Zone</h4>
          <p style={{ margin: '0 0 16px 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Once you delete your account, there is no going back. Please be certain.</p>
          <button className="btn-secondary" style={{ color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.3)' }}>Delete Account</button>
        </div>
      </div>
    </div>
  );
}
