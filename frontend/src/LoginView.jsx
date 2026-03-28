import React, { useState } from 'react';
import { Lock, Mail, ArrowRight, Loader } from 'lucide-react';

export default function LoginView({ onLogin, onSwitchToRegister }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username: email, password: password })
      });

      if (!response.ok) throw new Error('Invalid email or password');

      const data = await response.json();
      localStorage.setItem('token', data.access_token);
      onLogin(data.access_token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(circle at top left, hsla(222, 47%, 12%, 1), hsla(222, 47%, 4%, 1))' 
    }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '400px', padding: '40px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ 
            width: 48, height: 48, borderRadius: '12px', background: 'var(--color-primary)', 
            display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' 
          }}>
            <Lock color="white" size={24} />
          </div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '8px' }}>Welcome Back</h1>
          <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.9rem' }}>Enter your credentials to access the terminal.</p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>EMAIL ADDRESS</label>
            <div style={{ position: 'relative' }}>
              <Mail size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="email" 
                className="search-input" 
                placeholder="name@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{ width: '100%', paddingLeft: '40px', background: 'rgba(255,255,255,0.03)' }}
                required 
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>PASSWORD</label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="password" 
                className="search-input" 
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ width: '100%', paddingLeft: '40px', background: 'rgba(255,255,255,0.03)' }}
                required 
              />
            </div>
          </div>

          {error && <div style={{ color: '#ef4444', fontSize: '0.85rem', textAlign: 'center' }}>{error}</div>}

          <button 
            type="submit" 
            className="btn-primary" 
            disabled={loading}
            style={{ width: '100%', padding: '14px', marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          >
            {loading ? <Loader className="animate-spin" size={18} /> : <>Sign In <ArrowRight size={18} /></>}
          </button>
        </form>

        <div style={{ marginTop: '32px', textAlign: 'center', fontSize: '0.85rem', color: 'rgba(255,255,255,0.3)' }}>
          Don't have an account? {' '}
          <button 
            onClick={onSwitchToRegister}
            style={{ background: 'none', border: 'none', color: 'var(--color-primary)', fontWeight: 600, cursor: 'pointer', padding: 0 }}
          >
            Create one for free
          </button>
        </div>
      </div>
    </div>
  );
}
