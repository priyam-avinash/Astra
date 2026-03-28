import React, { useState } from 'react';
import { User, Mail, Lock, ArrowRight, Loader, CheckCircle } from 'lucide-react';

export default function RegisterView({ onRegister, onSwitchToLogin }) {
  const [formData, setFormData] = useState({ 
    username: '', 
    email: '', 
    password: '', 
    confirmPassword: '' 
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password
        })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Registration failed');
      }

      setSuccess(true);
      setTimeout(() => onSwitchToLogin(), 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div style={{ 
        minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'hsla(222, 47%, 4%, 1)'
      }}>
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px', maxWidth: '400px' }}>
          <CheckCircle size={64} color="#10b981" style={{ marginBottom: '24px' }} />
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '12px' }}>Registration Successful!</h2>
          <p style={{ color: 'rgba(255,255,255,0.4)' }}>Redirecting you to login...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ 
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(circle at top left, hsla(222, 47%, 12%, 1), hsla(222, 47%, 4%, 1))' 
    }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '440px', padding: '40px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '8px' }}>Create Account</h1>
          <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.9rem' }}>Join the community of elite AI traders.</p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>USERNAME</label>
            <div style={{ position: 'relative' }}>
              <User size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="text" 
                className="search-input" 
                placeholder="trader_sky"
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
                style={{ width: '100%', paddingLeft: '40px', background: 'rgba(255,255,255,0.03)' }}
                required 
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>EMAIL ADDRESS</label>
            <div style={{ position: 'relative' }}>
              <Mail size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="email" 
                className="search-input" 
                placeholder="name@company.com"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
                style={{ width: '100%', paddingLeft: '40px', background: 'rgba(255,255,255,0.03)' }}
                required 
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>PASSWORD</label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="password" 
                className="search-input" 
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                style={{ width: '100%', paddingLeft: '40px', background: 'rgba(255,255,255,0.03)' }}
                required 
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 600 }}>CONFIRM PASSWORD</label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.2)' }} />
              <input 
                type="password" 
                className="search-input" 
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
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
            {loading ? <Loader className="animate-spin" size={18} /> : <>Create Account <ArrowRight size={18} /></>}
          </button>
        </form>

        <div style={{ marginTop: '32px', textAlign: 'center', fontSize: '0.85rem', color: 'rgba(255,255,255,0.3)' }}>
          Already have an account? {' '}
          <button 
            onClick={onSwitchToLogin}
            style={{ background: 'none', border: 'none', color: 'var(--color-primary)', fontWeight: 600, cursor: 'pointer', padding: 0 }}
          >
            Sign in
          </button>
        </div>
      </div>
    </div>
  );
}
