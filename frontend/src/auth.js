/**
 * ASTRA Auth Utility
 * ====================
 * Single source of truth for:
 *   - Token storage (localStorage 'astra_token')
 *   - Authenticated fetch (auto-injects Authorization header)
 *   - Login / logout state
 *
 * Components should import `apiFetch` instead of plain fetch when hitting
 * protected endpoints. apiFetch auto-handles 401 by clearing the token and
 * firing a logout event.
 */

import { API_URL } from './config';

const TOKEN_KEY = 'astra_token';

// ── Token storage ────────────────────────────────────────────────────────

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
  // Notify listeners (App.jsx) so it can re-render
  window.dispatchEvent(new CustomEvent('astra:auth-changed'));
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new CustomEvent('astra:auth-changed'));
}

export function isAuthenticated() {
  return !!getToken();
}

// ── Authenticated fetch ──────────────────────────────────────────────────

/**
 * Wraps fetch() to auto-attach Authorization header and handle 401 → logout.
 *
 * Usage:
 *   import { apiFetch } from './auth';
 *   const res = await apiFetch('/api/positions');           // path-only
 *   const res = await apiFetch(`${API_URL}/api/positions`); // full URL also OK
 */
export async function apiFetch(input, init = {}) {
  const headers = new Headers(init.headers || {});
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  // Default Content-Type for non-form bodies if caller didn't set one
  if (init.body && !headers.has('Content-Type') && typeof init.body === 'string') {
    headers.set('Content-Type', 'application/json');
  }

  // Allow path-only input — auto-prefix API_URL
  const url = (typeof input === 'string' && input.startsWith('/'))
    ? `${API_URL}${input}`
    : input;

  const resp = await fetch(url, { ...init, headers });

  // 401 → token expired/invalid → force logout
  if (resp.status === 401 && getToken()) {
    clearToken();
  }
  return resp;
}

// ── Login / Register ─────────────────────────────────────────────────────

export async function login(username, password) {
  const body = new URLSearchParams({ username, password });
  const resp = await fetch(`${API_URL}/api/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  if (!resp.ok) {
    let detail = '';
    try { detail = (await resp.json()).detail; } catch (_) {}
    throw new Error(detail || (resp.status === 401 ? 'Incorrect email or password' : `Login failed (${resp.status})`));
  }
  const data = await resp.json();
  setToken(data.access_token);
  return data;
}

export async function register(username, password) {
  const resp = await fetch(`${API_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    let detail = '';
    try { detail = (await resp.json()).detail; } catch (_) {}
    throw new Error(detail || `Registration failed (${resp.status})`);
  }
  return resp.json();
}

// ── Global fetch patch ───────────────────────────────────────────────────
// Most views call fetch() directly. Attach the JWT to every same-origin API
// call (and to API_URL calls) so login works everywhere without touching each
// view; a 401 on an authenticated call logs the user out.
const API_PREFIXES = ['/api/', '/brokers', '/strategies', '/upstox'];

function isApiUrl(url) {
  try {
    const u = new URL(url, window.location.origin);
    const sameApiOrigin = u.origin === window.location.origin || (API_URL && url.startsWith(API_URL));
    return sameApiOrigin && API_PREFIXES.some(p => u.pathname.startsWith(p)) && !u.pathname.startsWith('/api/auth/');
  } catch (_) { return false; }
}

export function installAuthFetch() {
  if (window.__astraFetchPatched) return;
  window.__astraFetchPatched = true;
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input?.url;
    const token = getToken();
    if (token && url && isApiUrl(url)) {
      const headers = new Headers(init.headers || (typeof input !== 'string' ? input.headers : undefined) || {});
      if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
      init = { ...init, headers };
    }
    const resp = await nativeFetch(input, init);
    if (resp.status === 401 && token && url && isApiUrl(url)) clearToken();
    return resp;
  };
}

export function logout() {
  clearToken();
}
