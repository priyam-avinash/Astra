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
    const detail = await resp.text();
    throw new Error(`Login failed: ${detail || resp.statusText}`);
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
    const detail = await resp.text();
    throw new Error(`Registration failed: ${detail || resp.statusText}`);
  }
  return resp.json();
}

export function logout() {
  clearToken();
}
