/**
 * ASTRA Frontend Configuration
 * =============================
 * All environment-driven values centralized here.
 * Set in:
 *   - frontend/.env.development   (defaults to http://localhost:8000 for local dev)
 *   - frontend/.env.production    (set to your real API origin at deploy time)
 *
 * Vite exposes VITE_* prefixed vars at build time via import.meta.env.
 */

// Strip trailing slashes so concatenation is consistent
const stripTrail = (s) => (s || '').replace(/\/+$/, '');

// Public API origin (the FastAPI backend). Default localhost for dev.
export const API_URL = stripTrail(
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
);

// WebSocket origin — derived from API_URL by default (ws:// or wss://)
export const WS_URL = stripTrail(
  import.meta.env.VITE_WS_URL ||
  API_URL.replace(/^http/, 'ws')
);

// Whether the app is running in production build (toggles dev tooling)
export const IS_PROD = import.meta.env.PROD === true;

// Build a full URL for a given API path: `apiUrl('/api/positions')` → `http://localhost:8000/api/positions`
export const apiUrl = (path) => {
  if (!path) return API_URL;
  return `${API_URL}${path.startsWith('/') ? '' : '/'}${path}`;
};
