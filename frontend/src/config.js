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
// In dev (npm run dev) the Vite server proxies /api, /ws, /brokers, /strategies,
// /upstox and /health to the backend, so the app calls its own origin: no CORS,
// works in every browser. Override with VITE_API_URL for other setups.
export const API_URL = stripTrail(
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? '' : 'http://localhost:8000')
);

// WebSocket origin — derived from API_URL by default (ws:// or wss://)
export const WS_URL = stripTrail(
  import.meta.env.VITE_WS_URL ||
  (API_URL ? API_URL.replace(/^http/, 'ws')
           : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`)
);

// Whether the app is running in production build (toggles dev tooling)
export const IS_PROD = import.meta.env.PROD === true;

// Build a full URL for a given API path: `apiUrl('/api/positions')` → `http://localhost:8000/api/positions`
export const apiUrl = (path) => {
  if (!path) return API_URL;
  return `${API_URL}${path.startsWith('/') ? '' : '/'}${path}`;
};
