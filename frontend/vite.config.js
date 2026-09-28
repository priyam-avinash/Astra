import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies backend routes so the frontend talks to its own origin
// (no CORS, no cross-port blocking). Backend target overridable via ASTRA_BACKEND.
const backend = process.env.ASTRA_BACKEND || 'http://127.0.0.1:8000'
const proxied = ['/api', '/brokers', '/strategies', '/upstox', '/health']

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      ...Object.fromEntries(proxied.map(p => [p, { target: backend, changeOrigin: true }])),
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
})
