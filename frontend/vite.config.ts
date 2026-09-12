/// <reference types="vitest" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the SPA runs on Vite's server and the API on uvicorn, so /api
// is proxied: the browser sees one origin here exactly as it does in
// production, where FastAPI serves this build itself. That keeps the session
// cookie working the same way in both.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    // The free-tier host serves these files; keep the payload honest.
    chunkSizeWarningLimit: 1200,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: false,
  },
})
