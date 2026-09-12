/// <reference types="vitest" />
import { fileURLToPath } from 'node:url'

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
    // happy-dom, not jsdom. Measured on this project: a trivial Mantine render
    // took 4 s under jsdom and 63 ms under happy-dom, with the difference spent
    // outside the test body entirely (render itself was 99 ms either way). The
    // whole file went from 29 s to 0.1 s.
    environment: 'happy-dom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: false,
    alias: {
      // Plotly is megabytes of charting code. A panel test is about what the
      // panel hands to the chart, not about Plotly's own rendering.
      'react-plotly.js/factory': fileURLToPath(
        new URL('./src/test/plotlyStub.tsx', import.meta.url)),
    },
  },
})
