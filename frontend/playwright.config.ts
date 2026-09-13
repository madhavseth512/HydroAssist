/**
 * Browser end-to-end tests: the built SPA against the real API, in Chromium.
 *
 * Everything is real except the consultant, which scripts/e2e_server.py
 * switches off so a run spends no model quota. Build first:
 *
 *   npm run build && npm run e2e
 */
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig, devices } from '@playwright/test'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const PYTHON = process.platform === 'win32'
  ? join(ROOT, '.venv', 'Scripts', 'python.exe')
  : join(ROOT, '.venv', 'bin', 'python')
const PORT = 8790

export default defineConfig({
  testDir: 'e2e',
  // A fit and a PDF are real work on the server; allow for a cold start.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `"${PYTHON}" "${join(ROOT, 'scripts', 'e2e_server.py')}" --port ${PORT}`,
    cwd: ROOT,
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 180_000,
    reuseExistingServer: false,
    stdout: 'ignore',
    stderr: 'pipe',
  },
})
