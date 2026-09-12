/**
 * The shell talks to the API and says what happened.
 *
 * `fetch` is stubbed rather than mocking the client, so these tests exercise
 * the real request path: credentials, JSON parsing, and the error message a
 * visitor sees when the server is not there.
 */
import { MantineProvider } from '@mantine/core'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { App } from './App'

const SESSION = {
  session_id: 'test-session',
  project: {
    info: { title: 'Untitled project', client: '', project_number: '',
            location: '', test_date: '', analyst: '', notes: '' },
    tests: [],
    active_test_id: 'abc',
    filename: 'project.hydroassist.json',
    summary: '1 test · 0 saved analyses',
    analysis_count: 0,
    modified_at: '2026-09-12T00:00:00+00:00',
  },
  chat: {
    messages: [],
    context: { intent: 'unknown', aquifer: 'unknown', method: '',
               suggestion_confirmed: false, has_result: false, message_count: 0 },
    suggestion: null,
    retrieved: [],
  },
}

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <MantineProvider>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </MantineProvider>,
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the workspace shell', () => {
  it('reports the open project once the session exists', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify(SESSION), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })))

    renderApp()

    expect(await screen.findByText('1 test · 0 saved analyses')).toBeInTheDocument()
    expect(screen.getByText('HydroAssist')).toBeInTheDocument()
  })

  it('sends the session cookie with every request', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(SESSION), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }))
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    await screen.findByText('1 test · 0 saved analyses')

    expect(fetchMock).toHaveBeenCalledWith('/api/session',
      expect.objectContaining({ credentials: 'include' }))
  })

  it('explains a server it cannot reach', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('network') }))

    renderApp()

    expect(await screen.findByText(/Cannot reach the server/)).toBeInTheDocument()
  })

  it('shows what the server said when a request fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'That file is too large.' }), {
        status: 413,
        headers: { 'content-type': 'application/json' },
      })))

    renderApp()

    expect(await screen.findByText('That file is too large.')).toBeInTheDocument()
  })
})
