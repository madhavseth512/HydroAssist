/**
 * The shell talks to the API and says what happened.
 *
 * `fetch` is stubbed rather than the client, so these exercise the real request
 * path: credentials, JSON parsing, and the message a visitor sees when the
 * server is not there.
 */
import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { App } from './App'
import { renderWithProviders } from './test/render'
import { project, session, stubFetch } from './test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

describe('the workspace shell', () => {
  it('reports the open project once the session exists', async () => {
    stubFetch([
      { url: '/api/session', body: session() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<App />)

    expect(await screen.findByText('1 test · 0 saved analyses')).toBeInTheDocument()
    expect(screen.getByText('HydroAssist')).toBeInTheDocument()
  })

  it('sends the session cookie with every request', async () => {
    const stub = stubFetch([
      { url: '/api/session', body: session() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<App />)
    await screen.findByText('1 test · 0 saved analyses')

    expect(stub.mock).toHaveBeenCalledWith('/api/session',
      expect.objectContaining({ credentials: 'include' }))
  })

  it('explains a server it cannot reach', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('network') }))

    renderWithProviders(<App />)

    expect(await screen.findAllByText(/Cannot reach the server/)).not.toHaveLength(0)
  })

  it('shows the project panel and the data step together', async () => {
    stubFetch([
      { url: '/api/session', body: session() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<App />)

    expect(await screen.findByText('Untitled project')).toBeInTheDocument()
    expect(screen.getByText('Step 1 — Field data')).toBeInTheDocument()
  })
})
