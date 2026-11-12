/** Steps 4-5: the numbers, the caveats, saving, the report and manual matching. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ResultsPanel } from './ResultsPanel'
import { renderWithProviders } from '../test/render'
import { chatState, stubFetch } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

function parameter(overrides: Record<string, unknown> = {}) {
  return {
    name: 'T', symbol: 'T', value: 2.49e-3, unit: 'm^2/s', display: 215.29,
    display_unit: 'm^2/day', ci: [2.48e-3, 2.5e-3], display_ci: [214.9, 215.7],
    reliable: true, reliability_note: '', derived: false, fixed: false,
    at_bound: null, ...overrides,
  }
}

function result(overrides: Record<string, unknown> = {}) {
  return {
    success: true, error_message: '', method: 'Theis (1935)', method_id: 'theis_1935',
    T: 2.4918e-3, T_day: 215.29, T_ci: [2.487e-3, 2.496e-3],
    S: 5.05e-5, S_ci: [5.01e-5, 5.09e-5], estimates_storativity: true,
    rmse: 0.00244, r_squared: 0.99997,
    parameters: [parameter(), parameter({ name: 'S', symbol: 'S', value: 5.05e-5,
      unit: '-', display: 5.05e-5, display_unit: '-', ci: [5.01e-5, 5.09e-5],
      display_ci: [5.01e-5, 5.09e-5] })],
    gof: { aicc: -520.1 }, diagnostics: {},
    validity_notes: ['u < 0.02 satisfied for all late-time points.'],
    figure: JSON.stringify({ data: [{}, {}], layout: {} }),
    ...overrides,
  }
}

describe('the results', () => {
  it('shows nothing before an analysis has run', async () => {
    const stub = stubFetch([{ url: '/api/analyses/current', status: 404,
                              body: { detail: 'No analysis has been run yet.' } }])

    const { container } = renderWithProviders(<ResultsPanel />)

    await waitFor(() => expect(stub.calls.length).toBeGreaterThan(0))
    expect(container.textContent).not.toContain('Results')
  })

  it('reports transmissivity, storativity and fit quality', async () => {
    stubFetch([{ url: '/api/analyses/current', body: result() }])

    renderWithProviders(<ResultsPanel />)

    // Four significant figures: the 95% interval on this fit is about
    // +/- 4 m2/day, so a second decimal would be invented precision.
    expect(await screen.findByText('215.3')).toBeInTheDocument()
    expect(screen.getByText('5.05e-5')).toBeInTheDocument()
    expect(screen.getByText('1.0000')).toBeInTheDocument()
    expect(screen.getByText('Validity notes (1)')).toBeInTheDocument()
  })

  it('does not invent a storativity for a method that estimates none', async () => {
    stubFetch([{ url: '/api/analyses/current', body: result({
      method: 'Theis Recovery (1935)', S: null, S_ci: null,
      estimates_storativity: false }) }])

    renderWithProviders(<ResultsPanel />)

    expect(await screen.findByText('not determined')).toBeInTheDocument()
  })

  it('flags parameters the method itself does not trust', async () => {
    stubFetch([{ url: '/api/analyses/current', body: result({
      parameters: [parameter(), parameter({ name: 'S', symbol: 'S', unit: '-',
        display_unit: '-', reliable: false,
        reliability_note: 'Papadopulos & Cooper (1967) p. 244' }),
        parameter({ name: 'r_over_B', symbol: 'r/B', unit: '-', display_unit: '-',
                    ci: null, derived: true })] }) }])

    renderWithProviders(<ResultsPanel />)

    expect(await screen.findByText('low reliability')).toBeInTheDocument()
    expect(screen.getAllByText('derived').length).toBeGreaterThan(0)
  })

  it('explains a fit that did not converge', async () => {
    stubFetch([{ url: '/api/analyses/current', body: {
      success: false, error_message: 'Insufficient data: 8 points (minimum 10 required).',
      method: 'Theis (1935)', method_id: 'theis_1935' } }])

    renderWithProviders(<ResultsPanel />)

    expect(await screen.findByText(/Insufficient data: 8 points/)).toBeInTheDocument()
  })

  it('saves the analysis to the project, once', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/save', method: 'POST',
        body: { test_id: 'test-1', test_name: 'Test 1', rows: [], n_stale: 0 } },
      { url: '/api/analyses/current', body: result() },
      { url: '/api/project', body: {} },
    ])

    renderWithProviders(<ResultsPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /Save to project/ }))

    expect(await screen.findByRole('button', { name: /Saved to project/ })).toBeDisabled()
    expect(stub.calls.filter((c) => c.url.includes('/api/analyses/save'))).toHaveLength(1)
  })

  it('draws the manual curve at the fitted values first', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/manual-match', method: 'POST',
        body: { figure: JSON.stringify({ data: [{}], layout: {} }),
                rmse_mm: 2.44, auto_rmse_mm: 2.44 } },
      { url: '/api/analyses/current', body: result() },
    ])

    renderWithProviders(<ResultsPanel />)
    fireEvent.click(await screen.findByRole('tab', { name: 'Manual matching' }))

    expect(await screen.findByText('Manual RMSE 2.44 mm')).toBeInTheDocument()
    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/analyses/manual-match'))!
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.T).toBeCloseTo(2.4918e-3, 8)
    expect(body.S).toBeCloseTo(5.05e-5, 10)
  })

  it('matches on T alone when the method has no storativity', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/manual-match', method: 'POST',
        body: { figure: '{}', rmse_mm: 1, auto_rmse_mm: 1 } },
      { url: '/api/analyses/current', body: result({ S: null, estimates_storativity: false }) },
    ])

    renderWithProviders(<ResultsPanel />)
    fireEvent.click(await screen.findByRole('tab', { name: 'Manual matching' }))

    await waitFor(() =>
      expect(stub.calls.some((c) => c.url.includes('manual-match'))).toBe(true))
    expect(screen.queryByLabelText('log10 S')).not.toBeInTheDocument()
    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/analyses/manual-match'))!
    expect(JSON.parse((init as RequestInit).body as string).S).toBeNull()
  })
})

describe('asking the consultant about the result', () => {
  it('sends the interpretation question the Streamlit app asked', async () => {
    const stub = stubFetch([
      { url: '/api/chat/messages', method: 'POST', body: chatState },
      { url: '/api/analyses/current', body: result() },
    ])

    renderWithProviders(<ResultsPanel />)
    fireEvent.click(await screen.findByRole('button', { name: /Interpret in chat/ }))

    await waitFor(() =>
      expect(stub.calls.some((c) => c.url.includes('/api/chat/messages'))).toBe(true))
    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/chat/messages'))!
    expect(JSON.parse((init as RequestInit).body as string).message)
      .toMatch(/^Interpret the pumping test results I just computed/)
    expect(await screen.findByText('The consultant has replied')).toBeInTheDocument()
  })
})

describe('the unit switch', () => {
  it('shows transmissivity in gpd/ft when the workspace is set to US units', async () => {
    // The literature a user checks against is often in gpd/ft — Walton's
    // 11,000 for Gridley — and converting by hand to compare is arithmetic the
    // software should absorb.
    localStorage.setItem('hydroassist.units', 'US')
    try {
      stubFetch([{ url: '/api/analyses/current', body: result() }])

      renderWithProviders(<ResultsPanel />)

      expect(await screen.findByText('T (gpd/ft)')).toBeInTheDocument()
      // 215.29 m2/day, to the four significant figures the fit supports.
      expect(screen.getByText('17340')).toBeInTheDocument()
    } finally {
      // Without this a failure above leaks the preference into every later
      // test, which is how one broken assertion becomes two.
      localStorage.removeItem('hydroassist.units')
    }
  })

  it('shows SI by default', async () => {
    stubFetch([{ url: '/api/analyses/current', body: result() }])

    renderWithProviders(<ResultsPanel />)

    expect(await screen.findByText('T (m²/day)')).toBeInTheDocument()
    expect(screen.getByText('215.3')).toBeInTheDocument()
  })
})

function tracerResult() {
  const alpha = parameter({ name: 'alpha_L', symbol: 'α_L', value: 1.02, unit: 'm',
    display: 1.02, display_unit: 'm', ci: [0.9, 1.1], display_ci: [0.9, 1.1] })
  const tc = parameter({ name: 't_c', symbol: 't_c', value: 2.4e5, unit: 's',
    display: 66.7, display_unit: 'hours', ci: [2.3e5, 2.5e5], display_ci: [64, 69] })
  const ne = parameter({ name: 'n_e', symbol: 'n_e', value: 0.2, unit: '-', display: 0.2,
    display_unit: '-', ci: null, display_ci: null, derived: true })
  return result({
    method: 'Sauty (1980)', method_id: 'sauty_1980', T: 0, T_day: 0, T_ci: null,
    S: null, S_ci: null, estimates_storativity: false, rmse: 0.03,
    response_unit: 'mg/L', parameters: [tc, alpha, ne],
    manual_axes: [
      { name: 't_c', symbol: 't_c', value: 2.4e5, display_scale: 1 / 3600, display_unit: 'hours' },
      { name: 'alpha_L', symbol: 'α_L', value: 1.02, display_scale: 1, display_unit: 'm' },
    ],
  })
}

describe('a tracer result', () => {
  it('leads with what it estimated, and has no T or S to report', async () => {
    stubFetch([{ url: '/api/analyses/current', body: tracerResult() }])
    renderWithProviders(<ResultsPanel />)

    expect(await screen.findByText('Results — Sauty (1980)')).toBeInTheDocument()
    expect(screen.getAllByText('α_L').length).toBeGreaterThan(0)
    expect(screen.getAllByText('66.70 hours').length).toBeGreaterThan(0)
    expect(screen.getByText(/RMSE 0.0300 mg\/L/)).toBeInTheDocument()
    expect(screen.queryByText(/^T \(/)).not.toBeInTheDocument()
    expect(screen.queryByText('not determined')).not.toBeInTheDocument()
  })

  it('moves its own parameters when matching by hand, in concentration units', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/manual-match', method: 'POST',
        body: { figure: JSON.stringify({ data: [{}], layout: {} }), rmse_mm: 30, auto_rmse_mm: 30 } },
      { url: '/api/analyses/current', body: tracerResult() },
    ])
    renderWithProviders(<ResultsPanel />)
    fireEvent.click(await screen.findByRole('tab', { name: 'Manual matching' }))

    expect(await screen.findByLabelText('log10 alpha_L')).toBeInTheDocument()
    expect(screen.queryByLabelText('log10 T')).not.toBeInTheDocument()
    expect(await screen.findByText('Manual RMSE 0.0300 mg/L')).toBeInTheDocument()

    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/analyses/manual-match'))!
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.T).toBeCloseTo(2.4e5, 0)
    expect(body.S).toBeCloseTo(1.02, 6)
  })
})
