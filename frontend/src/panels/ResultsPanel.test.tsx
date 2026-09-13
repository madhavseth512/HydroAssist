/** Steps 4-5: the numbers, the caveats, saving, the report and manual matching. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ResultsPanel } from './ResultsPanel'
import { renderWithProviders } from '../test/render'
import { stubFetch } from '../test/stubFetch'

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

    expect(await screen.findByText('215.29')).toBeInTheDocument()
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
