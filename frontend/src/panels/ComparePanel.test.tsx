/** The comparison: ranked by AICc, with the evidence spelled out. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ComparePanel } from './ComparePanel'
import { renderWithProviders } from '../test/render'
import { project, stubFetch } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

function row(overrides: Record<string, unknown> = {}) {
  return {
    rank: 1, analysis_id: 'a1', method: 'Hantush-Jacob (1955)', n_params: 3,
    T_m2_per_day: 216.0, S: 5e-5, r_squared: 0.9999, aicc: -600.2,
    delta_aicc: 0, akaike_weight: 1.0, evidence: 'best', ...overrides,
  }
}

const TWO_ROWS = {
  test_id: 'test-1', test_name: 'Test 1', n_stale: 0,
  rows: [row(), row({ rank: 2, analysis_id: 'a2', method: 'Theis (1935)', n_params: 2,
                      delta_aicc: 41.7, akaike_weight: 0.0,
                      evidence: 'essentially no support' })],
}

describe('the comparison', () => {
  it('shows nothing until an analysis is saved', async () => {
    const stub = stubFetch([
      { url: '/comparison', body: { test_id: 'test-1', test_name: 'Test 1',
                                    rows: [], n_stale: 0 } },
      { url: '/api/project', body: project() },
    ])

    const { container } = renderWithProviders(<ComparePanel />)

    await waitFor(() =>
      expect(stub.calls.some((c) => c.url.includes('/comparison'))).toBe(true))
    expect(container.textContent).not.toContain('Compare saved analyses')
  })

  it('ranks the saved methods and says how strong the evidence is', async () => {
    stubFetch([
      { url: '/comparison', body: TWO_ROWS },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ComparePanel />)

    expect(await screen.findByText('Compare saved analyses — Test 1')).toBeInTheDocument()
    const methods = screen.getAllByRole('row').slice(1)
      .map((tr) => tr.textContent ?? '')
    expect(methods[0]).toContain('Hantush-Jacob (1955)')
    expect(methods[1]).toContain('Theis (1935)')
    expect(screen.getByText('essentially no support')).toBeInTheDocument()
    expect(screen.getByText('41.7')).toBeInTheDocument()
  })

  it('notes analyses run on readings the test no longer holds', async () => {
    stubFetch([
      { url: '/comparison', body: { ...TWO_ROWS, n_stale: 2 } },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ComparePanel />)

    expect(await screen.findByText(/2 earlier analysis\(es\) were run on different readings/))
      .toBeInTheDocument()
  })

  it('removes an analysis', async () => {
    const stub = stubFetch([
      { url: '/analyses/a2', method: 'DELETE', body: { ...TWO_ROWS, rows: [row()] } },
      { url: '/comparison', body: TWO_ROWS },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ComparePanel />)
    fireEvent.click(await screen.findByLabelText('Remove Theis (1935)'))

    await waitFor(() =>
      expect(stub.calls.some((c) => c.method === 'DELETE'
                                   && c.url.includes('/analyses/a2'))).toBe(true))
  })
})
