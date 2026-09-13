/** Step 2: the registry drives the form, and Run sends what the form shows. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SetupPanel } from './SetupPanel'
import { renderWithProviders } from '../test/render'
import { project, setup, stubFetch, test_ } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

function method(overrides: Record<string, unknown> = {}) {
  return {
    id: 'theis_1935', label: 'Theis (1935)', year: 1935, authors: ['Theis, C.V.'],
    citation: '', aquifer_types: ['confined'], test_types: ['constant_rate'],
    parameters: ['T', 'S'], data_source: 'observation_well', requires: [],
    ui_help: 'Full-curve type-curve fit.', selection_rationale: '',
    needs_well_geometry: false, estimates_storativity: true, ...overrides,
  }
}

const PC = method({
  id: 'papadopulos_cooper_1967', label: 'Papadopulos-Cooper (1967)',
  data_source: 'pumped_well', needs_well_geometry: true,
  ui_help: 'Large-diameter well with wellbore storage.',
})

const TEST_TYPES = [
  { id: 'constant_rate', label: 'Constant-rate pumping' },
  { id: 'recovery', label: 'Recovery (after the pump stops)' },
]

function methodList(offered: unknown[], extra: Record<string, unknown> = {}) {
  return { test_type: 'constant_rate', offered, gated_out: [], help: '',
           default_index: 0, ...extra }
}

const WITH_READINGS = project({ tests: [test_({ n_readings: 50 })] })

describe('the setup step', () => {
  it('offers the registry methods and explains the chosen one', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method(), PC]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByText('Full-curve type-curve fit.')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Theis (1935)')).toBeInTheDocument()
  })

  it('preselects the method the conversation settled on', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method(), PC], { default_index: 1 }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByDisplayValue('Papadopulos-Cooper (1967)')).toBeInTheDocument()
  })

  it('asks for the well radii only for a pumped-well method', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([PC]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByLabelText('Well screen radius r_w (m)')).toBeInTheDocument()
    expect(screen.getByLabelText('Casing radius r_c (m)')).toBeInTheDocument()
  })

  it('asks for the pumping duration on a recovery test', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method({
        id: 'theis_recovery_1935', label: 'Theis Recovery (1935)',
        test_types: ['recovery'], estimates_storativity: false })],
        { test_type: 'recovery' }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: project({ tests: [test_({
        n_readings: 50, setup: { ...setup, test_type: 'recovery', t_p_hours: 24 } })] }) },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByLabelText('Pumping duration before shut-down t_p (hours)'))
      .toBeInTheDocument()
    expect(screen.getByText('The time column counts')).toBeInTheDocument()
  })

  it('says why Run is unavailable without readings', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method()]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByText(/Upload readings in Step 1/)).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Run analysis/ })).toBeDisabled())
  })

  it('lists methods the validator did not offer', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method()],
        { gated_out: ['Cooper-Jacob (1946)'] }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByText(/Not offered for this dataset: Cooper-Jacob/))
      .toBeInTheDocument()
  })

  it('runs the chosen method with the setup on screen', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/run', method: 'POST', body: { success: true } },
      { url: '/api/methods', body: methodList([method()]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)
    await screen.findByDisplayValue('Theis (1935)')

    fireEvent.change(screen.getByLabelText('Observation well distance r (m)'),
                     { target: { value: '42' } })
    const run = screen.getByRole('button', { name: /Run analysis/ })
    await waitFor(() => expect(run).not.toBeDisabled())
    fireEvent.click(run)

    await waitFor(() =>
      expect(stub.calls.some((c) => c.url.includes('/api/analyses/run'))).toBe(true))

    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/analyses/run'))!
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.method).toBe('theis_1935')
    expect(body.setup.r_m).toBe(42)
    expect(body.setup.t_p_hours).toBeNull()
  })
})
