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
    needs_well_geometry: false, estimates_storativity: true, uses_pumping_rate: true, ...overrides,
  }
}

const PC = method({
  id: 'papadopulos_cooper_1967', label: 'Papadopulos-Cooper (1967)',
  data_source: 'pumped_well', needs_well_geometry: true,
  ui_help: 'Large-diameter well with wellbore storage.',
})

const CJ = method({
  id: 'cooper_jacob_1946', label: 'Cooper-Jacob (1946)',
  ui_help: 'Late-time semi-log straight line.',
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
    expect(screen.getByRole('button', { name: /^Theis \(1935\)/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Papadopulos-Cooper/ })).toBeInTheDocument()
  })

  it('preselects the method the conversation settled on', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method(), PC], { default_index: 1 }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    // The card the conversation settled on is the selected one, and the panel
    // shows what that method needs — the well radii, here.
    expect(await screen.findByText(/fitted to readings from the pumped well/))
      .toBeInTheDocument()
    expect(screen.getByLabelText(/Well screen radius/)).toBeInTheDocument()
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
      { url: 'filtered=false', body: methodList([method(), CJ]) },
      { url: '/api/methods', body: methodList([method()], {
        gated_out: ['Cooper-Jacob (1946)'],
        gated_reasons: {
          'Cooper-Jacob (1946)':
            'Cooper-Jacob (1946) may not be valid: u < 0.05 is not reached '
            + 'within the test duration.',
        },
      }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    // Shown as a card, greyed, carrying the reason — which is the part a
    // practitioner learns from. A name alone teaches nothing.
    const card = await screen.findByRole('button',
                                         { name: /Cooper-Jacob \(1946\).*not available/ })
    expect(card).toBeInTheDocument()
    expect(screen.getByText(/u < 0.05 is not reached/)).toBeInTheDocument()
  })

  it('runs the chosen method with the setup on screen', async () => {
    const stub = stubFetch([
      { url: '/api/analyses/run', method: 'POST', body: { success: true } },
      { url: '/api/methods', body: methodList([method()]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)
    await screen.findByRole('button', { name: /^Theis \(1935\)/ })

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

describe('the rate the calculators receive', () => {
  it('is shown in m³/s beside the form', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([method()]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    // 500 m³/day = 5.787e-3 m³/s
    expect(await screen.findByText('Q = 5.7870e-3 m³/s')).toBeInTheDocument()
  })

  it('explains α when the method is fitted at the pumped well', async () => {
    stubFetch([
      { url: '/api/methods', body: methodList([PC]) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<SetupPanel />)

    expect(await screen.findByText(/dimensionless wellbore-storage parameter/))
      .toBeInTheDocument()
  })
})

const THIERY = method({
  id: 'thiery_2022', label: 'Thiéry (2022)', test_types: ['tracer'],
  aquifer_types: ['porous'], parameters: ['t_c', 'alpha_L', 'amplitude'],
  estimates_storativity: false,
})
const OGATA = method({
  id: 'ogata_banks_1961', label: 'Ogata-Banks (1961)', test_types: ['tracer'],
  aquifer_types: ['porous'], parameters: ['v', 'alpha_L', 'amplitude'],
  estimates_storativity: false, uses_pumping_rate: false,
})
const TRACER_PROJECT = project({ tests: [test_({
  n_readings: 40, setup: { ...setup, test_type: 'tracer' } })] })

describe('the setup step for a tracer test', () => {
  function stubTracer(offered: unknown[]) {
    return stubFetch([
      { url: '/api/methods', body: methodList(offered, { test_type: 'tracer' }) },
      { url: '/api/test-types', body: TEST_TYPES },
      { url: '/api/analyses/run', method: 'POST', body: { success: true } },
      { url: '/api/project', body: TRACER_PROJECT },
    ])
  }

  it('asks a forced-gradient test for Q, thickness and mass', async () => {
    stubTracer([THIERY])
    renderWithProviders(<SetupPanel />)

    expect(await screen.findByLabelText('Pumping rate Q')).toBeInTheDocument()
    expect(screen.getByLabelText('Aquifer thickness b (m), for porosity')).toBeInTheDocument()
    expect(screen.getByLabelText('Injected mass (kg), for recovery')).toBeInTheDocument()
    expect(screen.queryByLabelText(/Hydraulic conductivity/)).not.toBeInTheDocument()
  })

  it('asks a natural-gradient test for K and the gradient, and no rate', async () => {
    stubTracer([OGATA])
    renderWithProviders(<SetupPanel />)

    expect(await screen.findByLabelText(/Hydraulic conductivity/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Hydraulic gradient/)).toBeInTheDocument()
    expect(screen.queryByLabelText('Pumping rate Q')).not.toBeInTheDocument()
    expect(screen.queryByText(/m³\/s/)).not.toBeInTheDocument()
  })

  it('calls r the distance from the injection', async () => {
    stubTracer([THIERY])
    renderWithProviders(<SetupPanel />)

    expect(await screen.findByLabelText('Distance from injection to sampling well (m)'))
      .toBeInTheDocument()
  })

  it('offers a plateau only for a continuous injection', async () => {
    stubTracer([THIERY])
    renderWithProviders(<SetupPanel />)

    await screen.findByLabelText('Injected mass (kg), for recovery')
    expect(screen.queryByLabelText(/Plateau concentration/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('Continuous'))

    expect(await screen.findByLabelText(/Plateau concentration/)).toBeInTheDocument()
    expect(screen.queryByLabelText('Injected mass (kg), for recovery')).not.toBeInTheDocument()
  })

  it('sends what was entered', async () => {
    const stub = stubTracer([THIERY])
    renderWithProviders(<SetupPanel />)

    fireEvent.change(await screen.findByLabelText('Aquifer thickness b (m), for porosity'),
                     { target: { value: '12' } })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run analysis' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Run analysis' }))

    await waitFor(() => expect(stub.calls.some((c) => c.url.includes('/analyses/run'))).toBe(true))
    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/analyses/run'))!
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.method).toBe('thiery_2022')
    expect(body.setup.tracer_b_m).toBe(12)
  })

  it('stores the test type as soon as it is chosen, since the next upload depends on it', async () => {
    const stub = stubFetch([
      { url: '/api/methods', body: methodList([method()]) },
      { url: '/api/test-types', body: [...TEST_TYPES, { id: 'tracer', label: 'Tracer test' }] },
      { url: '/setup', method: 'PUT', body: test_() },
      { url: '/api/project', body: WITH_READINGS },
    ])
    renderWithProviders(<SetupPanel />)

    fireEvent.click(await screen.findByText('Tracer test'))

    await waitFor(() => expect(stub.calls.some((c) => c.url.includes('/setup'))).toBe(true))
    const [, init] = stub.mock.mock.calls.find(([url]) => String(url).includes('/setup'))!
    expect(JSON.parse((init as RequestInit).body as string).test_type).toBe('tracer')
  })
})
