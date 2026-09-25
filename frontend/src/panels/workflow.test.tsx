/**
 * Upload, then run, then see the result — the panels together, as a user meets
 * them.
 *
 * Each panel's own tests passed while this sequence was broken in the browser:
 * clearing the result on upload with removeQueries deleted the cache entry the
 * results panel was subscribed to, so the run's result was written to a new
 * entry that nothing was watching and the panel never appeared. Only a test
 * that performs both steps in one page can catch that.
 */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DataPanel } from './DataPanel'
import { ResultsPanel } from './ResultsPanel'
import { SetupPanel } from './SetupPanel'
import { csvFile, fileInput, renderWithProviders } from '../test/render'
import { prepared, project, stubFetch, test_ } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

const THEIS = {
  id: 'theis_1935', label: 'Theis (1935)', year: 1935, authors: ['Theis, C.V.'],
  citation: '', aquifer_types: ['confined'], test_types: ['constant_rate'],
  parameters: ['T', 'S'], data_source: 'observation_well', requires: [],
  ui_help: 'Full-curve type-curve fit.', selection_rationale: '',
  needs_well_geometry: false, estimates_storativity: true,
}

const RESULT = {
  success: true, error_message: '', method: 'Theis (1935)', method_id: 'theis_1935',
  T: 2.4918e-3, T_day: 215.29, T_ci: [2.487e-3, 2.496e-3], S: 5.05e-5,
  S_ci: [5.01e-5, 5.09e-5], estimates_storativity: true, rmse: 0.00244,
  r_squared: 0.99997, parameters: [], gof: null, diagnostics: {},
  validity_notes: [], figure: null,
}

describe('the workspace flow', () => {
  it('shows the result of a run that follows an upload', async () => {
    stubFetch([
      { url: '/readings', method: 'POST', body: prepared() },
      { url: '/readings', body: prepared() },
      { url: '/api/analyses/run', method: 'POST', body: RESULT },
      { url: '/api/analyses/current', status: 404,
        body: { detail: 'No analysis has been run yet.' } },
      { url: '/api/methods', body: { test_type: 'constant_rate', offered: [THEIS],
                                     gated_out: [], help: '', default_index: 0 } },
      { url: '/api/test-types', body: [{ id: 'constant_rate',
                                         label: 'Constant-rate pumping' }] },
      { url: '/api/chat', body: {} },
      { url: '/api/project', body: project({ tests: [test_({ n_readings: 50 })] }) },
    ])

    const { container } = renderWithProviders(
      <>
        <DataPanel />
        <SetupPanel />
        <ResultsPanel />
      </>,
    )

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Replace file/ })).toBeEnabled())
    fireEvent.change(fileInput(container), { target: { files: [csvFile()] } })
    await waitFor(() =>
      expect(screen.getAllByText(/50 data points ready/).length).toBeGreaterThan(0))

    const run = screen.getByRole('button', { name: /Run analysis/ })
    await waitFor(() => expect(run).toBeEnabled())
    fireEvent.click(run)

    expect(await screen.findByText('Results — Theis (1935)')).toBeInTheDocument()
    expect(screen.getByText('215.29')).toBeInTheDocument()
  })
})
