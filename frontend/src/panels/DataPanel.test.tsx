/** Step 1: choosing a file, and what the server says about it. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DataPanel } from './DataPanel'
import { csvFile, fileInput, renderWithProviders } from '../test/render'
import { prepared, project, stubFetch, test_ } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

const WITH_READINGS = project({
  tests: [test_({ n_readings: 50, readings_source: 'pw1.csv' })],
})

describe('the data step', () => {
  it('explains what to upload when a test is empty', async () => {
    stubFetch([{ url: '/api/project', body: project() }])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText(/a time column and a drawdown column/))
      .toBeInTheDocument()
    expect(screen.getByText('Choose file')).toBeInTheDocument()
  })

  it('uploads a file and shows what was read', async () => {
    const stub = stubFetch([
      { url: '/readings', method: 'POST', body: prepared() },
      { url: '/api/project', body: project() },
    ])

    const { container } = renderWithProviders(<DataPanel />)
    // Wait for the project: until it has loaded there is no test to upload into.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Choose file/ })).toBeEnabled())

    fireEvent.change(fileInput(container), { target: { files: [csvFile()] } })

    expect(await screen.findByText(/50 data points ready/)).toBeInTheDocument()
    expect(stub.calls.some((c) => c.method === 'POST' && c.url.includes('/readings')))
      .toBe(true)
  })

  it('reads the flow regime off the data, before any method is chosen', async () => {
    stubFetch([
      { url: '/readings', body: prepared() },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText('Radial Flow')).toBeInTheDocument()
    expect(screen.getByText(/The derivative is flat at late time/))
      .toBeInTheDocument()
    expect(screen.getByText(/Consistent with: Theis \(1935\)/)).toBeInTheDocument()
  })

  it('shows the readings a reopened project already holds', async () => {
    stubFetch([
      { url: '/readings', body: prepared() },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText('From pw1.csv')).toBeInTheDocument()
    expect(screen.getByText('Replace file')).toBeInTheDocument()
  })

  it('reports what the validator refuses to analyse', async () => {
    stubFetch([
      { url: '/readings', body: prepared({
        validation: { is_usable: false, errors: ['Dataset: 4 points (minimum 10).'],
                      warnings: ['Dataset: time is not increasing.'],
                      info: [], suitable_methods: [] } }) },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText('Dataset: 4 points (minimum 10).')).toBeInTheDocument()
    expect(screen.getByText('Dataset: time is not increasing.')).toBeInTheDocument()
    expect(screen.getByText(/cannot be analysed as it stands/)).toBeInTheDocument()
  })

  it('explains a file the server could not use', async () => {
    stubFetch([
      { url: '/readings', method: 'POST', status: 400,
        body: { detail: 'That file could not be read as CSV.' } },
      { url: '/api/project', body: project() },
    ])

    const { container } = renderWithProviders(<DataPanel />)
    // Wait for the project: until it has loaded there is no test to upload into.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Choose file/ })).toBeEnabled())

    fireEvent.change(fileInput(container), { target: { files: [csvFile()] } })

    await waitFor(() =>
      expect(screen.getByText('That file could not be read as CSV.'))
        .toBeInTheDocument())
  })

  it('says when the flow regime could not be read', async () => {
    stubFetch([
      { url: '/readings', body: prepared({ diagnosis: null,
                                           diagnosis_error: 'too few points' }) },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText(/too few points/)).toBeInTheDocument()
  })
})

describe('what column detection was unsure about', () => {
  it('is shown as a warning', async () => {
    const issue = "Column 'dd' has 3 negative drawdown value(s) — check for recovery data."
    stubFetch([
      { url: '/readings', body: prepared({ inspection: {
        ...prepared().inspection, issues: [issue] } }) },
      { url: '/api/project', body: WITH_READINGS },
    ])

    renderWithProviders(<DataPanel />)

    expect(await screen.findByText(issue)).toBeInTheDocument()
  })
})

describe('pasting readings', () => {
  const PASTED = 'time_s\tdrawdown_m\n60\t0.12\n120\t0.19'

  it('is hidden until asked for', async () => {
    stubFetch([{ url: '/api/project', body: project() }])

    renderWithProviders(<DataPanel />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Paste data' })).toBeEnabled())
    expect(screen.queryByLabelText('Pasted readings')).toBeNull()
  })

  it('sends what was pasted to the server', async () => {
    // Most people have the numbers open in a spreadsheet; saving a file first
    // to upload it earns nothing.
    const stub = stubFetch([
      { url: '/api/project', body: project() },
      { url: '/readings/text', method: 'POST', body: prepared() },
    ])

    renderWithProviders(<DataPanel />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Paste data' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Paste data' }))
    fireEvent.change(screen.getByLabelText('Pasted readings'), {
      target: { value: PASTED },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Use these readings' }))

    await waitFor(() => {
      const call = stub.mock.mock.calls.find(([url]) =>
        String(url).includes('/readings/text'))
      expect(call).toBeTruthy()
      expect(JSON.parse((call![1] as RequestInit).body as string))
        .toEqual({ text: PASTED })
    })
  })

  it('will not send an empty box', async () => {
    stubFetch([{ url: '/api/project', body: project() }])

    renderWithProviders(<DataPanel />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Paste data' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Paste data' }))

    expect(screen.getByRole('button', { name: 'Use these readings' })).toBeDisabled()
  })

  it('shows what the server made of the readings', async () => {
    stubFetch([
      { url: '/api/project', body: project() },
      { url: '/readings/text', method: 'POST', body: prepared() },
    ])

    renderWithProviders(<DataPanel />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Paste data' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: 'Paste data' }))
    fireEvent.change(screen.getByLabelText('Pasted readings'), {
      target: { value: PASTED },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Use these readings' }))

    expect(await screen.findByText(/50 data points ready/)).toBeInTheDocument()
  })
})
