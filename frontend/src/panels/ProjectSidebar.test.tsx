/**
 * The project panel: open, save, switch test, edit details.
 *
 * Interactions use fireEvent: it dispatches exactly the event each handler
 * listens for, which keeps these tests quick and free of Mantine's portal and
 * pointer-capture behaviour.
 */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectSidebar } from './ProjectSidebar'
import { renderWithProviders } from '../test/render'
import { project, stubFetch, test_ } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

const TWO_TESTS = project({
  tests: [test_({ n_readings: 50 }), test_({ id: 'test-2', name: 'PW-2' })],
  summary: '2 tests · 0 saved analyses',
})

function called(stub: { calls: { url: string; method: string }[] },
                fragment: string, method?: string) {
  return stub.calls.some(
    (call) => call.url.includes(fragment) && (!method || call.method === method))
}

describe('the project panel', () => {
  it('shows the project title and what it holds', async () => {
    stubFetch([{ url: '/api/project', body: project() }])

    renderWithProviders(<ProjectSidebar />)

    expect(await screen.findByText('Untitled project')).toBeInTheDocument()
    expect(screen.getByText('1 test · 0 saved analyses')).toBeInTheDocument()
  })

  it('says whether the active test has readings', async () => {
    stubFetch([{ url: '/api/project', body: TWO_TESTS }])

    renderWithProviders(<ProjectSidebar />)

    // On the tree row for that test, rather than once under a dropdown.
    const row = await screen.findByRole('treeitem', { name: 'Test 1 (open)' })
    expect(row).toHaveTextContent('50 readings')
  })

  it('switches the active test', async () => {
    const stub = stubFetch([
      { url: '/api/tests/test-2/activate', method: 'POST', body: TWO_TESTS },
      { url: '/api/project', body: TWO_TESTS },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(await screen.findByRole('treeitem', { name: 'PW-2' }))

    await waitFor(() =>
      expect(called(stub, '/api/tests/test-2/activate', 'POST')).toBe(true))
  })

  it('adds a test', async () => {
    const stub = stubFetch([
      { url: '/api/tests', method: 'POST', body: TWO_TESTS },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByLabelText('Add test'))

    await waitFor(() => expect(called(stub, '/api/tests', 'POST')).toBe(true))
  })

  it('saves the project details that appear on the report', async () => {
    const stub = stubFetch([
      { url: '/api/project/info', method: 'PATCH', body: project() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('Project and test details'))
    fireEvent.change(await screen.findByLabelText('Client'),
                     { target: { value: 'IIT Kharagpur' } })
    fireEvent.click(screen.getByText('Save details'))

    await waitFor(() => expect(called(stub, '/api/project/info', 'PATCH')).toBe(true))
  })

  it('renames the active test when the field loses focus', async () => {
    const stub = stubFetch([
      { url: '/api/tests/test-1', method: 'PATCH', body: project() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('Project and test details'))
    const field = await screen.findByLabelText('Test name')
    fireEvent.change(field, { target: { value: 'PW-1 constant rate' } })
    fireEvent.blur(field)

    await waitFor(() => expect(called(stub, '/api/tests/test-1', 'PATCH')).toBe(true))
  })

  it('asks before discarding an unsaved project', async () => {
    const stub = stubFetch([
      { url: '/api/project/new', method: 'POST', body: project() },
      { url: '/api/project', body: project() },
    ])
    vi.stubGlobal('confirm', vi.fn(() => false))

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('New project'))

    expect(window.confirm).toHaveBeenCalled()
    expect(called(stub, '/api/project/new', 'POST')).toBe(false)
  })

  it('discards the project when the warning is accepted', async () => {
    const stub = stubFetch([
      { url: '/api/project/new', method: 'POST', body: project() },
      { url: '/api/project', body: project() },
    ])
    vi.stubGlobal('confirm', vi.fn(() => true))

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('New project'))

    await waitFor(() => expect(called(stub, '/api/project/new', 'POST')).toBe(true))
  })
})

describe('the wells a test was run between', () => {
  it('are recorded against the test', async () => {
    const stub = stubFetch([
      { url: '/api/tests/test-1', method: 'PATCH', body: project() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('Project and test details'))
    const pumping = await screen.findByLabelText('Pumping well')
    fireEvent.change(pumping, { target: { value: 'PW-1' } })
    fireEvent.blur(pumping)

    await waitFor(() => expect(called(stub, '/api/tests/test-1', 'PATCH')).toBe(true))
    const [, init] = stub.mock.mock.calls.find(
      ([url]) => String(url).includes('/api/tests/test-1'))!
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({ pumping_well: 'PW-1' })
  })

  it('never leaves a test without a name', async () => {
    const stub = stubFetch([
      { url: '/api/tests/test-1', method: 'PATCH', body: project() },
      { url: '/api/project', body: project() },
    ])

    renderWithProviders(<ProjectSidebar />)
    await screen.findByText('Untitled project')

    fireEvent.click(screen.getByText('Project and test details'))
    const name = await screen.findByLabelText('Test name')
    fireEvent.change(name, { target: { value: '   ' } })
    fireEvent.blur(name)

    expect(called(stub, '/api/tests/test-1', 'PATCH')).toBe(false)
    await waitFor(() => expect(name).toHaveValue('Test 1'))
  })
})
