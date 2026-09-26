/**
 * The offer to carry on, and the cases where it must not appear.
 *
 * Restoring replaces whatever is open, so the important tests here are the
 * refusals: never over work in progress, never twice, and never by itself.
 */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { RestoreBanner } from './RestoreBanner'
import { renderWithProviders } from '../test/render'
import { project, stubFetch, test_ } from '../test/stubFetch'
import { saveProject } from '../persistence'

afterEach(() => {
  vi.unstubAllGlobals()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

const KEPT = JSON.stringify({ version: 2, title: 'Gridley', tests: [] })
const FRESH = project()
const IN_PROGRESS = project({ tests: [test_({ n_readings: 50 })] })

describe('the offer to carry on', () => {
  it('offers a kept project when the session starts empty', () => {
    saveProject(KEPT, '1 test · 1 saved analysis')
    stubFetch([{ url: '/api/project', body: FRESH }])

    renderWithProviders(<RestoreBanner project={FRESH} />)

    expect(screen.getByText(/Carry on where you left off/)).toBeInTheDocument()
    expect(screen.getByText('1 test · 1 saved analysis')).toBeInTheDocument()
  })

  it('says nothing when there is nothing kept', () => {
    renderWithProviders(<RestoreBanner project={FRESH} />)

    expect(screen.queryByText(/Carry on where you left off/)).toBeNull()
  })

  it('does not offer to overwrite work already in progress', () => {
    // The dangerous case: someone who deliberately started fresh, or who has
    // uploaded readings already, must not be invited to replace them.
    saveProject(KEPT, 'an older project')

    renderWithProviders(<RestoreBanner project={IN_PROGRESS} />)

    expect(screen.queryByText(/Carry on where you left off/)).toBeNull()
  })

  it('waits for the project before deciding', () => {
    saveProject(KEPT, 'an older project')

    renderWithProviders(<RestoreBanner project={undefined} />)

    expect(screen.queryByText(/Carry on where you left off/)).toBeNull()
  })

  it('restores by the same route as a reopened file', async () => {
    saveProject(KEPT, 'a project')
    const stub = stubFetch([
      { url: '/api/project/open', method: 'POST', body: IN_PROGRESS },
      { url: '/api/project', body: FRESH },
    ])

    renderWithProviders(<RestoreBanner project={FRESH} />)
    fireEvent.click(screen.getByRole('button', { name: 'Restore it' }))

    await waitFor(() => expect(stub.calls.some(
      (call) => call.url.includes('/api/project/open') && call.method === 'POST',
    )).toBe(true))
  })

  it('goes away once answered, and stays away', () => {
    saveProject(KEPT, 'a project')
    stubFetch([{ url: '/api/project', body: FRESH }])

    const { unmount } = renderWithProviders(<RestoreBanner project={FRESH} />)
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }))

    expect(screen.queryByText(/Carry on where you left off/)).toBeNull()

    // Remounting — switching panels, say — must not ask again.
    unmount()
    renderWithProviders(<RestoreBanner project={FRESH} />)
    expect(screen.queryByText(/Carry on where you left off/)).toBeNull()
  })

  it('can forget the kept project outright', () => {
    saveProject(KEPT, 'a project')
    stubFetch([{ url: '/api/project', body: FRESH }])

    renderWithProviders(<RestoreBanner project={FRESH} />)
    fireEvent.click(screen.getByRole('button', { name: 'Forget it' }))

    expect(window.localStorage.getItem('hydroassist.project.v1')).toBeNull()
  })

  it('restores nothing by itself', () => {
    saveProject(KEPT, 'a project')
    const stub = stubFetch([{ url: '/api/project', body: FRESH }])

    renderWithProviders(<RestoreBanner project={FRESH} />)

    expect(stub.calls.some((call) => call.url.includes('/api/project/open')))
      .toBe(false)
  })
})
