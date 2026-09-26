/**
 * Keeping a project when the session does not.
 *
 * The behaviour worth pinning is the refusals: a store that throws must not take
 * the workspace down with it, and a fresh empty project must never overwrite the
 * copy that a restore offer depends on.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  clearSavedProject, dismissRestore, howLongAgo, readSavedProject,
  restoreWasDismissed, saveProject,
} from './persistence'

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
  vi.restoreAllMocks()
})

const PROJECT = JSON.stringify({ version: 2, title: 'Gridley', tests: [] })

describe('keeping a project in the browser', () => {
  it('reads back what was kept, with its summary', () => {
    expect(saveProject(PROJECT, '1 test · 1 saved analysis')).toBe(true)

    const saved = readSavedProject()
    expect(saved?.json).toBe(PROJECT)
    expect(saved?.summary).toBe('1 test · 1 saved analysis')
    expect(Date.parse(saved!.savedAt)).toBeGreaterThan(0)
  })

  it('has nothing to offer on a first visit', () => {
    expect(readSavedProject()).toBeNull()
  })

  it('refuses a project too large for the store', () => {
    // Browsers give a page about 5 MB; failing the write half way through would
    // leave a truncated project that looks restorable and is not.
    expect(saveProject('x'.repeat(4_000_000), 'huge')).toBe(false)
    expect(readSavedProject()).toBeNull()
  })

  it('survives a store that refuses to write', () => {
    vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError')
    })

    expect(saveProject(PROJECT, 'a project')).toBe(false)
  })

  it('survives a store that refuses to read', () => {
    vi.spyOn(window.localStorage, 'getItem').mockImplementation(() => {
      throw new Error('denied')
    })

    expect(readSavedProject()).toBeNull()
  })

  it('ignores a half-written or corrupted copy', () => {
    window.localStorage.setItem('hydroassist.project.v1', '{"json":')
    expect(readSavedProject()).toBeNull()

    window.localStorage.setItem('hydroassist.project.v1', '{"summary":"no json"}')
    expect(readSavedProject()).toBeNull()
  })

  it('forgets on request', () => {
    saveProject(PROJECT, 'a project')

    clearSavedProject()

    expect(readSavedProject()).toBeNull()
  })
})

describe('the restore offer', () => {
  it('is not dismissed to begin with', () => {
    expect(restoreWasDismissed()).toBe(false)
  })

  it('stays dismissed for this tab once answered', () => {
    dismissRestore()

    expect(restoreWasDismissed()).toBe(true)
  })

  it('does not outlive the tab', () => {
    // sessionStorage, not localStorage: declining once should not silence the
    // offer for every future visit.
    dismissRestore()
    window.sessionStorage.clear()          // what closing the tab does

    expect(restoreWasDismissed()).toBe(false)
  })
})

describe('how long ago it was kept', () => {
  const now = new Date('2026-09-27T12:00:00Z')
  const ago = (iso: string) => howLongAgo(iso, now)

  it('reads recent work as recent', () => {
    expect(ago('2026-09-27T11:59:30Z')).toBe('just now')
  })

  it('counts minutes, then hours, then days', () => {
    expect(ago('2026-09-27T11:40:00Z')).toBe('20 minutes ago')
    expect(ago('2026-09-27T11:00:00Z')).toBe('an hour ago')
    expect(ago('2026-09-27T07:00:00Z')).toBe('5 hours ago')
    expect(ago('2026-09-26T12:00:00Z')).toBe('yesterday')
    expect(ago('2026-09-24T12:00:00Z')).toBe('3 days ago')
  })

  it('says something sensible about a time it cannot read', () => {
    expect(ago('')).toBe('earlier')
    expect(ago('not a date')).toBe('earlier')
  })

  it('never reports the future as a negative age', () => {
    expect(ago('2026-09-27T12:05:00Z')).toBe('just now')
  })
})
