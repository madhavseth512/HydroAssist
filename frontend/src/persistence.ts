/**
 * Keeping the work when the page does not.
 *
 * A project lived only in the server's memory. That is fine while a tab is open
 * and fatal either side of it: the free host stops the container after fifteen
 * idle minutes and a reload starts a new session, so a reading uploaded, a test
 * set up and a fit run could all disappear between one cup of tea and the next,
 * with nothing to tell the user it was going to happen.
 *
 * The project file is already the unit of persistence — `GET /api/project/file`
 * produces exactly what the Save button downloads. So this keeps a copy of that
 * file in the browser after every change, and offers it back when a session
 * starts empty. Nothing new is invented, and what is restored is the same thing
 * a user would have got by saving and reopening by hand.
 *
 * It is a convenience, not a store of record: `localStorage` is per browser, is
 * cleared with site data, and can refuse to write at all. Every path here is
 * guarded so that a browser which says no costs the user a restore offer and
 * nothing else — the Save button remains the way to keep work for certain.
 */

const KEY = 'hydroassist.project.v1'
const DISMISSED = 'hydroassist.restore-dismissed'

/** Browsers give a page about 5 MB. A project of 20,000 readings is well under
 * this; something far larger should not be tried and then fail mid-write. */
const MAX_BYTES = 3_000_000

export interface SavedProject {
  /** The project file's contents, exactly as the server wrote them. */
  json: string
  /** When it was kept, as an ISO string. */
  savedAt: string
  /** The one-line summary, so the offer can say what it is offering. */
  summary: string
}

export function saveProject(json: string, summary: string): boolean {
  if (!json || json.length > MAX_BYTES) return false
  try {
    const saved: SavedProject = { json, summary, savedAt: new Date().toISOString() }
    window.localStorage.setItem(KEY, JSON.stringify(saved))
    return true
  } catch {
    // Out of quota, or storage refused. The user still has the Save button.
    return false
  }
}

export function readSavedProject(): SavedProject | null {
  try {
    const stored = window.localStorage.getItem(KEY)
    if (!stored) return null
    const parsed = JSON.parse(stored) as Partial<SavedProject>
    if (typeof parsed.json !== 'string' || !parsed.json) return null
    return {
      json: parsed.json,
      savedAt: typeof parsed.savedAt === 'string' ? parsed.savedAt : '',
      summary: typeof parsed.summary === 'string' ? parsed.summary : 'a project',
    }
  } catch {
    return null
  }
}

export function clearSavedProject(): void {
  try {
    window.localStorage.removeItem(KEY)
  } catch {
    // Nothing to do: it will be overwritten by the next change anyway.
  }
}

/**
 * Whether the restore offer has already been answered in this tab.
 *
 * In `sessionStorage`, not `localStorage`: declining once should not silence the
 * offer for every future visit, only for this one — otherwise a decline made by
 * accident is permanent.
 */
export function restoreWasDismissed(): boolean {
  try {
    return window.sessionStorage.getItem(DISMISSED) === 'yes'
  } catch {
    return false
  }
}

export function dismissRestore(): void {
  try {
    window.sessionStorage.setItem(DISMISSED, 'yes')
  } catch {
    // Then it may be offered again; harmless.
  }
}

/** "2 minutes ago", for the offer. Precision beyond a day is not useful here. */
export function howLongAgo(iso: string, now: Date = new Date()): string {
  const then = Date.parse(iso)
  if (!Number.isFinite(then)) return 'earlier'
  const seconds = Math.max(0, Math.round((now.getTime() - then) / 1000))
  if (seconds < 90) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} minutes ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return hours === 1 ? 'an hour ago' : `${hours} hours ago`
  const days = Math.round(hours / 24)
  return days === 1 ? 'yesterday' : `${days} days ago`
}
