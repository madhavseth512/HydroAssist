/**
 * How wide each of the three columns is, and remembering it.
 *
 * The shell was three fixed widths: a 320 px project tree, whatever was left,
 * and a 400 px chat. A 400 px column is too narrow for the kind of answer the
 * consultant gives — a derivation with an equation in it wraps into a ribbon —
 * and the project tree could not be widened to show a long test name. Neither
 * is a size this file can guess: it depends on the screen and on what the person
 * is doing, so it belongs to them.
 *
 * Widths live here rather than in component state because two things need them:
 * the `AppShell` props that size the columns, and the drag handles that change
 * them. They are kept per browser in `localStorage`, which is the right store
 * for a per-viewer convenience — nothing here is data, and losing it costs a
 * drag.
 */

export interface Widths {
  /** The project tree, on the left. */
  navbar: number
  /** The consultant, on the right. */
  aside: number
}

/** Below the minimum a column cannot show its content; above the maximum it
 * starves the workspace between them. */
export const LIMITS = {
  navbar: { min: 220, max: 560, initial: 320 },
  aside: { min: 320, max: 760, initial: 400 },
} as const

const KEY = 'hydroassist.layout.v1'

export function clamp(value: number, column: keyof Widths): number {
  const { min, max } = LIMITS[column]
  if (!Number.isFinite(value)) return LIMITS[column].initial
  return Math.min(max, Math.max(min, Math.round(value)))
}

export const DEFAULT_WIDTHS: Widths = {
  navbar: LIMITS.navbar.initial,
  aside: LIMITS.aside.initial,
}

/**
 * The widths this browser last used, clamped, falling back to the defaults.
 *
 * Every path through storage is guarded: a private window, cleared site data or
 * a thumbnail capture can make the read throw or return nonsense, and a layout
 * preference must never be the reason the workspace fails to render.
 */
export function readWidths(): Widths {
  try {
    const stored = window.localStorage.getItem(KEY)
    if (!stored) return DEFAULT_WIDTHS
    const parsed = JSON.parse(stored) as Partial<Widths>
    return {
      navbar: clamp(Number(parsed.navbar ?? DEFAULT_WIDTHS.navbar), 'navbar'),
      aside: clamp(Number(parsed.aside ?? DEFAULT_WIDTHS.aside), 'aside'),
    }
  } catch {
    return DEFAULT_WIDTHS
  }
}

export function writeWidths(widths: Widths): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(widths))
  } catch {
    // A browser that refuses storage still resizes; it just forgets.
  }
}
