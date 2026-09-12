/** Test setup: jest-dom matchers, and the browser APIs jsdom lacks. */
import '@testing-library/jest-dom/vitest'

/**
 * matchMedia, memoised per query — which is what a browser does: the same
 * MediaQueryList comes back for the same query, so a hook that keeps it in
 * state and compares identity does not re-subscribe on every render.
 */
const mediaQueryLists = new Map<string, MediaQueryList>()

function matchMedia(query: string): MediaQueryList {
  const existing = mediaQueryLists.get(query)
  if (existing) return existing

  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  const list = {
    matches: false,
    media: query,
    onchange: null,
    addListener: (cb: (event: MediaQueryListEvent) => void) => listeners.add(cb),
    removeListener: (cb: (event: MediaQueryListEvent) => void) => listeners.delete(cb),
    addEventListener: (_type: string, cb: (event: MediaQueryListEvent) => void) =>
      listeners.add(cb),
    removeEventListener: (_type: string, cb: (event: MediaQueryListEvent) => void) =>
      listeners.delete(cb),
    dispatchEvent: () => false,
  } as unknown as MediaQueryList

  mediaQueryLists.set(query, list)
  return list
}

Object.defineProperty(window, 'matchMedia', { writable: true, value: matchMedia })

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = window.ResizeObserver ?? (ResizeObserverStub as never)
