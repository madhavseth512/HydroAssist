/** Test setup: jest-dom matchers, and the browser APIs jsdom lacks. */
import '@testing-library/jest-dom/vitest'

// Mantine components query these; jsdom implements neither.
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
})

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = window.ResizeObserver ?? (ResizeObserverStub as never)
