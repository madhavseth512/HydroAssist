/**
 * Dragging a column wider, and reaching it without a pointer.
 *
 * happy-dom has no layout, so these drive the handle through the events a
 * pointer produces rather than through a simulated mouse path. What is being
 * checked is the arithmetic and the direction — a handle that grew the wrong
 * column on drag would be worse than none.
 */
import { render, screen } from '@testing-library/react'
import { fireEvent } from '@testing-library/dom'
import { describe, expect, it, vi } from 'vitest'

import { ResizeHandle } from './ResizeHandle'
import { LIMITS, clamp, readWidths, writeWidths, DEFAULT_WIDTHS } from '../layout'

function show(props: Partial<Parameters<typeof ResizeHandle>[0]> = {}) {
  const onChange = vi.fn()
  const onCommit = vi.fn()
  render(
    <ResizeHandle
      edge="right-of-column"
      label="Width of the project panel"
      value={320}
      min={220}
      max={560}
      onChange={onChange}
      onCommit={onCommit}
      {...props}
    />,
  )
  return { onChange, onCommit, handle: screen.getByRole('separator') }
}

/** A pointer press, a move, and a release. */
function dragBy(handle: HTMLElement, dx: number) {
  // happy-dom does not implement pointer capture.
  handle.setPointerCapture = vi.fn()
  handle.releasePointerCapture = vi.fn()
  fireEvent.pointerDown(handle, { button: 0, clientX: 100, pointerId: 1 })
  fireEvent.pointerMove(handle, { clientX: 100 + dx, pointerId: 1 })
  fireEvent.pointerUp(handle, { clientX: 100 + dx, pointerId: 1 })
}

describe('ResizeHandle', () => {
  it('announces itself as a separator with its current width', () => {
    const { handle } = show()

    expect(handle).toHaveAttribute('aria-orientation', 'vertical')
    expect(handle).toHaveAttribute('aria-valuenow', '320')
    expect(handle).toHaveAttribute('aria-valuemin', '220')
    expect(handle).toHaveAttribute('aria-valuemax', '560')
  })

  it('a rightward drag widens a column on its left', () => {
    const { onChange, handle } = show()

    dragBy(handle, 60)

    expect(onChange).toHaveBeenLastCalledWith(380)
  })

  it('a rightward drag narrows a column on its right', () => {
    const { onChange, handle } = show({ edge: 'left-of-column', value: 400,
                                        min: 320, max: 760 })

    dragBy(handle, 60)

    expect(onChange).toHaveBeenLastCalledWith(340)
  })

  it('never goes past the minimum', () => {
    const { onChange, handle } = show()

    dragBy(handle, -500)

    expect(onChange).toHaveBeenLastCalledWith(220)
  })

  it('never goes past the maximum', () => {
    const { onChange, handle } = show()

    dragBy(handle, 5000)

    expect(onChange).toHaveBeenLastCalledWith(560)
  })

  it('commits once, when the drag ends', () => {
    const { onCommit, handle } = show()

    dragBy(handle, 40)

    expect(onCommit).toHaveBeenCalledTimes(1)
  })

  it('ignores a drag that did not start on it', () => {
    const { onChange, handle } = show()

    fireEvent.pointerMove(handle, { clientX: 400, pointerId: 1 })

    expect(onChange).not.toHaveBeenCalled()
  })

  it('does not start a drag on a secondary button', () => {
    const { onChange, handle } = show()
    handle.setPointerCapture = vi.fn()

    fireEvent.pointerDown(handle, { button: 2, clientX: 100, pointerId: 1 })
    fireEvent.pointerMove(handle, { clientX: 200, pointerId: 1 })

    expect(onChange).not.toHaveBeenCalled()
  })

  it('can be moved with the arrow keys', () => {
    const { onChange, handle } = show()

    fireEvent.keyDown(handle, { key: 'ArrowRight' })

    expect(onChange).toHaveBeenCalledWith(336)
  })

  it('moves the right-hand column the other way for the same key', () => {
    const { onChange, handle } = show({ edge: 'left-of-column', value: 400,
                                        min: 320, max: 760 })

    fireEvent.keyDown(handle, { key: 'ArrowRight' })

    expect(onChange).toHaveBeenCalledWith(384)
  })

  it('shift moves it further', () => {
    const { onChange, handle } = show()

    fireEvent.keyDown(handle, { key: 'ArrowLeft', shiftKey: true })

    expect(onChange).toHaveBeenCalledWith(256)
  })

  it('Home and End go to the limits', () => {
    const { onChange, handle } = show()

    fireEvent.keyDown(handle, { key: 'End' })
    expect(onChange).toHaveBeenLastCalledWith(560)

    fireEvent.keyDown(handle, { key: 'Home' })
    expect(onChange).toHaveBeenLastCalledWith(220)
  })

  it('leaves other keys to the page', () => {
    const { onChange, handle } = show()

    fireEvent.keyDown(handle, { key: 'a' })

    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('remembering the widths', () => {
  it('falls back to the defaults when nothing is stored', () => {
    window.localStorage.clear()

    expect(readWidths()).toEqual(DEFAULT_WIDTHS)
  })

  it('reads back what was written', () => {
    try {
      writeWidths({ navbar: 400, aside: 520 })

      expect(readWidths()).toEqual({ navbar: 400, aside: 520 })
    } finally {
      window.localStorage.clear()
    }
  })

  it('clamps a stored width that is out of range', () => {
    try {
      window.localStorage.setItem('hydroassist.layout.v1',
                                  JSON.stringify({ navbar: 5000, aside: 10 }))

      expect(readWidths()).toEqual({ navbar: LIMITS.navbar.max,
                                     aside: LIMITS.aside.min })
    } finally {
      window.localStorage.clear()
    }
  })

  it('survives nonsense in storage', () => {
    try {
      window.localStorage.setItem('hydroassist.layout.v1', 'not json')

      expect(readWidths()).toEqual(DEFAULT_WIDTHS)
    } finally {
      window.localStorage.clear()
    }
  })

  it('survives a browser that refuses storage', () => {
    const getItem = vi.spyOn(window.localStorage, 'getItem')
      .mockImplementation(() => { throw new Error('denied') })
    try {
      expect(readWidths()).toEqual(DEFAULT_WIDTHS)
      expect(() => writeWidths(DEFAULT_WIDTHS)).not.toThrow()
    } finally {
      getItem.mockRestore()
    }
  })

  it('clamp keeps a column inside its own limits', () => {
    expect(clamp(0, 'navbar')).toBe(LIMITS.navbar.min)
    expect(clamp(9999, 'aside')).toBe(LIMITS.aside.max)
    expect(clamp(Number.NaN, 'aside')).toBe(LIMITS.aside.initial)
    expect(clamp(401.6, 'aside')).toBe(402)
  })
})
