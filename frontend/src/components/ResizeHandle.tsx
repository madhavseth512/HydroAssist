/**
 * The divider between two columns, which can be dragged.
 *
 * Hand-written rather than a splitter library, for two reasons. The widths are
 * already `AppShell` props — the shell owns the geometry, so all a handle has to
 * do is report a new number — and a library would bring a second layout engine
 * to argue with the one in place.
 *
 * It is a real `separator` with a value, so it works from the keyboard as well
 * as the pointer: someone who cannot drag can still widen the consultant with
 * the arrow keys. `setPointerCapture` keeps the drag alive when the pointer
 * outruns the 6 px strip, which is what makes a thin handle usable at all.
 */
import { useRef } from 'react'

export interface ResizeHandleProps {
  /** Current width of the column being sized, in pixels. */
  value: number
  min: number
  max: number
  /** Which edge this handle sits on: the left column's right edge, or the
   * right column's left edge. It decides which way a drag grows the column. */
  edge: 'right-of-column' | 'left-of-column'
  label: string
  onChange: (width: number) => void
  /** Called once when a drag or key-press run ends, for persisting. */
  onCommit?: (width: number) => void
}

/** How far one arrow key moves the divider. Shift multiplies it. */
const STEP = 16

export function ResizeHandle({
  value, min, max, edge, label, onChange, onCommit,
}: ResizeHandleProps) {
  const drag = useRef<{ x: number; width: number } | null>(null)

  const clamp = (width: number) => Math.min(max, Math.max(min, Math.round(width)))

  function begin(event: React.PointerEvent<HTMLDivElement>) {
    // Only the primary button drags; a right-click must not start one.
    if (event.button !== 0) return
    drag.current = { x: event.clientX, width: value }
    event.currentTarget.setPointerCapture(event.pointerId)
    // Without this, dragging over the workspace selects its text as it goes.
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'col-resize'
  }

  function move(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag.current) return
    const travelled = event.clientX - drag.current.x
    const delta = edge === 'right-of-column' ? travelled : -travelled
    onChange(clamp(drag.current.width + delta))
  }

  function end(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag.current) return
    drag.current = null
    try {
      event.currentTarget.releasePointerCapture(event.pointerId)
    } catch {
      // The pointer may already be gone; the drag is over either way.
    }
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
    onCommit?.(value)
  }

  function key(event: React.KeyboardEvent<HTMLDivElement>) {
    const step = event.shiftKey ? STEP * 4 : STEP
    // Arrow keys move the divider itself, so "left" always means left on
    // screen; which column grows by it depends on the edge.
    const towards = edge === 'right-of-column' ? 1 : -1
    let next: number | null = null
    if (event.key === 'ArrowRight') next = value + step * towards
    else if (event.key === 'ArrowLeft') next = value - step * towards
    else if (event.key === 'Home') next = min
    else if (event.key === 'End') next = max
    if (next === null) return
    event.preventDefault()
    const width = clamp(next)
    onChange(width)
    onCommit?.(width)
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={value}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      onPointerDown={begin}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      onKeyDown={key}
      onDoubleClick={() => { onChange(min); onCommit?.(min) }}
      style={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        [edge === 'right-of-column' ? 'right' : 'left']: -3,
        width: 7,
        cursor: 'col-resize',
        // Invisible until pointed at: a permanent grey bar between every pair of
        // columns is three vertical lines of furniture the eye has to ignore.
        background: 'transparent',
        zIndex: 200,
        touchAction: 'none',
      }}
      // The hit area is 7 px; the visible hint is a 1 px line that appears on
      // hover and focus, drawn by the ::after rule in index.css.
      data-resize-handle
    />
  )
}
