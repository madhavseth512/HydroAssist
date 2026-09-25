/** How a measurement is shown: precision, notation, and the unit switch. */
import { describe, expect, it } from 'vitest'

import {
  formatInterval, formatNumber, formatQuantity, length, residual, transmissivity,
} from './units'

describe('converting for display', () => {
  it('reports transmissivity in gpd/ft when asked', () => {
    // Walton (1962) published 11,000 gpd/ft for the Gridley test, which is
    // 136.6 m2/day — the conversion this exists to spare people.
    const converted = transmissivity(136.6, 'US')

    expect(converted.value).toBeCloseTo(11000, -2)
    expect(converted.unit).toBe('gpd/ft')
  })

  it('leaves SI alone', () => {
    expect(transmissivity(215.29, 'SI')).toEqual({ value: 215.29, unit: 'm²/day' })
  })

  it('converts a length', () => {
    expect(length(251.16, 'US').value).toBeCloseTo(824, 0)   // Gridley's r
    expect(length(251.16, 'SI').unit).toBe('m')
  })

  it('shows a residual at a readable size', () => {
    // A 4 mm residual as "0.004 m" reads as nothing at all.
    expect(residual(0.004, 'SI')).toEqual({ value: 4, unit: 'mm' })
    expect(residual(0.004, 'US').unit).toBe('in')
  })

  it('round-trips within a hair', () => {
    const there = transmissivity(215.29, 'US').value
    expect(there * 0.012419).toBeCloseTo(215.29, 1)
  })
})

describe('how many digits', () => {
  it('gives four significant figures by default', () => {
    // The fit's own interval is about +/- 4 m2/day, so a second decimal would
    // be precision the pumping test cannot support.
    expect(formatNumber(215.29)).toBe('215.3')
  })

  it('switches to scientific notation where fixed notation stops helping', () => {
    expect(formatNumber(1.02e-4)).toBe('1.020e-4')
    expect(formatNumber(2.5e6)).toBe('2.500e+6')
  })

  it('keeps small ordinary numbers ordinary', () => {
    expect(formatNumber(0.0512)).toBe('0.0512')
    expect(formatNumber(7)).toBe('7')
  })

  it('drops trailing zeros rather than padding', () => {
    expect(formatNumber(215.3)).toBe('215.3')
    expect(formatNumber(100)).toBe('100')
  })

  it('says nothing rather than zero when there is no value', () => {
    // A missing storativity shown as "0" is a wrong answer, not a blank.
    for (const missing of [null, undefined, NaN, Infinity]) {
      expect(formatNumber(missing as number)).toBe('—')
    }
  })

  it('shows zero as zero', () => {
    expect(formatNumber(0)).toBe('0')
  })
})

describe('a value with its unit', () => {
  it('reads as a measurement', () => {
    expect(formatQuantity(transmissivity(215.29, 'SI'))).toBe('215.3 m²/day')
  })

  it('carries the unit through a conversion', () => {
    expect(formatQuantity(transmissivity(136.6, 'US'), 3)).toContain('gpd/ft')
  })
})

describe('confidence intervals', () => {
  it('are shown in the same units as the value', () => {
    const text = formatInterval([2.487e-3, 2.496e-3],
                                (value) => transmissivity(value * 86400, 'SI'))

    expect(text).toContain('m²/day')
    expect(text).toContain('215')
  })

  it('say so when absent rather than inventing a range', () => {
    expect(formatInterval(null, (v) => transmissivity(v, 'SI'))).toBe('—')
    expect(formatInterval([1], (v) => transmissivity(v, 'SI'))).toBe('—')
  })
})
