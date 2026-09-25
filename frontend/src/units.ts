/**
 * How a measurement is shown.
 *
 * Two problems, one small module. First, numbers were formatted at each call
 * site with `toFixed` and `toExponential`, so the same quantity could appear
 * with different precision in two panels. Second, the workspace only spoke SI,
 * while the literature a user is checking against — Walton's published
 * transmissivities, for instance — is in US customary units, and converting
 * 11,000 gpd/ft by hand to compare it with 136.6 m²/day is exactly the kind of
 * arithmetic software should absorb.
 *
 * This is display only. Everything computed, stored and sent to the server
 * stays SI; the calculators never see a converted number.
 */

export type UnitSystem = 'SI' | 'US'

export const UNIT_SYSTEMS: UnitSystem[] = ['SI', 'US']

/** Gallons per day per foot, per m²/day. Walton (1962) and most US practice. */
const GPD_FT_PER_M2_DAY = 1 / 0.012419
const FEET_PER_METRE = 1 / 0.3048

export interface Quantity {
  value: number
  unit: string
}

/** Transmissivity, in the system asked for. */
export function transmissivity(m2PerDay: number, system: UnitSystem): Quantity {
  return system === 'US'
    ? { value: m2PerDay * GPD_FT_PER_M2_DAY, unit: 'gpd/ft' }
    : { value: m2PerDay, unit: 'm²/day' }
}

/** A length — a distance to an observation well, a well radius. */
export function length(metres: number, system: UnitSystem): Quantity {
  return system === 'US'
    ? { value: metres * FEET_PER_METRE, unit: 'ft' }
    : { value: metres, unit: 'm' }
}

/** A residual, which is small: millimetres, or inches to three places. */
export function residual(metres: number, system: UnitSystem): Quantity {
  return system === 'US'
    ? { value: metres * FEET_PER_METRE * 12, unit: 'in' }
    : { value: metres * 1000, unit: 'mm' }
}

/**
 * A number a reader can compare at a glance.
 *
 * Fixed notation while the magnitude is readable, scientific outside it —
 * 215.3, but 1.02e-4 rather than 0.000102, because storativity is always read
 * as a power of ten. `significant` is what a fitted parameter deserves: four
 * figures says more precision than a pumping test supports.
 */
export function formatNumber(value: number | null | undefined,
                             significant = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }
  if (value === 0) return '0'

  const magnitude = Math.abs(value)
  if (magnitude >= 1e5 || magnitude < 1e-3) {
    return value.toExponential(Math.max(significant - 1, 0))
  }

  // Significant figures, not decimal places: 17336 gpd/ft to four figures is
  // 17340, and counting decimals would have shown all five. Converting back
  // through Number drops the padding toPrecision adds, so 215.30 reads 215.3.
  return Number(value.toPrecision(significant)).toString()
}

/** A value with its unit, as it should appear on screen. */
export function formatQuantity(quantity: Quantity, significant = 4): string {
  return `${formatNumber(quantity.value, significant)} ${quantity.unit}`
}

/** A confidence interval in the same units as the value it belongs to. */
export function formatInterval(
  interval: number[] | null | undefined,
  convert: (value: number) => Quantity,
  significant = 3,
): string {
  if (!interval || interval.length !== 2) return '—'
  const low = convert(interval[0])
  const high = convert(interval[1])
  return `${formatNumber(low.value, significant)} – ${formatNumber(high.value, significant)} ${low.unit}`
}

/**
 * Digits that line up in a column.
 *
 * Proportional figures make 215.3 and 1.02e-4 impossible to compare down a
 * table, which is most of why a results panel can look like a web page rather
 * than an instrument.
 */
export const NUMERIC_STYLE = { fontVariantNumeric: 'tabular-nums' } as const
