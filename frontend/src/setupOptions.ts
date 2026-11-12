/**
 * The fixed choices on the setup form.
 *
 * These mirror `Q_UNITS` and `CLOCK_LABELS` in src/project/workspace.py, which
 * is where a pumping rate is actually converted to SI. They are duplicated here
 * rather than fetched, and tests/unit/test_frontend_constants.py fails if the two
 * lists ever disagree — a unit the server cannot convert must never be offered.
 */

export const Q_UNITS = ['m³/day', 'm³/hour', 'm³/s', 'L/s'] as const

/** m³/s per unit of each pumping rate — `Q_TO_SI` in workspace.py. Used only to
 * show the SI rate beside the form; the server does the conversion that counts. */
export const Q_TO_SI: Record<(typeof Q_UNITS)[number], number> = {
  'm³/day': 1 / 86400,
  'm³/hour': 1 / 3600,
  'm³/s': 1,
  'L/s': 1e-3,
}

/** Concentrations are never converted. The unit labels them, and tells the server
 * how to turn a fitted amplitude into a mass — `TO_GRAMS_PER_M3` in tracer.py, which
 * has no entry for relative concentration because it carries no mass. */
export const CONCENTRATION_UNITS = ['mg/L', 'µg/L', 'ppm', 'ppb', 'relative'] as const

export const TRACER_INJECTIONS = [
  { value: 'pulse', label: 'Pulse' },
  { value: 'continuous', label: 'Continuous' },
] as const

export const RECOVERY_CLOCKS = [
  { value: 'since_stop', label: 'Time since pumping stopped (t′)' },
  { value: 'since_start', label: 'Time since pumping started (t)' },
] as const
