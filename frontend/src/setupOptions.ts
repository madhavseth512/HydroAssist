/**
 * The fixed choices on the setup form.
 *
 * These mirror `Q_UNITS` and `CLOCK_LABELS` in src/project/workspace.py, which
 * is where a pumping rate is actually converted to SI. They are duplicated here
 * rather than fetched, and tests/unit/test_frontend_constants.py fails if the two
 * lists ever disagree — a unit the server cannot convert must never be offered.
 */

export const Q_UNITS = ['m³/day', 'm³/hour', 'm³/s', 'L/s'] as const

export const RECOVERY_CLOCKS = [
  { value: 'since_stop', label: 'Time since pumping stopped (t′)' },
  { value: 'since_start', label: 'Time since pumping started (t)' },
] as const
