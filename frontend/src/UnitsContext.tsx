/**
 * Which unit system the workspace shows.
 *
 * Held in one place so a result, a comparison table and a report cannot
 * disagree, and remembered between visits: someone who works in gpd/ft works in
 * gpd/ft every time, and being asked again each session is the kind of small
 * friction that makes software feel careless.
 *
 * Display only — the calculators, the project file and the API stay SI.
 */
import { createContext, useCallback, useContext, useEffect, useState } from 'react'

import type { UnitSystem } from './units'

const STORAGE_KEY = 'hydroassist.units'

interface UnitsValue {
  system: UnitSystem
  setSystem: (system: UnitSystem) => void
}

const UnitsContext = createContext<UnitsValue>({ system: 'SI', setSystem: () => {} })

function remembered(): UnitSystem {
  // Storage can be unavailable — a private window, blocked site data — and a
  // preference is never worth failing to render over.
  try {
    return localStorage.getItem(STORAGE_KEY) === 'US' ? 'US' : 'SI'
  } catch {
    return 'SI'
  }
}

export function UnitsProvider({ children }: { children: React.ReactNode }) {
  const [system, setSystemState] = useState<UnitSystem>(remembered)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, system)
    } catch {
      /* not worth telling the user about */
    }
  }, [system])

  const setSystem = useCallback((next: UnitSystem) => setSystemState(next), [])

  return (
    <UnitsContext.Provider value={{ system, setSystem }}>
      {children}
    </UnitsContext.Provider>
  )
}

export function useUnits(): UnitsValue {
  return useContext(UnitsContext)
}
