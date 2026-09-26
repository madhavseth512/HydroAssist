/** Rendering a panel the way the app renders it: same providers, fresh cache. */
import { UnitsProvider } from '../UnitsContext'
import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement, ReactNode } from 'react'

export function renderWithProviders(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MantineProvider>
        <Notifications />
        <QueryClientProvider client={queryClient}>
          <UnitsProvider>{children}</UnitsProvider>
        </QueryClientProvider>
      </MantineProvider>
    )
  }

  return { queryClient, ...render(ui, { wrapper: Wrapper }) }
}

/** The file input Mantine's Dropzone and FileButton both render.
 *
 * Chosen by what it accepts, never by being the first one on the panel: Step 1
 * offers a data file and a photograph of a field sheet, and those go to
 * different routes. `e2e/workspace.spec.ts` picks its input the same way.
 */
export function fileInput(container: HTMLElement,
                          accept = '.csv'): HTMLInputElement {
  const input = container.querySelector(`input[type="file"][accept*="${accept}"]`)
  if (!input) throw new Error(`no file input accepting ${accept}`)
  return input as HTMLInputElement
}

/** The camera input, which proposes a grid rather than storing readings. */
export function imageInput(container: HTMLElement): HTMLInputElement {
  return fileInput(container, 'image/')
}

export function imageFile(name = 'field-sheet.png'): File {
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], name,
                  { type: 'image/png' })
}

export function csvFile(name = 'readings.csv'): File {
  return new File(['time_s,drawdown_m\n10,0.12\n'], name, { type: 'text/csv' })
}
