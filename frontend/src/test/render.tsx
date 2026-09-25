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

/** The file input Mantine's Dropzone and FileButton both render. */
export function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]')
  if (!input) throw new Error('no file input rendered')
  return input as HTMLInputElement
}

export function csvFile(name = 'readings.csv'): File {
  return new File(['time_s,drawdown_m\n10,0.12\n'], name, { type: 'text/csv' })
}
