/**
 * The application root.
 *
 * Two providers wrap everything: Mantine for the component styles, and React
 * Query for server state. Server state is deliberately not duplicated into
 * local state — the project, the readings and the result live on the server,
 * and the browser reads them back rather than keeping a second copy that can
 * disagree.
 */
import { MantineProvider, createTheme } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import ReactDOM from 'react-dom/client'

import { App } from './App'

import '@mantine/core/styles.css'
import '@mantine/dropzone/styles.css'
import '@mantine/notifications/styles.css'

const theme = createTheme({
  primaryColor: 'blue',
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  headings: { fontWeight: '600' },
})

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // A fit is expensive and the data behind it rarely changes on its own,
      // so refetching on every window focus would waste the free tier.
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <Notifications position="top-right" />
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </MantineProvider>
  </React.StrictMode>,
)
