/**
 * The workspace shell.
 *
 * Layout follows the decision taken in planning — chat drives, the form
 * confirms: the project on the left, the analysis in the centre, and the
 * consultation in a drawer that is always one click away.
 */
import {
  AppShell, Badge, Burger, Button, Group, Loader, SegmentedControl, Stack, Text,
  Title,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { useRef } from 'react'
import { IconMessageCircle } from '@tabler/icons-react'

import { ChatDrawer } from './panels/ChatDrawer'
import { ComparePanel } from './panels/ComparePanel'
import { DataPanel } from './panels/DataPanel'
import { ResultsPanel } from './panels/ResultsPanel'
import { SetupPanel } from './panels/SetupPanel'
import { ProjectSidebar } from './panels/ProjectSidebar'
import { api } from './api/client'
import { Toolbar } from './components/Toolbar'
import {
  useCurrentResult, useNewProject, useOpenProject, useSession,
} from './hooks'
import { useUnits } from './UnitsContext'
import { UNIT_SYSTEMS } from './units'

export function App() {
  // Creates the session on first contact; every later request carries its cookie.
  //
  // Nothing else may ask the server for state until this has answered. On a
  // first visit there is no cookie yet, so requests sent in parallel would each
  // create a session of their own: the page would show a test from one session
  // and send the upload to another, where that test does not exist.
  const session = useSession()
  const ready = Boolean(session.data)
  const [navOpen, nav] = useDisclosure()
  const [chatOpen, chat] = useDisclosure(false)
  const units = useUnits()
  const openProject = useOpenProject()
  const newProject = useNewProject()
  const result = useCurrentResult(ready)

  // Opening a project is a file the user picks; the hidden input is what the
  // File menu and the toolbar both drive.
  const fileInput = useRef<HTMLInputElement>(null)

  /** Take the user where the assistant offered to take them, once they ask. */
  const goTo = (panel: string) => {
    if (panel === 'project') {
      nav.open()
      return
    }
    document.getElementById(`panel-${panel}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 320, breakpoint: 'sm', collapsed: { mobile: !navOpen } }}
      // Docked, not overlaid: consulting the assistant used to hide the plot
      // being discussed.
      aside={{ width: 400, breakpoint: 'md',
               collapsed: { desktop: !chatOpen, mobile: !chatOpen } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="xs" wrap="nowrap">
            <Burger opened={navOpen} onClick={nav.toggle} hiddenFrom="sm" size="sm" />
            <Text size="xl">💧</Text>
            <Title order={4} visibleFrom="sm">HydroAssist</Title>
            {ready && (
              <Toolbar
                chatOpen={chatOpen}
                hasResult={Boolean(result.data?.success)}
                onNewProject={() => {
                  if (window.confirm('Discard the open project? Anything not '
                                     + 'saved to a file is lost.')) {
                    newProject.mutate(undefined)
                  }
                }}
                onOpenProject={() => fileInput.current?.click()}
                onSaveProject={() => api.saveProject().catch(() => undefined)}
                onDownloadReport={() => api.report().catch(() => undefined)}
                onGoTo={goTo}
                onToggleChat={chat.toggle}
              />
            )}
          </Group>
          <Group gap="xs" wrap="nowrap">
            {/* Which units results are shown in. The literature a user checks
                against is often in gpd/ft, and converting by hand to compare is
                arithmetic the software should absorb. Display only. */}
            <SegmentedControl
              size="xs"
              aria-label="Units"
              value={units.system}
              onChange={(value) => units.setSystem(value as typeof units.system)}
              data={UNIT_SYSTEMS.map((system) => ({
                value: system,
                label: system === 'SI' ? 'SI' : 'US',
              }))}
              visibleFrom="sm"
            />
            {session.data && (
              <Badge variant="light" color="blue" visibleFrom="sm">
                {session.data.project.summary}
              </Badge>
            )}
            <Button size="xs" variant={chatOpen ? 'filled' : 'light'}
                    leftSection={<IconMessageCircle size={16} />}
                    onClick={chat.toggle}>
              Consultant
            </Button>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        {session.isError ? (
          <Text c="red" size="sm">{(session.error as Error).message}</Text>
        ) : ready ? (
          <ProjectSidebar />
        ) : (
          <Loader size="sm" />
        )}
      </AppShell.Navbar>

      <AppShell.Main>
        {ready && (
          <Stack gap="md">
            {/* Anchors the assistant can offer to scroll to. It never moves the
                page itself — the offer is a button the user presses. */}
            <div id="panel-data"><DataPanel /></div>
            <div id="panel-setup"><SetupPanel /></div>
            <div id="panel-results"><ResultsPanel /></div>
            <div id="panel-compare"><ComparePanel /></div>
          </Stack>
        )}
      </AppShell.Main>

      {/* Driven by the File menu and the toolbar; never shown. */}
      <input ref={fileInput} type="file" accept=".hydro,application/json"
             style={{ display: 'none' }} aria-hidden="true" tabIndex={-1}
             onChange={(event) => {
               const file = event.currentTarget.files?.[0]
               if (file) openProject.mutate(file)
               event.currentTarget.value = ''
             }} />

      <AppShell.Aside p="md">
        {ready && <ChatDrawer onGoTo={goTo} />}
      </AppShell.Aside>
    </AppShell>
  )
}
