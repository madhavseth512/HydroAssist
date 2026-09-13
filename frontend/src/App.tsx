/**
 * The workspace shell.
 *
 * Layout follows the decision taken in planning — chat drives, the form
 * confirms: the project on the left, the analysis in the centre, and the
 * consultation in a drawer that is always one click away.
 */
import { AppShell, Badge, Burger, Button, Group, Loader, Stack, Text, Title } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { IconMessageCircle } from '@tabler/icons-react'

import { ChatDrawer } from './panels/ChatDrawer'
import { ComparePanel } from './panels/ComparePanel'
import { DataPanel } from './panels/DataPanel'
import { ResultsPanel } from './panels/ResultsPanel'
import { SetupPanel } from './panels/SetupPanel'
import { ProjectSidebar } from './panels/ProjectSidebar'
import { useSession } from './hooks'

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

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 320, breakpoint: 'sm', collapsed: { mobile: !navOpen } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="xs" wrap="nowrap">
            <Burger opened={navOpen} onClick={nav.toggle} hiddenFrom="sm" size="sm" />
            <Text size="xl">💧</Text>
            <Title order={4}>HydroAssist</Title>
            <Text size="sm" c="dimmed" visibleFrom="md">
              Pumping test analysis that explains itself
            </Text>
          </Group>
          <Group gap="xs" wrap="nowrap">
            {session.data && (
              <Badge variant="light" color="blue" visibleFrom="sm">
                {session.data.project.summary}
              </Badge>
            )}
            <Button size="xs" leftSection={<IconMessageCircle size={16} />}
                    onClick={chat.open}>
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
            <DataPanel />
            <SetupPanel />
            <ResultsPanel />
            <ComparePanel />
          </Stack>
        )}
      </AppShell.Main>

      {ready && <ChatDrawer opened={chatOpen} onClose={chat.close} />}
    </AppShell>
  )
}
