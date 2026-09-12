/**
 * The workspace shell.
 *
 * Layout follows the decision taken in planning — chat drives, the form
 * confirms: the project on the left, the analysis in the centre, and (in a
 * later slice) the consultation in a drawer that is always one click away.
 */
import { AppShell, Badge, Burger, Group, Stack, Text, Title } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'

import { DataPanel } from './panels/DataPanel'
import { ProjectSidebar } from './panels/ProjectSidebar'
import { useSession } from './hooks'

export function App() {
  // Creates the session on first contact; every later request carries its cookie.
  const session = useSession()
  const [navOpen, nav] = useDisclosure()

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
          {session.data && (
            <Badge variant="light" color="blue">{session.data.project.summary}</Badge>
          )}
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        {session.isError ? (
          <Text c="red" size="sm">{(session.error as Error).message}</Text>
        ) : (
          <ProjectSidebar />
        )}
      </AppShell.Navbar>

      <AppShell.Main>
        <Stack gap="md">
          <DataPanel />
        </Stack>
      </AppShell.Main>
    </AppShell>
  )
}
