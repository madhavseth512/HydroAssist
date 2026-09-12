/**
 * The workspace shell.
 *
 * Layout follows the decision taken in planning — chat drives, the form
 * confirms: the project on the left, the analysis in the centre, and the
 * consultation in a drawer that is always one click away.
 */
import { AppShell, Badge, Group, Loader, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'

import { api } from './api/client'

export function App() {
  // Creates the session on first contact; every later request carries its cookie.
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="xs">
            <Text size="xl">💧</Text>
            <Title order={4}>HydroAssist</Title>
            <Text size="sm" c="dimmed">
              Pumping test analysis that explains itself
            </Text>
          </Group>
          {session.data && (
            <Badge variant="light" color="blue">
              {session.data.project.summary}
            </Badge>
          )}
        </Group>
      </AppShell.Header>

      <AppShell.Main>
        {session.isPending && <Loader />}
        {session.isError && (
          <Text c="red">{(session.error as Error).message}</Text>
        )}
        {session.data && (
          <Text c="dimmed">
            Connected. The workspace panels land in the next slice.
          </Text>
        )}
      </AppShell.Main>
    </AppShell>
  )
}
