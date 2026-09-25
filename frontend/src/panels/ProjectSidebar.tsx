/**
 * The project: what a hydrogeologist actually organises work in.
 *
 * A project holds several tests, each with its own readings, setup and saved
 * analyses. It lives on the server for the session and is kept by downloading
 * it — there are no accounts, and a file can be emailed to a colleague or
 * attached to a report.
 */
import {
  ActionIcon,
  Badge,
  Button,
  Collapse,
  Divider,
  Group,
  Loader,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
  UnstyledButton,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  IconChevronDown, IconChevronRight, IconFlask, IconPlus,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'

import { type ProjectInfo, type Test } from '../api/client'
import {
  useActivateTest,
  useAddTest,
  useNewProject,
  useProject,
  useUpdateInfo,
  useUpdateTest,
} from '../hooks'

export function ProjectSidebar() {
  const project = useProject()
  const newProject = useNewProject()
  const addTest = useAddTest()
  const activateTest = useActivateTest()
  const [detailsOpen, details] = useDisclosure(false)

  if (project.isPending) return <Loader size="sm" />
  if (!project.data) {
    return <Text c="red">{(project.error as Error)?.message ?? 'No project.'}</Text>
  }

  const { info, tests, active_test_id, summary } = project.data
  const activeTest = tests.find((test) => test.id === active_test_id)

  return (
    <Stack gap="sm">
      <div>
        <Title order={5}>{info.title}</Title>
        <Text size="xs" c="dimmed">{summary}</Text>
      </div>

      {/* Open, Save and New live in the File menu and the toolbar. They were
          here as well, which made "Save" ambiguous on the page and left two
          places to change when the behaviour changes. */}
      <Divider label="Tests" labelPosition="center" />

      <Group gap="xs" justify="space-between" wrap="nowrap">
        <Text size="xs" c="dimmed">
          {tests.length} test{tests.length === 1 ? '' : 's'}
        </Text>
        <Tooltip label="Add a test">
          <ActionIcon variant="light" size="sm" aria-label="Add test"
                      loading={addTest.isPending}
                      onClick={() => addTest.mutate(undefined)}>
            <IconPlus size={14} />
          </ActionIcon>
        </Tooltip>
      </Group>

      <Stack gap={2} role="tree" aria-label="Tests">
        {tests.map((test) => {
          const open = test.id === active_test_id
          return (
            <UnstyledButton
              key={test.id}
              role="treeitem"
              aria-selected={open}
              aria-label={`${test.name}${open ? ' (open)' : ''}`}
              onClick={() => !open && activateTest.mutate(test.id)}
              p={6}
              style={{
                borderRadius: 4,
                background: open ? 'var(--mantine-color-blue-light)' : undefined,
              }}
            >
              <Group gap={6} wrap="nowrap">
                <IconFlask size={14} />
                <Text size="sm" fw={open ? 600 : 400} style={{ flex: 1 }}>
                  {test.name}
                </Text>
              </Group>
              {/* What the test holds, which is what a tree is for: the shape of
                  the work, visible without opening each one. */}
              <Text size="xs" c="dimmed" pl={20}>
                {test.n_readings ? `${test.n_readings} readings` : 'no readings'}
                {test.n_analyses ? ` · ${test.n_analyses} saved` : ''}
                {test.n_stale_analyses ? ` · ${test.n_stale_analyses} stale` : ''}
              </Text>
            </UnstyledButton>
          )
        })}
      </Stack>

      {activeTest && (
        <Group gap="xs">
          <Badge variant="light" color={activeTest.n_readings ? 'teal' : 'gray'}>
            {activeTest.n_readings
              ? `${activeTest.n_readings} readings`
              : 'no readings'}
          </Badge>
          {activeTest.n_analyses > 0 && (
            <Badge variant="light">{activeTest.n_analyses} saved</Badge>
          )}
        </Group>
      )}

      <Button
        variant="subtle"
        size="xs"
        justify="space-between"
        rightSection={detailsOpen ? <IconChevronDown size={14} />
                                  : <IconChevronRight size={14} />}
        onClick={details.toggle}
      >
        Project and test details
      </Button>

      <Collapse in={detailsOpen}>
        <ProjectDetails info={info} />
        {activeTest && <TestDetails test={activeTest} />}
      </Collapse>

      <Divider />

      <Button
        variant="subtle"
        color="red"
        size="xs"
        loading={newProject.isPending}
        onClick={() => {
          // Anything not downloaded is lost, so this asks first.
          if (window.confirm(
            'Start a new project? Anything not saved to a file will be lost.',
          )) {
            newProject.mutate(undefined)
          }
        }}
      >
        New project
      </Button>
    </Stack>
  )
}

function ProjectDetails({ info }: { info: ProjectInfo }) {
  const updateInfo = useUpdateInfo()
  const [draft, setDraft] = useState<ProjectInfo>(info)

  // Follow the server when the project is replaced under us (open, new).
  useEffect(() => setDraft(info), [info])

  const field = (key: keyof ProjectInfo, label: string, placeholder?: string) => (
    <TextInput
      size="xs"
      label={label}
      placeholder={placeholder}
      value={draft[key] ?? ''}
      onChange={(event) =>
        setDraft({ ...draft, [key]: event.currentTarget.value })}
    />
  )

  return (
    <Stack gap="xs" mt="xs">
      {field('title', 'Project title')}
      {field('client', 'Client')}
      {field('project_number', 'Project number')}
      {field('location', 'Location')}
      {field('test_date', 'Test date', 'YYYY-MM-DD')}
      {field('analyst', 'Analyst')}
      <Textarea size="xs" label="Notes" autosize minRows={2}
                value={draft.notes ?? ''}
                onChange={(event) =>
                  setDraft({ ...draft, notes: event.currentTarget.value })} />
      <Button
        size="xs"
        loading={updateInfo.isPending}
        onClick={() =>
          updateInfo.mutate(draft, {
            onSuccess: () => notifications.show({
              color: 'teal', title: 'Saved',
              message: 'Project details updated. They appear on the report.',
            }),
          })}
      >
        Save details
      </Button>
    </Stack>
  )
}

type TestField = 'name' | 'pumping_well' | 'observation_well'

/** The active test's name and the wells it was run between — which the report
 * and the project file carry, so a colleague can tell the tests apart. */
function TestDetails({ test }: { test: Test }) {
  const updateTest = useUpdateTest()
  const saved = {
    name: test.name,
    pumping_well: test.pumping_well ?? '',
    observation_well: test.observation_well ?? '',
  }
  const [draft, setDraft] = useState(saved)

  useEffect(() => setDraft(saved),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [test.id, test.name, test.pumping_well, test.observation_well])

  const commit = (field: TestField) => {
    const value = draft[field].trim()
    if (value === saved[field]) return
    // A test must keep a name; the wells may be left blank.
    if (field === 'name' && !value) {
      setDraft({ ...draft, name: saved.name })
      return
    }
    updateTest.mutate({ testId: test.id, [field]: value })
  }

  const field = (key: TestField, label: string) => (
    <TextInput
      mt="xs"
      size="xs"
      label={label}
      value={draft[key]}
      onChange={(event) => setDraft({ ...draft, [key]: event.currentTarget.value })}
      onBlur={() => commit(key)}
    />
  )

  return (
    <>
      {field('name', 'Test name')}
      {field('pumping_well', 'Pumping well')}
      {field('observation_well', 'Observation well')}
    </>
  )
}
