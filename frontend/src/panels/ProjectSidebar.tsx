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
  FileButton,
  Group,
  Loader,
  Select,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import { IconChevronDown, IconChevronRight, IconPlus } from '@tabler/icons-react'
import { useEffect, useState } from 'react'

import { api, type ProjectInfo } from '../api/client'
import {
  reportError,
  useActivateTest,
  useAddTest,
  useNewProject,
  useOpenProject,
  useProject,
  useUpdateInfo,
  useUpdateTest,
} from '../hooks'

export function ProjectSidebar() {
  const project = useProject()
  const openProject = useOpenProject()
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

      <Group gap="xs" grow>
        <FileButton
          accept="application/json,.json"
          onChange={(file) => file && openProject.mutate(file)}
        >
          {(props) => (
            <Button {...props} variant="light" size="xs"
                    loading={openProject.isPending}>
              Open
            </Button>
          )}
        </FileButton>
        <Tooltip label="Download the project as a file you can reopen later">
          <Button
            variant="light"
            size="xs"
            onClick={() => api.saveProject().catch((e) => reportError(e))}
          >
            Save
          </Button>
        </Tooltip>
      </Group>

      <Divider label="Tests" labelPosition="center" />

      <Group gap="xs" align="flex-end" wrap="nowrap">
        <Select
          label="Active test"
          data={tests.map((test) => ({ value: test.id, label: test.name }))}
          value={active_test_id}
          onChange={(value) => value && value !== active_test_id
            && activateTest.mutate(value)}
          allowDeselect={false}
          size="xs"
          style={{ flex: 1 }}
        />
        <Tooltip label="Add a test">
          <ActionIcon variant="light" size="lg" aria-label="Add test"
                      loading={addTest.isPending}
                      onClick={() => addTest.mutate(undefined)}>
            <IconPlus size={16} />
          </ActionIcon>
        </Tooltip>
      </Group>

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
        {activeTest && (
          <TestName testId={activeTest.id} name={activeTest.name} />
        )}
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

function TestName({ testId, name }: { testId: string; name: string }) {
  const updateTest = useUpdateTest()
  const [draft, setDraft] = useState(name)

  useEffect(() => setDraft(name), [name, testId])

  return (
    <TextInput
      mt="xs"
      size="xs"
      label="Test name"
      value={draft}
      onChange={(event) => setDraft(event.currentTarget.value)}
      onBlur={() => {
        const trimmed = draft.trim()
        if (trimmed && trimmed !== name) {
          updateTest.mutate({ testId, name: trimmed })
        }
      }}
    />
  )
}
