/**
 * Step 1 — the field data.
 *
 * The derivative diagnostic sits here, before the method is chosen, because
 * that is the order modern practice works in: read the flow regime off the
 * data, then pick the model that matches it — rather than fitting a model
 * chosen in advance and checking R² afterwards.
 *
 * Drag-and-drop lands in a later slice; a file button keeps the behaviour
 * obvious for now.
 */
import {
  Accordion,
  Alert,
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  List,
  Loader,
  Stack,
  Table,
  Text,
  Textarea,
  Title,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import {
  IconAlertTriangle,
  IconChartHistogram,
  IconClipboard,
  IconFileUpload,
  IconInfoCircle,
} from '@tabler/icons-react'
import { useState } from 'react'

import type { PreparedData, Test } from '../api/client'
import { PlotFigure } from '../components/PlotFigure'
import {
  useActiveTest, usePasteReadings, useReadings, useUploadReadings,
} from '../hooks'

function titleCase(regime: string): string {
  return regime
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export function DataPanel() {
  const activeTest: Test | undefined = useActiveTest()
  const testId = activeTest?.id

  const upload = useUploadReadings(testId)
  const paste = usePasteReadings(testId)
  const [pasting, pastingControls] = useDisclosure(false)
  const [pasted, setPasted] = useState('')
  // Only ask for readings a test actually has; otherwise every empty test
  // would fetch a guaranteed 404 on render.
  const readings = useReadings(testId, Boolean(activeTest?.n_readings))

  const prepared = upload.data ?? readings.data

  return (
    <Card withBorder padding="md">
      <Stack gap="sm">
        <Group justify="space-between" align="center">
          <Title order={5}>Step 1 — Field data</Title>
          <Group gap="xs">
            <Button size="xs" variant="subtle"
                    leftSection={<IconClipboard size={16} />}
                    onClick={pastingControls.toggle}
                    disabled={!testId}>
              Paste data
            </Button>
            {/* Whatever the numbers arrive in. The server sniffs the delimiter
                and reads Excel workbooks, so this list is about the file
                dialog, not about what the reader can do. */}
            <FileButton accept=".csv,.txt,.tsv,.dat,.prn,.xlsx,.xlsm,.lev,.mon,.xle,text/csv,text/plain"
                        onChange={(file) => file && upload.mutate(file)}>
              {(props) => (
                <Button {...props} size="xs" variant="light"
                        leftSection={<IconFileUpload size={16} />}
                        loading={upload.isPending}
                        disabled={!testId}>
                  {activeTest?.n_readings ? 'Replace file' : 'Choose file'}
                </Button>
              )}
            </FileButton>
          </Group>
        </Group>

        {pasting && (
          <Stack gap="xs">
            <Textarea
              label="Paste time and drawdown columns"
              description="Copy two columns from a spreadsheet, headers included. Tabs, commas or spaces all work."
              placeholder={'time_s\tdrawdown_m\n60\t0.12\n120\t0.19'}
              aria-label="Pasted readings"
              autosize
              minRows={4}
              maxRows={10}
              value={pasted}
              onChange={(event) => setPasted(event.currentTarget.value)}
            />
            <Group gap="xs" justify="flex-end">
              <Button size="xs" variant="default" onClick={() => setPasted('')}
                      disabled={!pasted}>
                Clear
              </Button>
              <Button size="xs" loading={paste.isPending} disabled={!pasted.trim()}
                      onClick={() => paste.mutate(pasted, {
                        onSuccess: () => { setPasted(''); pastingControls.close() },
                      })}>
                Use these readings
              </Button>
            </Group>
          </Stack>
        )}

        {!prepared && !upload.isPending && !readings.isFetching && (
          <Alert variant="light" color="blue" icon={<IconInfoCircle size={18} />}>
            Upload or paste a time column and a drawdown column — CSV, tab- or
            space-delimited text, or an Excel sheet. Time in seconds, minutes or
            hours and drawdown in metres or feet are all recognised, and
            converted for you. A logger export (Solinst, Diver, In-Situ) is read
            directly: its timestamps become elapsed time and its water levels
            become drawdown, with every conversion listed.
          </Alert>
        )}

        {(upload.isPending || readings.isFetching) && <Loader size="sm" />}

        {prepared && <PreparedView prepared={prepared} test={activeTest} />}
      </Stack>
    </Card>
  )
}

function PreparedView({ prepared, test }: {
  prepared: PreparedData
  test?: Test
}) {
  const { validation, diagnosis, inspection } = prepared
  const rows = prepared.preview as unknown as Record<string, number | string>[]
  const columns = rows.length > 0 ? Object.keys(rows[0]) : []

  // The server always sends these lists, but a field with a default is optional
  // in the generated schema, so they are read defensively.
  const errors = validation.errors ?? []
  const warnings = validation.warnings ?? []
  const dataNotes = validation.info ?? []
  const regimeNotes = diagnosis?.notes ?? []
  const consistentMethods = diagnosis?.consistent_methods ?? []
  const conversions = prepared.conversions ?? []
  const rowsRemoved = prepared.rows_removed ?? []

  return (
    <Stack gap="sm">
      <Text size="sm" fw={500}>{prepared.summary}</Text>
      {test?.readings_source && (
        <Text size="xs" c="dimmed">From {test.readings_source}</Text>
      )}

      {inspection && (
        <Group gap="xs">
          <Badge variant="light">time: {inspection.time_col} ({inspection.time_unit})</Badge>
          <Badge variant="light">
            drawdown: {(inspection.drawdown_cols ?? []).join(', ')} ({inspection.drawdown_unit})
          </Badge>
          <Badge variant="light" color="gray">
            detected with {Math.round(inspection.confidence * 100)}% confidence
          </Badge>
        </Group>
      )}

      {/* What column detection was unsure about: negative drawdowns, a unit it
          guessed, a column it could not place. */}
      {(inspection?.issues ?? []).map((issue) => (
        <Alert key={issue} color="yellow" variant="light">{issue}</Alert>
      ))}
      {errors.map((message) => (
        <Alert key={message} color="red" icon={<IconAlertTriangle size={18} />}>
          {message}
        </Alert>
      ))}
      {warnings.map((message) => (
        <Alert key={message} color="yellow" variant="light">{message}</Alert>
      ))}
      {!validation.is_usable && (
        <Alert color="red">
          This dataset cannot be analysed as it stands — resolve the problems
          above before running a fit.
        </Alert>
      )}

      {diagnosis && (
        <Card withBorder padding="sm" bg="var(--mantine-color-body)">
          <Group gap="xs" mb="xs">
            <IconChartHistogram size={18} />
            <Text fw={600}>{titleCase(diagnosis.regime)}</Text>
          </Group>
          <Text size="sm">{diagnosis.evidence}</Text>
          {regimeNotes.length > 0 && (
            <List size="xs" mt="xs" c="dimmed">
              {regimeNotes.map((note) => <List.Item key={note}>{note}</List.Item>)}
            </List>
          )}
          {consistentMethods.length > 0 && (
            <Text size="xs" mt="xs">
              Consistent with: {consistentMethods.join(', ')}
            </Text>
          )}
          <PlotFigure figure={prepared.derivative_figure} height={360}
                      emptyMessage="The derivative could not be drawn for this dataset." />
        </Card>
      )}
      {!diagnosis && prepared.diagnosis_error && (
        <Alert color="gray" variant="light">
          The flow regime could not be read from this dataset:{' '}
          {prepared.diagnosis_error}
        </Alert>
      )}

      <Accordion variant="contained" chevronPosition="left">
        {columns.length > 0 && (
          <Accordion.Item value="preview">
            <Accordion.Control>Preview the formatted data</Accordion.Control>
            <Accordion.Panel>
              <Table striped withTableBorder fz="xs">
                <Table.Thead>
                  <Table.Tr>
                    {columns.map((column) => (
                      <Table.Th key={column}>{column}</Table.Th>
                    ))}
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {rows.map((row, index) => (
                    <Table.Tr key={index}>
                      {columns.map((column) => (
                        <Table.Td key={column}>{String(row[column])}</Table.Td>
                      ))}
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Accordion.Panel>
          </Accordion.Item>
        )}
        {conversions.length > 0 && (
          <Accordion.Item value="conversions">
            <Accordion.Control>
              Conversions applied ({conversions.length})
            </Accordion.Control>
            <Accordion.Panel>
              <List size="xs">
                {conversions.map((item) => (
                  <List.Item key={item}>{item}</List.Item>
                ))}
              </List>
            </Accordion.Panel>
          </Accordion.Item>
        )}
        {rowsRemoved.length > 0 && (
          <Accordion.Item value="removed">
            <Accordion.Control>
              Rows removed ({rowsRemoved.length})
            </Accordion.Control>
            <Accordion.Panel>
              <List size="xs">
                {rowsRemoved.map((item) => (
                  <List.Item key={item}>{item}</List.Item>
                ))}
              </List>
            </Accordion.Panel>
          </Accordion.Item>
        )}
        {dataNotes.length > 0 && (
          <Accordion.Item value="notes">
            <Accordion.Control>
              Data notes ({dataNotes.length})
            </Accordion.Control>
            <Accordion.Panel>
              <List size="xs">
                {dataNotes.map((item) => (
                  <List.Item key={item}>{item}</List.Item>
                ))}
              </List>
            </Accordion.Panel>
          </Accordion.Item>
        )}
      </Accordion>
    </Stack>
  )
}
