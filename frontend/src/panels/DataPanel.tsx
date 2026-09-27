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
  IconCamera,
  IconChartHistogram,
  IconClipboard,
  IconFileUpload,
  IconInfoCircle,
} from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'

import type { PreparedData, Test } from '../api/client'
import { OCRGrid } from '../components/OCRGrid'
import { PlotFigure } from '../components/PlotFigure'
import {
  useActiveTest, usePasteReadings, useReadImage, useReadings, useUploadReadings,
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
  // A photograph does not become readings here. It returns a grid, the user
  // corrects it, and the corrected grid goes out through `paste` — the same
  // route typed numbers take. See components/OCRGrid.tsx.
  const photograph = useReadImage(testId)
  const [pasting, pastingControls] = useDisclosure(false)
  const [pasted, setPasted] = useState('')
  // Emptying the file inputs after a choice, so choosing the same file again is
  // still a change the browser reports.
  const resetFile = useRef<() => void>(null)
  const resetPhoto = useRef<() => void>(null)
  // Whether the reader asked for the whole table rather than the first rows.
  // The count comes from the test, so the request is made once and cached.
  const [allRows, setAllRows] = useState(false)
  // Only ask for readings a test actually has; otherwise every empty test
  // would fetch a guaranteed 404 on render.
  const readings = useReadings(testId, Boolean(activeTest?.n_readings),
                               allRows ? activeTest?.n_readings : undefined)

  // A freshly uploaded file answers with a preview; asking for every row is a
  // separate request, and once it arrives it is what should be shown.
  const prepared = (allRows ? readings.data : undefined)
    ?? upload.data ?? readings.data

  // A mutation's result outlives the test it came from, so opening another test
  // showed the previous one's readings until its own arrived — the same fault as
  // a result outliving its test, one layer up. Cleared when the test changes.
  useEffect(() => {
    upload.reset()
    paste.reset()
    photograph.reset()
    setAllRows(false)
    // Only when the test changes: including the mutations would clear them the
    // moment they succeeded.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [testId])

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
            {/* A field sheet that exists only on paper. `capture` offers the
                camera directly on a phone, which is where the photograph is. */}
            <FileButton resetRef={resetPhoto} accept="image/*" capture="environment"
                        onChange={(file) => {
                          if (file) photograph.mutate(file)
                          resetPhoto.current?.()
                        }}>
              {(props) => (
                <Button {...props} size="xs" variant="subtle"
                        leftSection={<IconCamera size={16} />}
                        loading={photograph.isPending}
                        disabled={!testId}>
                  Photograph of a table
                </Button>
              )}
            </FileButton>
            {/* `resetRef` empties the input after each choice. Without it the
                browser fires no change event when the same file is picked again —
                so re-uploading an edited file, or giving a second test the same
                readings, silently did nothing. */}
            <FileButton resetRef={resetFile}
                        accept=".csv,.txt,.tsv,.dat,.prn,.xlsx,.xlsm,.lev,.mon,.xle,text/csv,text/plain"
                        onChange={(file) => {
                          if (file) upload.mutate(file)
                          resetFile.current?.()
                        }}>
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

        <OCRGrid
          table={photograph.data ?? null}
          opened={Boolean(photograph.data)}
          busy={paste.isPending}
          onClose={() => photograph.reset()}
          onConfirm={(text) => paste.mutate(text, {
            onSuccess: () => photograph.reset(),
          })}
        />

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
            become drawdown, with every conversion listed. A photograph of a
            printed table is read too — into a grid you check first, because a
            misread digit fits as well as the right number.
          </Alert>
        )}

        {(upload.isPending || readings.isFetching) && <Loader size="sm" />}

        {prepared && (
          <PreparedView prepared={prepared} test={activeTest}
                        allRows={allRows} loadingRows={readings.isFetching}
                        onShowAllRows={() => setAllRows(true)} />
        )}
      </Stack>
    </Card>
  )
}

function PreparedView({ prepared, test, allRows, loadingRows, onShowAllRows }: {
  prepared: PreparedData
  test?: Test
  allRows: boolean
  loadingRows: boolean
  onShowAllRows: () => void
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
        <Text size="xs" c="dimmed">
          Read from <strong>{test.readings_source}</strong> — {prepared.n_points}{' '}
          readings. The file itself is not kept; these numbers are.
        </Text>
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

      {/* Open by default. It was collapsed, so the readings a fit was about to
          run on were never in front of the person running it. */}
      <Accordion variant="contained" chevronPosition="left"
                 defaultValue="preview">
        {columns.length > 0 && (
          <Accordion.Item value="preview">
            <Accordion.Control>
              The readings, as they were read
              {allRows
                ? ` (all ${rows.length})`
                : ` (first ${rows.length} of ${prepared.n_points})`}
            </Accordion.Control>
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
              {!allRows && prepared.n_points > rows.length && (
                <Button size="xs" variant="subtle" mt="xs" loading={loadingRows}
                        onClick={onShowAllRows}>
                  Show all {prepared.n_points} readings
                </Button>
              )}
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
