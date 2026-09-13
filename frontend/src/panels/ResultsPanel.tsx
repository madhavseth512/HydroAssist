/**
 * Steps 4 and 5 — what the fit found, and how far to trust it.
 *
 * Every estimated parameter is listed, not only T and S, with the flags a
 * method raises about its own numbers: derived rather than fitted, pinned at a
 * bound, or unreliable by the method's own account (Papadopulos-Cooper on S).
 * A method that estimates no storativity says so rather than printing a zero.
 */
import {
  Accordion,
  Alert,
  Badge,
  Button,
  Card,
  Group,
  List,
  SimpleGrid,
  Slider,
  Stack,
  Table,
  Tabs,
  Text,
  Title,
} from '@mantine/core'
import { IconDeviceFloppy, IconFileTypePdf } from '@tabler/icons-react'
import { useEffect, useState } from 'react'

import { api, type Parameter, type Result } from '../api/client'
import { PlotFigure } from '../components/PlotFigure'
import {
  reportError,
  useCurrentResult,
  useManualMatch,
  useSaveAnalysis,
} from '../hooks'

const SECONDS_PER_DAY = 86400

function formatCi(ci: number[] | null | undefined, scale = 1, digits = 3): string {
  if (!ci || ci.length !== 2) return 'not available'
  return `${(ci[0] * scale).toPrecision(digits)} – ${(ci[1] * scale).toPrecision(digits)}`
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card withBorder padding="sm">
      <Text size="xs" c="dimmed">{label}</Text>
      <Text size="lg" fw={600}>{value}</Text>
      {hint && <Text size="xs" c="dimmed">{hint}</Text>}
    </Card>
  )
}

export function ResultsPanel() {
  const current = useCurrentResult()
  const result = current.data

  // No analysis yet is the normal starting state, not an error.
  if (!result) return null

  if (!result.success) {
    return (
      <Card withBorder padding="md">
        <Stack gap="sm">
          <Title order={5}>Results</Title>
          <Alert color="red" title="The analysis did not converge">
            {result.error_message}
          </Alert>
        </Stack>
      </Card>
    )
  }

  return <SuccessfulResult result={result} />
}

function SuccessfulResult({ result }: { result: Result }) {
  const save = useSaveAnalysis()
  const [savedFor, setSavedFor] = useState<Result | null>(null)
  const saved = savedFor === result

  return (
    <Card withBorder padding="md">
      <Stack gap="md">
        <Group justify="space-between">
          <Title order={5}>Results — {result.method}</Title>
          <Group gap="xs">
            <Button
              size="xs"
              variant={saved ? 'light' : 'filled'}
              leftSection={<IconDeviceFloppy size={16} />}
              disabled={saved}
              loading={save.isPending}
              onClick={() => save.mutate(undefined, { onSuccess: () => setSavedFor(result) })}
            >
              {saved ? 'Saved to project' : 'Save to project'}
            </Button>
            <Button
              size="xs"
              variant="light"
              leftSection={<IconFileTypePdf size={16} />}
              onClick={() => api.report().catch((error) => reportError(error))}
            >
              Download report
            </Button>
          </Group>
        </Group>

        <SimpleGrid cols={{ base: 2, sm: 4 }}>
          <Metric label="T (m²/day)" value={(result.T_day ?? 0).toFixed(2)}
                  hint={`95% CI ${formatCi(result.T_ci, SECONDS_PER_DAY, 4)}`} />
          <Metric label="T (m²/s)" value={(result.T ?? 0).toExponential(3)} />
          {result.estimates_storativity ? (
            <Metric label="S" value={(result.S ?? 0).toExponential(2)}
                    hint={`95% CI ${formatCi(result.S_ci)}`} />
          ) : (
            <Metric label="S" value="not determined"
                    hint="This method does not estimate storativity." />
          )}
          <Metric label="R²" value={(result.r_squared ?? 0).toFixed(4)}
                  hint={`RMSE ${((result.rmse ?? 0) * 1000).toFixed(2)} mm`} />
        </SimpleGrid>

        <ParameterTable parameters={result.parameters ?? []} />

        {(result.validity_notes ?? []).length > 0 && (
          <Accordion variant="contained" chevronPosition="left">
            <Accordion.Item value="notes">
              <Accordion.Control>
                Validity notes ({result.validity_notes!.length})
              </Accordion.Control>
              <Accordion.Panel>
                <List size="sm" spacing="xs">
                  {result.validity_notes!.map((note) => (
                    <List.Item key={note}>{note}</List.Item>
                  ))}
                </List>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>
        )}

        <Tabs defaultValue="fit" keepMounted={false}>
          <Tabs.List>
            <Tabs.Tab value="fit">Fit diagnostic</Tabs.Tab>
            <Tabs.Tab value="manual">Manual matching</Tabs.Tab>
          </Tabs.List>
          <Tabs.Panel value="fit" pt="sm">
            <PlotFigure figure={result.figure} height={520} />
          </Tabs.Panel>
          <Tabs.Panel value="manual" pt="sm">
            <ManualMatching result={result} />
          </Tabs.Panel>
        </Tabs>
      </Stack>
    </Card>
  )
}

function ParameterTable({ parameters }: { parameters: Parameter[] }) {
  if (parameters.length === 0) return null

  const cell = (p: Parameter) =>
    p.display_unit && p.display_unit !== p.unit
      ? `${p.display.toPrecision(4)} ${p.display_unit.replace('^2', '²')}`
      : `${p.value.toPrecision(4)}${p.unit && p.unit !== '-' ? ` ${p.unit}` : ''}`

  const interval = (p: Parameter) => {
    if (!p.ci) return p.derived ? 'derived' : '—'
    const ci = p.display_unit && p.display_unit !== p.unit ? p.display_ci : p.ci
    return formatCi(ci)
  }

  return (
    <Table striped withTableBorder fz="sm">
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Parameter</Table.Th>
          <Table.Th>Value</Table.Th>
          <Table.Th>95% CI</Table.Th>
          <Table.Th>Notes</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {parameters.map((p) => (
          <Table.Tr key={p.name}>
            <Table.Td>{p.symbol}</Table.Td>
            <Table.Td>{cell(p)}</Table.Td>
            <Table.Td>{interval(p)}</Table.Td>
            <Table.Td>
              <Group gap={4}>
                {p.derived && <Badge size="xs" variant="light">derived</Badge>}
                {p.at_bound && (
                  <Badge size="xs" color="yellow" variant="light">at {p.at_bound} bound</Badge>
                )}
                {!p.reliable && (
                  <Badge size="xs" color="red" variant="light" title={p.reliability_note}>
                    low reliability
                  </Badge>
                )}
              </Group>
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  )
}

function ManualMatching({ result }: { result: Result }) {
  const match = useManualMatch()
  const fittedLogT = Math.log10(result.T ?? 1e-3)
  const fittedLogS = Math.log10(Math.max(result.S ?? 1e-4, 1e-7))
  const [logT, setLogT] = useState(fittedLogT)
  const [logS, setLogS] = useState(fittedLogS)

  const request = (nextLogT: number, nextLogS: number) =>
    match.mutate({
      T: 10 ** nextLogT,
      S: result.estimates_storativity ? 10 ** nextLogS : null,
    })

  // Start at the fitted values, so the first curve shown IS the fit.
  useEffect(() => {
    setLogT(fittedLogT)
    setLogS(fittedLogS)
    request(fittedLogT, fittedLogS)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Move the sliders to match the curve by eye; every other parameter stays at
        its fitted value. The figure updates when a slider is released.
      </Text>
      <SimpleGrid cols={{ base: 1, sm: result.estimates_storativity ? 2 : 1 }}>
        <Stack gap={4}>
          <Text size="sm">
            T = {(10 ** logT).toExponential(3)} m²/s ({(10 ** logT * SECONDS_PER_DAY).toFixed(2)} m²/day)
          </Text>
          <Slider
            aria-label="log10 T"
            min={-7} max={0} step={0.05}
            value={logT}
            onChange={setLogT}
            onChangeEnd={(value) => request(value, logS)}
            label={(value) => value.toFixed(2)}
          />
        </Stack>
        {result.estimates_storativity && (
          <Stack gap={4}>
            <Text size="sm">S = {(10 ** logS).toExponential(2)}</Text>
            <Slider
              aria-label="log10 S"
              min={-7} max={-2} step={0.05}
              value={logS}
              onChange={setLogS}
              onChangeEnd={(value) => request(logT, value)}
              label={(value) => value.toFixed(2)}
            />
          </Stack>
        )}
      </SimpleGrid>

      {match.data && (
        <Group gap="xs">
          <Badge variant="light">Manual RMSE {match.data.rmse_mm.toFixed(2)} mm</Badge>
          <Badge variant="light" color="gray">Auto RMSE {match.data.auto_rmse_mm.toFixed(2)} mm</Badge>
        </Group>
      )}
      <PlotFigure figure={match.data?.figure} loading={match.isPending && !match.data}
                  height={480} emptyMessage="Release a slider to draw the manual curve." />
    </Stack>
  )
}
