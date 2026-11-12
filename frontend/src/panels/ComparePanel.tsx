/**
 * Saved analyses for the active test, ranked by AICc.
 *
 * Not by R²: R² can only improve as parameters are added, so ranking on it
 * would always favour the most complex method. AICc charges for each extra
 * parameter, and ΔAICc says whether the data can tell two methods apart at all.
 */
import { ActionIcon, Badge, Card, Stack, Table, Text, Title, Tooltip } from '@mantine/core'
import { IconTrash } from '@tabler/icons-react'

import type { ComparisonRow } from '../api/client'
import { MethodCard } from '../components/MethodCard'
import { useAllMethods, useComparison, useProject, useRemoveAnalysis } from '../hooks'
import { useUnits } from '../UnitsContext'
import { NUMERIC_STYLE, formatNumber, formatQuantity, transmissivity } from '../units'

const EVIDENCE_COLOR: Record<string, string> = {
  best: 'teal',
  'substantial support': 'blue',
  'less support': 'yellow',
  'essentially no support': 'red',
}

function fixed(value: number | null | undefined, digits: number): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

export function ComparePanel() {
  const project = useProject()
  const testId = project.data?.active_test_id
  const activeTest = project.data?.tests.find((t) => t.id === testId)
  const testName = activeTest?.name
  const comparison = useComparison(testId)
  const remove = useRemoveAnalysis(testId)

  const { system } = useUnits()
  const everyMethod = useAllMethods(activeTest?.setup?.test_type ?? 'constant_rate')

  const rows: ComparisonRow[] = comparison.data?.rows ?? []
  const best = rows.find((row) => row.rank === 1)
  const bestMethod = (everyMethod.data?.offered ?? [])
    .find((candidate) => candidate.label === best?.method)
  const stale = comparison.data?.n_stale ?? 0
  if (rows.length === 0 && stale === 0) return null
  // A tracer test has no T or S to line up, so its estimates take those columns.
  const isTracer = rows.length > 0 && rows.every((row) => Boolean(row.estimates))

  return (
    <Card withBorder padding="md">
      <Stack gap="sm">
        <Title order={5}>Compare saved analyses — {testName}</Title>

        {/* The winner as a card, which is how the choice is made; the table
            below is how the numbers are actually compared, and a table is the
            right instrument for that. */}
        {best && bestMethod && (
          <MethodCard
            method={bestMethod}
            result={{
              rank: 1,
              transmissivity: best.estimates ? undefined : formatQuantity(
                transmissivity(best.T_m2_per_day ?? 0, system)),
              estimates: best.estimates || undefined,
              storativity: best.S === null || best.S === undefined
                ? undefined : best.S.toExponential(2),
              rSquared: best.r_squared ?? undefined,
              deltaAicc: best.delta_aicc,
              weight: best.akaike_weight,
              evidence: best.evidence ?? undefined,
            }}
          />
        )}

        {rows.length > 0 && (
          <Table striped withTableBorder fz="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Rank</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Parameters</Table.Th>
                {isTracer ? (
                  <Table.Th>Estimates</Table.Th>
                ) : (
                  <>
                    <Table.Th>T ({transmissivity(0, system).unit})</Table.Th>
                    <Table.Th>S</Table.Th>
                  </>
                )}
                <Table.Th>R²</Table.Th>
                <Table.Th>AICc</Table.Th>
                <Table.Th>ΔAICc</Table.Th>
                <Table.Th>Weight</Table.Th>
                <Table.Th>Evidence</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((row) => (
                <Table.Tr key={row.analysis_id}>
                  <Table.Td>{row.rank}</Table.Td>
                  <Table.Td>{row.method}</Table.Td>
                  <Table.Td>{row.n_params ?? '—'}</Table.Td>
                  {isTracer ? (
                    <Table.Td style={NUMERIC_STYLE}>{row.estimates}</Table.Td>
                  ) : (
                    <>
                      <Table.Td style={NUMERIC_STYLE}>
                        {formatNumber(transmissivity(row.T_m2_per_day ?? 0, system).value)}
                      </Table.Td>
                      <Table.Td>
                        {row.S === null || row.S === undefined ? '—' : row.S.toExponential(2)}
                      </Table.Td>
                    </>
                  )}
                  <Table.Td>{fixed(row.r_squared, 4)}</Table.Td>
                  <Table.Td>{fixed(row.aicc, 1)}</Table.Td>
                  <Table.Td>{fixed(row.delta_aicc, 1)}</Table.Td>
                  <Table.Td>{fixed(row.akaike_weight, 2)}</Table.Td>
                  <Table.Td>
                    <Badge variant="light" color={EVIDENCE_COLOR[row.evidence ?? ''] ?? 'gray'}>
                      {row.evidence}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Tooltip label="Remove from the project">
                      <ActionIcon variant="subtle" color="red"
                                  aria-label={`Remove ${row.method}`}
                                  onClick={() => remove.mutate(row.analysis_id)}>
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Tooltip>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}

        <Text size="xs" c="dimmed">
          Ranked by AICc, which charges for every extra parameter — not by R², which
          always favours the most complex method. ΔAICc under 2 means the data
          cannot tell two methods apart; over 10 is decisive. Weight is the relative
          likelihood that each method is the best of this set.
        </Text>
        {stale > 0 && (
          <Text size="xs" c="dimmed">
            {stale} earlier analysis(es) were run on different readings and are left
            out of this comparison.
          </Text>
        )}
      </Stack>
    </Card>
  )
}
