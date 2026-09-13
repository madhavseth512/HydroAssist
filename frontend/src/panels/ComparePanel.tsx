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
import { useComparison, useProject, useRemoveAnalysis } from '../hooks'

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
  const testName = project.data?.tests.find((t) => t.id === testId)?.name
  const comparison = useComparison(testId)
  const remove = useRemoveAnalysis(testId)

  const rows: ComparisonRow[] = comparison.data?.rows ?? []
  const stale = comparison.data?.n_stale ?? 0
  if (rows.length === 0 && stale === 0) return null

  return (
    <Card withBorder padding="md">
      <Stack gap="sm">
        <Title order={5}>Compare saved analyses — {testName}</Title>

        {rows.length > 0 && (
          <Table striped withTableBorder fz="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Rank</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Parameters</Table.Th>
                <Table.Th>T (m²/day)</Table.Th>
                <Table.Th>S</Table.Th>
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
                  <Table.Td>{fixed(row.T_m2_per_day, 2)}</Table.Td>
                  <Table.Td>
                    {row.S === null || row.S === undefined ? '—' : row.S.toExponential(2)}
                  </Table.Td>
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
