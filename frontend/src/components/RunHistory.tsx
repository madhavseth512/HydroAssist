/**
 * Every fit run in this session, in the order they were run.
 *
 * The project keeps analyses that were *saved* to it, and the comparison panel
 * ranks those by AICc. A fit that was run and not saved existed nowhere: the next
 * run replaced it, on screen and in the conversation. Three methods tried in
 * turn therefore looked like one, and asking the consultant to compare them got
 * an answer about the only one it could still see.
 *
 * This is the log of what was run, which is a different question from which ones
 * were saved for comparison. It belongs to the test and travels in the project
 * file, so it survives a reload and switching between tests.
 *
 * Runs against readings that have since been replaced are kept in the file — they
 * happened — but are not listed here, for the same reason a stale analysis is not
 * offered for comparison.
 */
import { Badge, Group, Stack, Text, Tooltip } from '@mantine/core'

import { useAnalysisHistory } from '../hooks'
import { formatQuantity, transmissivity } from '../units'
import { useUnits } from '../UnitsContext'

export function RunHistory() {
  const history = useAnalysisHistory()
  const units = useUnits()
  const runs = history.data ?? []

  if (runs.length === 0) return null

  return (
    <Stack gap={4}>
      <Text size="xs" fw={600} c="dimmed">
        This session: {runs.length} {runs.length === 1 ? 'fit' : 'fits'}
      </Text>
      {runs.map((run, index) => {
        const T = run.T_day == null ? null : transmissivity(run.T_day, units.system)
        return (
          <Group key={`${run.method}-${index}`} gap="xs" wrap="nowrap"
                 justify="space-between">
            <Text size="xs" truncate>{run.method}</Text>
            {run.success && T ? (
              <Tooltip label={run.r_squared == null ? 'no R² recorded'
                : `R² = ${run.r_squared.toFixed(4)}`}>
                <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                  {formatQuantity(T)}
                </Text>
              </Tooltip>
            ) : (
              <Badge size="xs" variant="light" color="red">failed</Badge>
            )}
          </Group>
        )
      })}
      <Text size="xs" c="dimmed">
        Save an analysis to add it to the comparison and the report.
      </Text>
    </Stack>
  )
}
