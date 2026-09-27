/**
 * Every fit run in this session, in the order they were run.
 *
 * The project keeps analyses that were *saved* to it, and the comparison panel
 * ranks those by AICc. A fit that was run and not saved existed nowhere: the next
 * run replaced it, on screen and in the conversation. Three methods tried in
 * turn therefore looked like one, and asking the consultant to compare them got
 * an answer about the only one it could still see.
 *
 * This is the log of what was run, which is a different question from what was
 * kept. It is per session and goes when the session does — saving to the project
 * is still what makes an analysis part of the record, and the badge says which
 * ones are.
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
        Kept only while this session lasts. Save an analysis to the project to
        compare it, export it, or keep it.
      </Text>
    </Stack>
  )
}
