/**
 * A method, as something to choose between rather than a line in a dropdown.
 *
 * The analogy came from the professor: browsing methods like products. The
 * scientific version has to do the opposite of selling, though — the useful
 * card is the one that says *why a method does not apply to your data*,
 * because that is the part that teaches. So an unavailable method is shown,
 * greyed, with the validator's own sentence underneath, rather than hidden.
 *
 * Everything here already came from the registry: assumptions, what the method
 * estimates, what it needs, and the citation. No new API was required.
 */
import {
  Badge, Card, Group, List, Stack, Text, Tooltip, UnstyledButton,
} from '@mantine/core'
import { IconCheck, IconLock } from '@tabler/icons-react'

import type { Method } from '../api/client'
import { NUMERIC_STYLE } from '../units'

export interface MethodCardProps {
  method: Method
  selected?: boolean
  /** Absent when the method is available; the validator's reason when not. */
  unavailableBecause?: string
  onSelect?: (id: string) => void
  /** Shown once a fit exists, turning the chooser into a comparison. */
  result?: {
    rank?: number
    transmissivity?: string
    storativity?: string
    /** For a method that reports neither T nor S. */
    estimates?: string
    rSquared?: number
    deltaAicc?: number | null
    weight?: number | null
    evidence?: string
    notes?: number
  }
}

export function MethodCard({
  method, selected, unavailableBecause, onSelect, result,
}: MethodCardProps) {
  const available = !unavailableBecause
  const assumptions = method.assumptions ?? []

  return (
    <Card withBorder padding="sm" h="100%"
          style={{
            opacity: available ? 1 : 0.6,
            borderColor: selected ? 'var(--mantine-color-blue-filled)' : undefined,
            borderWidth: selected ? 2 : 1,
          }}>
      <UnstyledButton
        disabled={!available}
        onClick={() => available && onSelect?.(method.id)}
        style={{ cursor: available && onSelect ? 'pointer' : 'default' }}
        aria-label={`${method.label}${available ? '' : ' (not available)'}`}
      >
        <Stack gap={6}>
          <Group justify="space-between" wrap="nowrap" align="flex-start">
            <Text fw={600} size="sm">{method.label}</Text>
            {result?.rank === 1 && <Badge color="teal" size="sm">best fit</Badge>}
            {selected && !result && (
              <IconCheck size={16} color="var(--mantine-color-blue-filled)" />
            )}
            {!available && <IconLock size={14} />}
          </Group>

          <Group gap={6}>
            {method.aquifer_types.map((type) => (
              <Badge key={type} size="xs" variant="light">{type}</Badge>
            ))}
            <Badge size="xs" variant="light" color="gray">
              {method.parameters.length} parameter
              {method.parameters.length === 1 ? '' : 's'}
            </Badge>
            {!method.estimates_storativity && !method.test_types.includes('tracer') && (
              <Badge size="xs" variant="light" color="gray">no S</Badge>
            )}
          </Group>

          {result ? (
            <Stack gap={2}>
              {result.transmissivity && (
                <Text size="xs" style={NUMERIC_STYLE}>T {result.transmissivity}</Text>
              )}
              {result.estimates && (
                <Text size="xs" style={NUMERIC_STYLE}>{result.estimates}</Text>
              )}
              {result.storativity && (
                <Text size="xs" style={NUMERIC_STYLE}>S {result.storativity}</Text>
              )}
              <Text size="xs" c="dimmed" style={NUMERIC_STYLE}>
                r² {result.rSquared?.toFixed(4)}
                {result.deltaAicc !== null && result.deltaAicc !== undefined &&
                  ` · ΔAICc ${result.deltaAicc.toFixed(1)}`}
                {result.weight !== null && result.weight !== undefined &&
                  ` · weight ${result.weight.toFixed(2)}`}
              </Text>
              {result.evidence && (
                <Text size="xs" c="dimmed">{result.evidence}</Text>
              )}
              {/* Ranking says which model fits, not which is right. A method
                  that does not apply can still win on AICc, so the notes
                  travel with the rank rather than behind it. */}
              {!!result.notes && (
                <Text size="xs" c="orange">
                  {result.notes} validity note{result.notes === 1 ? '' : 's'}
                </Text>
              )}
            </Stack>
          ) : (
            <Text size="xs" c="dimmed">{method.ui_help}</Text>
          )}

          {!available && (
            <Text size="xs" c="orange">{unavailableBecause}</Text>
          )}

          {assumptions.length > 0 && available && !result && (
            <Tooltip
              multiline w={320} withArrow position="bottom"
              label={
                <List size="xs" spacing={2}>
                  {assumptions.map((assumption) => (
                    <List.Item key={assumption}>{assumption}</List.Item>
                  ))}
                </List>
              }
            >
              <Text size="xs" c="dimmed" td="underline">
                {assumptions.length} assumptions
              </Text>
            </Tooltip>
          )}
        </Stack>
      </UnstyledButton>
    </Card>
  )
}
