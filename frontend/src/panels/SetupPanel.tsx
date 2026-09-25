/**
 * Step 2 — the test, and the method to fit it with.
 *
 * The form is driven by the registry: which methods a test type offers, which of
 * them need the pumped well's radii, and what a recovery test needs on top all
 * come from /api/methods. A newly registered method appears here without any
 * change to this file.
 *
 * The method preselected is the one the conversation settled on, when it is
 * offered for this test type.
 */
import {
  Alert,
  Button,
  Card,
  Group,
  NumberInput,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { IconPlayerPlay } from '@tabler/icons-react'
import { useEffect, useState } from 'react'

import type { Method, Setup, Test } from '../api/client'
import { MethodCard } from '../components/MethodCard'
import {
  useAllMethods, useMethods, useProject, useRunAnalysis, useTestTypes,
} from '../hooks'
import { Q_TO_SI, Q_UNITS, RECOVERY_CLOCKS } from '../setupOptions'

const RECOVERY = 'recovery'

function asNumber(value: number | string): number | null {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function SetupPanel() {
  const project = useProject()
  const activeTest: Test | undefined = project.data?.tests.find(
    (test) => test.id === project.data?.active_test_id,
  )

  const [setup, setSetup] = useState<Setup | null>(null)
  // Reload the form when a different test is shown, not on every refetch —
  // otherwise saving a result would overwrite what the user is typing.
  useEffect(() => {
    if (activeTest) setSetup(activeTest.setup)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTest?.id])

  const testType = setup?.test_type ?? 'constant_rate'
  const testTypes = useTestTypes()
  const methods = useMethods(testType)
  const everyMethod = useAllMethods(testType)
  const run = useRunAnalysis()

  const offered = methods.data?.offered ?? []
  const [methodId, setMethodId] = useState<string | null>(null)
  // When the offered list changes, keep the user's choice if it is still
  // offered; otherwise take the server's default (the conversation's method).
  useEffect(() => {
    if (!methods.data) return
    if (methodId && offered.some((m) => m.id === methodId)) return
    setMethodId(offered[methods.data.default_index ?? 0]?.id ?? null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [methods.data])

  const method = offered.find((m) => m.id === methodId)

  if (!setup) {
    return (
      <Card withBorder padding="md">
        <Title order={5}>Step 2 — Test setup</Title>
      </Card>
    )
  }

  const update = (changes: Partial<Setup>) => setSetup({ ...setup, ...changes })
  const hasReadings = Boolean(activeTest?.n_readings)
  const gatedOut = methods.data?.gated_out ?? []
  const gatedReasons = methods.data?.gated_reasons ?? {}
  const allMethods = everyMethod.data?.offered ?? []

  return (
    <Card withBorder padding="md">
      <Stack gap="sm">
        <Title order={5}>Step 2 — Test setup</Title>

        <SegmentedControl
          value={testType}
          onChange={(value) => update({ test_type: value })}
          data={(testTypes.data ?? [{ id: 'constant_rate', label: 'Constant-rate pumping' }])
            .map((type) => ({ value: type.id, label: type.label }))}
        />

        <SimpleGrid cols={{ base: 1, sm: 3 }}>
          <Group gap="xs" align="flex-end" wrap="nowrap">
            <NumberInput
              label="Pumping rate Q"
              value={setup.q_value ?? 500}
              min={0.0001}
              onChange={(value) => update({ q_value: asNumber(value) ?? undefined })}
              style={{ flex: 1 }}
            />
            <Select
              aria-label="Q unit"
              data={[...Q_UNITS]}
              value={setup.q_unit ?? 'm³/day'}
              onChange={(value) => value && update({ q_unit: value })}
              allowDeselect={false}
              w={110}
            />
          </Group>

          <NumberInput
            label="Observation well distance r (m)"
            value={setup.r_m ?? 50}
            min={0.1}
            onChange={(value) => update({ r_m: asNumber(value) ?? undefined })}
          />

        </SimpleGrid>

        <Text size="sm" fw={500}>Analysis method</Text>
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="sm">
          {offered.map((candidate) => (
            <MethodCard key={candidate.id} method={candidate}
                        selected={candidate.id === methodId}
                        onSelect={setMethodId} />
          ))}
          {/* Shown, not hidden: the reason a method does not apply to this
              data set is the part a practitioner learns from. */}
          {gatedOut.map((label) => {
            const excluded = allMethods.find((m: Method) => m.label === label)
            return excluded ? (
              <MethodCard key={label} method={excluded}
                          unavailableBecause={gatedReasons[label] ??
                            'Not offered for this dataset.'} />
            ) : null
          })}
        </SimpleGrid>
        {/* The rate the calculators receive, so a unit mix-up is visible here
            rather than as an implausible transmissivity later. */}
        <Text size="xs" c="dimmed">
          Q = {(Number(setup.q_value ?? 0)
                * (Q_TO_SI[(setup.q_unit ?? 'm³/day') as keyof typeof Q_TO_SI] ?? 0))
                .toExponential(4)} m³/s
        </Text>


        {method?.needs_well_geometry && (
          <>
            <Alert variant="light" color="blue">
              {method.label} is fitted to readings from the pumped well itself, not
              an observation well. Enter the well's dimensions.
            </Alert>
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <NumberInput
                label="Well screen radius r_w (m)"
                value={setup.r_w_m ?? 0.15}
                min={0.01}
                step={0.01}
                decimalScale={3}
                onChange={(value) => update({ r_w_m: asNumber(value) ?? undefined })}
              />
              <NumberInput
                label="Casing radius r_c (m)"
                value={setup.r_c_m ?? 0.1}
                min={0.01}
                step={0.01}
                decimalScale={3}
                onChange={(value) => update({ r_c_m: asNumber(value) ?? undefined })}
              />
            </SimpleGrid>
            <Text size="xs" c="dimmed">
              α = r_w²·S/r_c², the dimensionless wellbore-storage parameter. The
              fit reports it once S has been estimated.
            </Text>
          </>
        )}

        {testType === RECOVERY && (
          <SimpleGrid cols={{ base: 1, sm: 2 }}>
            <NumberInput
              label="Pumping duration before shut-down t_p (hours)"
              value={setup.t_p_hours ?? 24}
              min={0.01}
              step={0.5}
              onChange={(value) => update({ t_p_hours: asNumber(value) })}
            />
            <Stack gap={4}>
              <Text size="sm" fw={500}>The time column counts</Text>
              <SegmentedControl
                value={setup.recovery_clock ?? 'since_stop'}
                onChange={(value) => update({ recovery_clock: value })}
                data={RECOVERY_CLOCKS.map((clock) => ({ ...clock }))}
              />
            </Stack>
          </SimpleGrid>
        )}

        {!hasReadings && (
          <Text size="sm" c="dimmed">Upload readings in Step 1 to run an analysis.</Text>
        )}

        <Button
          leftSection={<IconPlayerPlay size={16} />}
          disabled={!hasReadings || !method}
          loading={run.isPending}
          onClick={() => method && run.mutate({
            method: method.id,
            setup: {
              ...setup,
              // A recovery duration only means something for a recovery test.
              t_p_hours: testType === RECOVERY ? (setup.t_p_hours ?? 24) : null,
            },
          })}
        >
          Run analysis
        </Button>
      </Stack>
    </Card>
  )
}
