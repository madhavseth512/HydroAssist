/**
 * Step 2 — the test, and the method to fit it with.
 *
 * The form is driven by the registry: which methods a test type offers, which of
 * them need the pumped well's radii or a pumping rate at all, and what a recovery
 * or tracer test needs on top all come from /api/methods. A newly registered method appears here without any
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
  useActiveTest, useAllMethods, useMethods, useRunAnalysis, useSaveSetup, useTestTypes,
} from '../hooks'
import {
  CONCENTRATION_UNITS, Q_TO_SI, Q_UNITS, RECOVERY_CLOCKS, TRACER_INJECTIONS,
} from '../setupOptions'

const RECOVERY = 'recovery'
const TRACER = 'tracer'

function asNumber(value: number | string): number | null {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function optionalNumber(value: number | string): number | null {
  const parsed = asNumber(value)
  return parsed !== null && parsed > 0 ? parsed : null
}

/** How the tracer entered, and what is known about it. Everything below the first
 * row is optional: each value only unlocks a derived result, so leaving it empty
 * costs that number and not the fit. */
function TracerInputs({ setup, update, pumped }: {
  setup: Setup
  update: (changes: Partial<Setup>) => void
  pumped: boolean
}) {
  const injection = setup.tracer_injection ?? 'pulse'
  const unit = setup.tracer_conc_unit ?? 'mg/L'

  return (
    <Stack gap="xs">
      <SimpleGrid cols={{ base: 1, sm: 3 }}>
        <Stack gap={4}>
          <Text size="sm" fw={500}>Injection</Text>
          <SegmentedControl
            value={injection}
            onChange={(value) => update({ tracer_injection: value })}
            data={TRACER_INJECTIONS.map((option) => ({ ...option }))}
          />
        </Stack>
        <Select
          label="Concentration unit"
          data={[...CONCENTRATION_UNITS]}
          value={unit}
          onChange={(value) => value && update({ tracer_conc_unit: value })}
          allowDeselect={false}
        />
        <NumberInput
          label={`Background concentration (${unit})`}
          value={setup.tracer_background ?? 0}
          min={0}
          onChange={(value) => update({ tracer_background: asNumber(value) ?? 0 })}
        />
      </SimpleGrid>
      <SimpleGrid cols={{ base: 1, sm: 3 }}>
        {injection === 'continuous' && (
          <NumberInput
            label={`Plateau concentration C₀ (${unit}), if known`}
            value={setup.tracer_C0 ?? ''}
            min={0}
            onChange={(value) => update({ tracer_C0: optionalNumber(value) })}
          />
        )}
        {pumped && (
          <NumberInput
            label="Aquifer thickness b (m), for porosity"
            value={setup.tracer_b_m ?? ''}
            min={0}
            onChange={(value) => update({ tracer_b_m: optionalNumber(value) })}
          />
        )}
        {pumped && injection === 'pulse' && (
          <NumberInput
            label="Injected mass (kg), for recovery"
            value={setup.tracer_M_kg ?? ''}
            min={0}
            onChange={(value) => update({ tracer_M_kg: optionalNumber(value) })}
          />
        )}
        {!pumped && (
          <NumberInput
            label="Hydraulic conductivity K (m/day), for porosity"
            value={setup.tracer_K_m_day ?? ''}
            min={0}
            onChange={(value) => update({ tracer_K_m_day: optionalNumber(value) })}
          />
        )}
        {!pumped && (
          <NumberInput
            label="Hydraulic gradient i, for porosity"
            value={setup.tracer_gradient ?? ''}
            min={0}
            step={0.001}
            onChange={(value) => update({ tracer_gradient: optionalNumber(value) })}
          />
        )}
      </SimpleGrid>
      <Text size="xs" c="dimmed">
        Times must count from the moment of injection.
      </Text>
    </Stack>
  )
}

export function SetupPanel() {
  const activeTest: Test | undefined = useActiveTest()


  const [setup, setSetup] = useState<Setup | null>(null)
  // Reload the form when a different test is shown, not on every refetch —
  // otherwise saving a result would overwrite what the user is typing. The test
  // type counts as well: readings of concentration switch the test to Tracer.
  useEffect(() => {
    if (activeTest) setSetup(activeTest.setup)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTest?.id, activeTest?.setup.test_type])

  const testType = setup?.test_type ?? 'constant_rate'
  const testTypes = useTestTypes()
  const methods = useMethods(testType)
  const everyMethod = useAllMethods(testType)
  const run = useRunAnalysis()
  const saveSetup = useSaveSetup(activeTest?.id)

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
  // Stored straight away, because the next upload is read according to it: choosing
  // Tracer and then uploading has to read the column as a concentration.
  const chooseTestType = (value: string) => {
    const next = { ...setup, test_type: value }
    setSetup(next)
    saveSetup.mutate(next)
  }
  const isTracer = testType === TRACER
  // A natural-gradient test has no rate, and until a method is known the rate stays.
  const usesRate = method?.uses_pumping_rate ?? true
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
          onChange={chooseTestType}
          data={(testTypes.data ?? [{ id: 'constant_rate', label: 'Constant-rate pumping' }])
            .map((type) => ({ value: type.id, label: type.label }))}
        />

        <SimpleGrid cols={{ base: 1, sm: 3 }}>
          {usesRate && (
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
          )}

          <NumberInput
            label={isTracer
              ? 'Distance from injection to sampling well (m)'
              : 'Observation well distance r (m)'}
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
        {usesRate && (
          <Text size="xs" c="dimmed">
            Q = {(Number(setup.q_value ?? 0)
                  * (Q_TO_SI[(setup.q_unit ?? 'm³/day') as keyof typeof Q_TO_SI] ?? 0))
                  .toExponential(4)} m³/s
          </Text>
        )}


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

        {isTracer && (
          <TracerInputs setup={setup} update={update} pumped={usesRate} />
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
