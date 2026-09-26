/**
 * A photographed field sheet, as a grid to confirm.
 *
 * This component exists because of one asymmetry. Every other way readings
 * arrive either works or fails loudly: a malformed CSV is rejected, a logger
 * file with no data section says so. OCR never fails — it proposes, and a `3.5`
 * read as `8.5` fits the type curve just as well, reports a transmissivity out
 * by a factor of two, and gives nobody anywhere a reason to look twice.
 *
 * So the grid is not a preview to skim past. It is the step where the numbers
 * become the user's rather than Tesseract's:
 *
 * - every cell is editable, and starts as the characters OCR saw, not as a
 *   number this has decided to believe;
 * - the cells Tesseract was unsure of are marked, because those are the ones
 *   worth reading against the sheet first;
 * - the columns must be named before the readings can be used. A photograph has
 *   no header row, and nothing here can tell minutes from seconds. Guessing
 *   would be the same silent error one level up.
 *
 * On confirmation the grid is serialised as tab-delimited text and sent through
 * the ordinary paste route, which is the same door typed numbers come through.
 */
import {
  Alert,
  Badge,
  Button,
  Group,
  List,
  Modal,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core'
import { IconAlertTriangle, IconTrash } from '@tabler/icons-react'
import { useEffect, useMemo, useState } from 'react'

import type { OCRTable } from '../api/client'

/** What a column can be called.
 *
 * These are the names `CSVInspector` recognises without having to guess, so
 * naming a column here settles its unit deterministically — which is the point.
 */
export const COLUMN_ROLES = [
  { value: 'time_s', label: 'Time (seconds)' },
  { value: 'time_min', label: 'Time (minutes)' },
  { value: 'time_h', label: 'Time (hours)' },
  { value: 'drawdown_m', label: 'Drawdown (metres)' },
  { value: 'drawdown_ft', label: 'Drawdown (feet)' },
  { value: '', label: 'Leave this column out' },
] as const

const IGNORED = ''

export interface OCRGridProps {
  table: OCRTable | null
  opened: boolean
  onClose: () => void
  /** Called with tab-delimited text, headers included, for the paste route. */
  onConfirm: (text: string) => void
  busy?: boolean
}

export function OCRGrid({ table, opened, onClose, onConfirm, busy }: OCRGridProps) {
  const columns = table?.columns ?? 0
  // The text of each cell, editable. Held here rather than in the query cache:
  // these are the user's corrections, and re-reading the photograph must not
  // quietly discard them.
  const [cells, setCells] = useState<string[][]>([])
  const [roles, setRoles] = useState<string[]>([])
  const [dropped, setDropped] = useState<Set<number>>(new Set())

  useEffect(() => {
    if (!table) return
    // Every list here has a server-side default, which the generated schema
    // makes optional, so they are read defensively as the other panels do.
    setCells((table.rows ?? []).map((row) => row.map((cell) => cell?.text ?? '')))
    // Two columns are almost always time and drawdown, in that order, but which
    // unit is the user's to say — so the kind is offered and the unit is not
    // assumed. Anything wider starts unnamed.
    setRoles(Array.from({ length: table.columns }, () => IGNORED))
    setDropped(new Set())
  }, [table])

  const rows = table?.rows ?? []
  const named = roles.filter((role) => role !== IGNORED)
  const hasTime = named.some((role) => role.startsWith('time'))
  const hasDrawdown = named.some((role) => role.startsWith('drawdown'))
  const duplicated = new Set(named).size !== named.length
  const kept = rows.length - dropped.size

  const text = useMemo(() => {
    const used = roles
      .map((role, index) => ({ role, index }))
      .filter(({ role }) => role !== IGNORED)
    const lines = [used.map(({ role }) => role).join('\t')]
    cells.forEach((row, rowIndex) => {
      if (dropped.has(rowIndex)) return
      lines.push(used.map(({ index }) => (row[index] ?? '').trim()).join('\t'))
    })
    return lines.join('\n')
  }, [cells, roles, dropped])

  // A row with a hole in it, or with something that is not a number in it,
  // would reach the server as a blank or a NaN — neither of which is a reading.
  // Recognition does leave the odd `5)` behind, so this is not hypothetical.
  const { incomplete, unreadable } = useMemo(() => {
    const used = roles.map((role, index) => ({ role, index }))
      .filter(({ role }) => role !== IGNORED)
    let blank = 0
    let notNumbers = 0
    cells.forEach((row, rowIndex) => {
      if (dropped.has(rowIndex)) return
      const values = used.map(({ index }) => (row[index] ?? '').trim())
      if (values.some((value) => !value)) blank += 1
      else if (values.some((value) => !Number.isFinite(Number(value)))) {
        notNumbers += 1
      }
    })
    return { incomplete: blank, unreadable: notNumbers }
  }, [cells, roles, dropped])

  const ready = hasTime && hasDrawdown && !duplicated && kept > 0
    && incomplete === 0 && unreadable === 0

  function edit(rowIndex: number, columnIndex: number, value: string) {
    setCells((previous) => previous.map((row, index) =>
      index === rowIndex
        ? row.map((cell, position) => (position === columnIndex ? value : cell))
        : row))
  }

  function toggleRow(rowIndex: number) {
    setDropped((previous) => {
      const next = new Set(previous)
      if (next.has(rowIndex)) next.delete(rowIndex)
      else next.add(rowIndex)
      return next
    })
  }

  return (
    <Modal opened={opened} onClose={onClose} size="xl"
           title="Check the readings from the photograph">
      <Stack gap="sm">
        <Alert color="yellow" variant="light" icon={<IconAlertTriangle size={18} />}>
          Read every value against the sheet before using it. A misread digit
          fits a type curve as well as the right one does, and reports a
          transmissivity that is simply wrong — so nothing here becomes readings
          until you confirm it.
        </Alert>

        {table && (
          <Group gap="xs">
            <Badge variant="light">{rows.length} rows × {columns} columns</Badge>
            {table.uncertain_cells > 0 && (
              <Badge variant="light" color="yellow">
                {table.uncertain_cells} of {table.total_cells} cells uncertain
              </Badge>
            )}
            {dropped.size > 0 && (
              <Badge variant="light" color="gray">{dropped.size} rows left out</Badge>
            )}
          </Group>
        )}

        {(table?.notes?.length ?? 0) > 0 && (
          <List size="xs" c="dimmed">
            {(table?.notes ?? []).map((note) => (
              <List.Item key={note}>{note}</List.Item>
            ))}
          </List>
        )}

        <Text size="sm">
          Name each column. A photograph carries no header row, so the units
          cannot be read from it — and minutes taken for seconds would change the
          answer by a factor of sixty.
        </Text>

        <ScrollArea.Autosize mah={420}>
          <Table withTableBorder withColumnBorders fz="xs"
                 style={{ fontVariantNumeric: 'tabular-nums' }}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th w={34} />
                {Array.from({ length: columns }, (_, index) => (
                  <Table.Th key={index}>
                    <Select
                      size="xs"
                      aria-label={`Column ${index + 1}`}
                      data={COLUMN_ROLES as unknown as { value: string; label: string }[]}
                      value={roles[index] ?? IGNORED}
                      onChange={(value) => setRoles((previous) =>
                        previous.map((role, position) =>
                          position === index ? (value ?? IGNORED) : role))}
                      comboboxProps={{ withinPortal: true }}
                    />
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((row, rowIndex) => (
                <Table.Tr key={rowIndex}
                          opacity={dropped.has(rowIndex) ? 0.4 : 1}>
                  <Table.Td>
                    <Tooltip label={dropped.has(rowIndex)
                      ? 'Put this row back' : 'Leave this row out'}>
                      <Button size="compact-xs" variant="subtle" color="gray"
                              aria-label={`Row ${rowIndex + 1}`}
                              onClick={() => toggleRow(rowIndex)}>
                        <IconTrash size={14} />
                      </Button>
                    </Tooltip>
                  </Table.Td>
                  {Array.from({ length: columns }, (_, columnIndex) => {
                    const cell = row[columnIndex]
                    const doubtful = cell?.uncertain ?? false
                    return (
                      <Table.Td key={columnIndex} p={2}>
                        <Tooltip
                          disabled={!cell}
                          label={cell
                            ? `read as "${cell.text}", ${Math.round(cell.confidence)}% sure`
                            : ''}>
                          <TextInput
                            size="xs"
                            variant={doubtful ? 'filled' : 'default'}
                            error={doubtful || undefined}
                            aria-label={`Row ${rowIndex + 1} column ${columnIndex + 1}`}
                            disabled={dropped.has(rowIndex)}
                            value={cells[rowIndex]?.[columnIndex] ?? ''}
                            onChange={(event) =>
                              edit(rowIndex, columnIndex, event.currentTarget.value)}
                            styles={{ input: { fontVariantNumeric: 'tabular-nums' } }}
                          />
                        </Tooltip>
                      </Table.Td>
                    )
                  })}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea.Autosize>

        {!hasTime && <Text size="xs" c="red">Name the column holding the time.</Text>}
        {!hasDrawdown && (
          <Text size="xs" c="red">Name the column holding the drawdown.</Text>
        )}
        {duplicated && (
          <Text size="xs" c="red">
            Two columns carry the same name — one of them is something else.
          </Text>
        )}
        {incomplete > 0 && (
          <Text size="xs" c="red">
            {incomplete} row(s) have an empty cell in a named column. Fill them in
            from the sheet, or leave those rows out.
          </Text>
        )}
        {unreadable > 0 && (
          <Text size="xs" c="red">
            {unreadable} row(s) hold something that is not a number. Correct them
            from the sheet, or leave those rows out.
          </Text>
        )}

        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onClose}>Discard</Button>
          <Button disabled={!ready} loading={busy} onClick={() => onConfirm(text)}>
            Use these {kept} readings
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}
