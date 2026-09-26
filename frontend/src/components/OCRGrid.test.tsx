/**
 * The grid that stands between a photograph and a fit.
 *
 * These tests are about refusals rather than rendering. OCR output is the one
 * input that is confidently wrong, so what matters is that the component will
 * not hand anything on until the user has named the columns and filled the
 * holes — and that what it does hand on is exactly what is on screen.
 */
import { MantineProvider } from '@mantine/core'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { OCRTable } from '../api/client'
import { OCRGrid } from './OCRGrid'

function cell(text: string, confidence = 96) {
  return { text, value: Number(text), confidence, uncertain: confidence < 80 }
}

const TABLE: OCRTable = {
  columns: 2,
  rows: [
    [cell('1'), cell('0.45')],
    [cell('2'), cell('0.61', 42)],
    [cell('5'), cell('0.83')],
  ],
  notes: ['3 rows and 2 columns of numbers were read from the image'],
  total_cells: 6,
  uncertain_cells: 1,
  mostly_uncertain: false,
}

function show(table: OCRTable = TABLE, onConfirm = vi.fn()) {
  render(
    <MantineProvider>
      <OCRGrid table={table} opened onClose={vi.fn()} onConfirm={onConfirm} />
    </MantineProvider>,
  )
  return onConfirm
}

/** Name the two columns, which the component requires before anything else. */
async function nameColumns(user: ReturnType<typeof userEvent.setup>,
                           time = 'Time (minutes)', drawdown = 'Drawdown (metres)') {
  // By role, not by label: Mantine's Select carries its aria-label on both the
  // visible input and the hidden one it submits with.
  await user.click(screen.getByRole('textbox', { name: 'Column 1' }))
  await user.click(await screen.findByRole('option', { name: time }))
  await user.click(screen.getByRole('textbox', { name: 'Column 2' }))
  await user.click(await screen.findByRole('option', { name: drawdown }))
}

describe('OCRGrid', () => {
  it('shows every cell as the characters that were read', () => {
    show()

    expect(screen.getByLabelText('Row 1 column 1')).toHaveValue('1')
    expect(screen.getByLabelText('Row 2 column 2')).toHaveValue('0.61')
  })

  it('warns that a misread digit fits as well as the right one', () => {
    show()

    expect(screen.getByText(/misread digit/i)).toBeInTheDocument()
  })

  it('marks the cells Tesseract was unsure of', () => {
    show()

    // Mantine flags an errored input with aria-invalid, which is what a screen
    // reader and a sighted user both get from the yellow cell.
    expect(screen.getByLabelText('Row 2 column 2'))
      .toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByLabelText('Row 1 column 2'))
      .not.toHaveAttribute('aria-invalid', 'true')
  })

  it('says how much of the reading is uncertain', () => {
    show()

    expect(screen.getByText('1 of 6 cells uncertain')).toBeInTheDocument()
  })

  it('will not hand on readings until the columns are named', () => {
    show()

    expect(screen.getByRole('button', { name: /Use these/ })).toBeDisabled()
    expect(screen.getByText(/Name the column holding the time/)).toBeInTheDocument()
  })

  it('hands on tab-delimited text headed by the names chosen', async () => {
    const user = userEvent.setup()
    const onConfirm = show()

    await nameColumns(user)
    await user.click(screen.getByRole('button', { name: /Use these 3 readings/ }))

    expect(onConfirm).toHaveBeenCalledWith(
      'time_min\tdrawdown_m\n1\t0.45\n2\t0.61\n5\t0.83')
  })

  it('hands on the corrected value, not the one that was read', async () => {
    const user = userEvent.setup()
    const onConfirm = show()

    await nameColumns(user)
    const doubtful = screen.getByLabelText('Row 2 column 2')
    await user.clear(doubtful)
    await user.type(doubtful, '0.67')
    await user.click(screen.getByRole('button', { name: /Use these/ }))

    expect(onConfirm).toHaveBeenCalledWith(expect.stringContaining('2\t0.67'))
  })

  it('leaves a dropped row out of what it hands on', async () => {
    const user = userEvent.setup()
    const onConfirm = show()

    await nameColumns(user)
    await user.click(screen.getByRole('button', { name: 'Row 2' }))
    await user.click(screen.getByRole('button', { name: /Use these 2 readings/ }))

    expect(onConfirm).toHaveBeenCalledWith('time_min\tdrawdown_m\n1\t0.45\n5\t0.83')
  })

  it('refuses two columns with the same name', async () => {
    const user = userEvent.setup()
    show()

    await nameColumns(user, 'Time (minutes)', 'Time (minutes)')

    expect(screen.getByText(/same name/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Use these/ })).toBeDisabled()
  })

  it('refuses a row with a hole in a named column', async () => {
    const user = userEvent.setup()
    show()

    await nameColumns(user)
    await user.clear(screen.getByLabelText('Row 3 column 1'))

    expect(screen.getByText(/empty cell in a named column/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Use these/ })).toBeDisabled()
  })

  it('refuses a cell that is not a number', async () => {
    // Not hypothetical: recognition reads a printed "5" as "5)" often enough
    // that the server tidies those and flags them. A leftover would otherwise
    // reach the calculator as a NaN.
    const user = userEvent.setup()
    show()

    await nameColumns(user)
    const cell = screen.getByLabelText('Row 1 column 2')
    await user.clear(cell)
    await user.type(cell, '0.4o')

    expect(screen.getByText(/not a number/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Use these/ })).toBeDisabled()
  })

  it('accepts that row once it is filled in from the sheet', async () => {
    const user = userEvent.setup()
    const onConfirm = show()

    await nameColumns(user)
    const empty = screen.getByLabelText('Row 3 column 1')
    await user.clear(empty)
    await user.type(empty, '4')
    await user.click(screen.getByRole('button', { name: /Use these/ }))

    expect(onConfirm).toHaveBeenCalledWith(expect.stringContaining('4\t0.83'))
  })

  it('lets a column be left out of the readings entirely', async () => {
    const user = userEvent.setup()
    const table: OCRTable = {
      ...TABLE,
      columns: 3,
      rows: TABLE.rows!.map((row, index) => [...row, cell(`14.${index + 1}`)]),
    }
    const onConfirm = show(table)

    // Column 3 is the water temperature, which is not a reading.
    await nameColumns(user)
    await user.click(screen.getByRole('button', { name: /Use these/ }))

    expect(onConfirm).toHaveBeenCalledWith(
      'time_min\tdrawdown_m\n1\t0.45\n2\t0.61\n5\t0.83')
  })

  it('reports what the image gave back', () => {
    show()

    expect(screen.getByText('3 rows × 2 columns')).toBeInTheDocument()
    expect(screen.getByText(/3 rows and 2 columns of numbers/)).toBeInTheDocument()
  })

  it('renders every cell of the grid, editable', () => {
    show()
    const grid = screen.getByRole('table')

    // Three rows of two, each its own field — and a control per row to drop it.
    expect(within(grid).getAllByLabelText(/^Row \d column \d$/)).toHaveLength(6)
    expect(within(grid).getAllByRole('button', { name: /^Row \d$/ })).toHaveLength(3)
  })
})
