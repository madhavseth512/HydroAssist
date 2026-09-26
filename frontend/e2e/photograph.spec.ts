/**
 * A photographed field sheet, end to end, in a real browser.
 *
 * The one import that cannot fail loudly. Recognition always produces numbers,
 * and a misread digit fits a type curve as well as the right one, so the whole
 * design rests on a claim that only a test like this can check: the photograph
 * reaches Tesseract on the server, comes back as a grid, and becomes readings
 * only after a person has named the columns and pressed the button.
 *
 * Skipped where the server has no Tesseract — it is a program, not a package.
 */
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

const FIELD_SHEET = fileURLToPath(
  new URL('../../tests/fixtures/field_data/field_sheet.png', import.meta.url))

/** Pick an option from a Mantine Select by the column it labels.
 *
 * `exact`, because Playwright matches an accessible name as a substring and
 * every cell of the grid is labelled "Row 3 column 1".
 */
async function nameColumn(page: import('@playwright/test').Page,
                          column: string, role: string) {
  await page.getByRole('textbox', { name: column, exact: true }).click()
  await page.getByRole('option', { name: role, exact: true }).click()
}

async function photograph(page: import('@playwright/test').Page) {
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Photograph of a table' }))
    .toBeEnabled()
  await page.locator('input[type="file"][accept*="image/"]')
    .setInputFiles(FIELD_SHEET)

  // A server without Tesseract answers 503, and the panel reports it rather
  // than opening a grid. That is a deployment fact, not a failure of this test.
  const grid = page.getByText('Check the readings from the photograph')
  const refused = page.getByText(/Tesseract/)
  await expect(grid.or(refused).first()).toBeVisible({ timeout: 30_000 })
  if (await refused.isVisible()) {
    test.skip(true, 'the server has no Tesseract installed')
  }
}

test('a photograph proposes a grid, and only then becomes readings', async ({ page }) => {
  await photograph(page)

  // Thirteen readings off the sheet, and the title and heading rows left out.
  await expect(page.getByText('13 rows × 2 columns')).toBeVisible()
  await expect(page.getByLabel('Row 1 column 1', { exact: true }))
    .toHaveValue('1')
  await expect(page.getByLabel('Row 13 column 2', { exact: true }))
    .toHaveValue('1.07')
  // The one cell recognition was unsure of, marked for a second look.
  await expect(page.getByText('1 of 26 cells uncertain')).toBeVisible()

  // Nothing has become readings yet.
  await expect(page.getByText(/data points ready/)).toHaveCount(0)

  await nameColumn(page, 'Column 1', 'Time (minutes)')
  await nameColumn(page, 'Column 2', 'Drawdown (metres)')
  await page.getByRole('button', { name: /Use these 13 readings/ }).click()

  await expect(page.getByText(/13 data points ready/)).toBeVisible()
  await expect(page.getByText('Check the readings from the photograph'))
    .toHaveCount(0)
})

test('a correction in the grid is the number that is analysed', async ({ page }) => {
  await photograph(page)

  // A reading the sheet shows as 0.68, mis-transcribed and put right here.
  const cell = page.getByLabel('Row 7 column 2', { exact: true })
  await cell.fill('0.95')

  await nameColumn(page, 'Column 1', 'Time (minutes)')
  await nameColumn(page, 'Column 2', 'Drawdown (metres)')
  await page.getByRole('button', { name: /Use these 13 readings/ }).click()

  await expect(page.getByText(/13 data points ready/)).toBeVisible()
  await page.getByRole('button', { name: /Preview the formatted data/ }).click()
  await expect(page.getByRole('cell', { name: '0.95', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '0.68', exact: true }))
    .toHaveCount(0)
})

test('the readings from a sheet can be fitted like any others', async ({ page }) => {
  await photograph(page)

  await nameColumn(page, 'Column 1', 'Time (minutes)')
  await nameColumn(page, 'Column 2', 'Drawdown (metres)')
  await page.getByRole('button', { name: /Use these 13 readings/ }).click()
  await expect(page.getByText(/13 data points ready/)).toBeVisible()

  const run = page.getByRole('button', { name: 'Run analysis' })
  await expect(run).toBeEnabled()
  await run.click()

  await expect(page.getByText('Results — Theis (1935)')).toBeVisible()
})
