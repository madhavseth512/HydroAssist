/**
 * Tracer tests, end to end: a concentration file switches the workspace over by
 * itself, and nothing downstream talks about T, S or drawdown.
 */
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

function sample(name: string): string {
  return fileURLToPath(new URL(`../../tests/${name}`, import.meta.url))
}

async function csvInput(page: import('@playwright/test').Page) {
  await expect(page.getByRole('button', { name: 'Choose file' })).toBeEnabled()
  return page.locator('input[type="file"][accept*=".csv"]')
}

test('a forced-gradient test: upload, fit, match, report, save', async ({ page }) => {
  await page.goto('/')
  await (await csvInput(page)).setInputFiles(sample('tracer_convergent_sample.csv'))

  await expect(page.getByText(/40 data points ready/)).toBeVisible()
  await expect(page.getByText(/concentration: concentration_mg_L/)).toBeVisible()
  // A flow regime means nothing on a breakthrough curve.
  await expect(page.getByText(/Consistent with/)).toHaveCount(0)

  // The form has followed the readings to the tracer methods.
  await page.getByRole('button', { name: /^Sauty \(1980\)/ }).click()
  await page.getByLabel('Distance from injection to sampling well (m)').fill('15')
  await page.getByLabel('Aquifer thickness b (m), for porosity').fill('10')
  await page.getByLabel('Injected mass (kg), for recovery').fill('5')
  await page.screenshot({ path: 'test-results/tracer-setup.png', fullPage: true })

  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByText('Results — Sauty (1980)')).toBeVisible()
  await expect(page.getByText('n_e', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('cell', { name: 'Mass recovery' })).toBeVisible()
  await expect(page.getByText(/^T \(/)).toHaveCount(0)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: 'test-results/tracer-results.png', fullPage: true })

  await page.getByRole('tab', { name: 'Manual matching' }).click()
  await expect(page.getByText(/Manual RMSE .* mg\/L/)).toBeVisible()

  const report = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download report' }).click()
  expect((await report).suggestedFilename()).toBe('sauty_report.pdf')

  await page.getByRole('button', { name: 'Save to project' }).click()
  await expect(page.getByText(/Compare saved analyses/)).toBeVisible()
  await expect(page.getByRole('columnheader', { name: 'Estimates' })).toBeVisible()
})

test('a natural-gradient test asks for no pumping rate', async ({ page }) => {
  await page.goto('/')
  await (await csvInput(page)).setInputFiles(sample('tracer_natural_sample.csv'))
  await expect(page.getByText(/40 data points ready/)).toBeVisible()

  await page.getByRole('button', { name: /^Ogata-Banks \(1961\)/ }).click()
  await expect(page.getByLabel('Pumping rate Q')).toHaveCount(0)
  await page.getByLabel('Distance from injection to sampling well (m)').fill('25')

  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByText('Results — Ogata-Banks (1961)')).toBeVisible()
  await expect(page.getByText(/m\/day/).first()).toBeVisible()
})

test('choosing Tracer first reads an unnamed column as a concentration', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Step 2 — Test setup')).toBeVisible()
  await page.getByText('Tracer test', { exact: true }).click()
  await expect(page.getByText('Injection', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Paste data' }).click()
  const rows = [0, 0, 1, 40, 300, 700, 500, 200, 80, 20, 5, 1]
    .map((value, i) => `${i + 1}\t${value}`).join('\n')
  await page.getByLabel('Pasted readings').fill(`time_hours\tvalue\n${rows}`)
  await page.getByRole('button', { name: 'Use these readings' }).click()

  await expect(page.getByText(/12 data points ready/)).toBeVisible()
  // 700 would have been converted from feet if it had been read as drawdown.
  await expect(page.getByText(/C: 0 – 700/)).toBeVisible()
  await expect(page.getByRole('button', { name: /^Sauty \(1980\)/ })).toBeVisible()
})
