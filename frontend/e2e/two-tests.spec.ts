/**
 * Two tests in one project, in a real browser.
 *
 * Reported from the deployed site: Test 1 and Test 2 both listed, and clicking
 * either showed no analysis at all. The readings were kept per test; the result
 * was kept once per session and cleared whenever a test was opened — deliberately,
 * so a fit was never shown against another test's data, and with the effect that
 * an afternoon's work left nothing behind.
 *
 * This drives the sequence that was reported, which unit tests cannot: two tests,
 * a fit on each, then clicking back and forth.
 */
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

function sample(name: string): string {
  return fileURLToPath(new URL(`../../tests/${name}`, import.meta.url))
}

const CSV = 'input[type="file"][accept*=".csv"]'

/** Give the active test readings and fit it with whatever is preselected. */
async function fitTheActiveTest(page: import('@playwright/test').Page) {
  await expect(page.getByRole('button', { name: /Choose file|Replace file/ }))
    .toBeEnabled()
  await page.locator(CSV).setInputFiles(sample('theis_cooper_jacob_sample.csv'))
  await expect(page.getByText(/50 data points ready/)).toBeVisible()

  const run = page.getByRole('button', { name: 'Run analysis' })
  await expect(run).toBeEnabled()
  await run.click()
  await expect(page.getByText(/^Results —/)).toBeVisible()
}

test('each test keeps its own analysis', async ({ page }) => {
  await page.goto('/')
  await fitTheActiveTest(page)
  await expect(page.getByText('Results — Theis (1935)')).toBeVisible()

  // A second test, fitted with a different method.
  await page.getByRole('button', { name: 'Add test' }).click();
  await expect(page.getByText('Test 2')).toBeVisible()
  await fitTheActiveTest(page)
  await page.getByRole('button', { name: /Cooper-Jacob \(1946\)/ }).first().click()
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByText('Results — Cooper-Jacob (1946)')).toBeVisible()

  // Back to the first: its own fit, not the second's, and not an empty panel.
  await page.getByText('Test 1').click()
  await expect(page.getByText('Results — Theis (1935)')).toBeVisible()
  await expect(page.getByText(/50 data points ready/)).toBeVisible()

  // And forward again.
  await page.getByText('Test 2').click()
  await expect(page.getByText('Results — Cooper-Jacob (1946)')).toBeVisible()
})

test('the run log follows the test', async ({ page }) => {
  await page.goto('/')
  await fitTheActiveTest(page)

  // Two methods on the same test: both listed, most recent last.
  await page.getByRole('button', { name: /Cooper-Jacob \(1946\)/ }).first().click()
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByText('Results — Cooper-Jacob (1946)')).toBeVisible()

  await expect(page.getByText('This session: 2 fits')).toBeVisible()

  // A new test has none of them.
  await page.getByRole('button', { name: 'Add test' }).click()
  await expect(page.getByText(/This session: \d fits?/)).toHaveCount(0)
})
