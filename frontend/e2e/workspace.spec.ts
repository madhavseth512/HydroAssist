/**
 * The workspace, end to end, in a real browser.
 *
 * These drive what a hydrogeologist actually does — upload a file, fit it,
 * download the report, keep the project — against the real API and the built
 * SPA. Each test gets a fresh browser context, and so a fresh server session.
 */
import { fileURLToPath } from 'node:url'

import { expect, test } from '@playwright/test'

function sample(name: string): string {
  return fileURLToPath(new URL(`../../tests/${name}`, import.meta.url))
}

/** The data step's file input, once the project it uploads into has loaded. */
async function csvInput(page: import('@playwright/test').Page) {
  await expect(page.getByRole('button', { name: 'Choose CSV' })).toBeEnabled()
  // The project panel has a file input too; this is the data step's.
  return page.locator('input[type="file"][accept*="csv"]')
}

test('a constant-rate test: upload, fit, report, save', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Step 1 — Field data')).toBeVisible()

  await (await csvInput(page)).setInputFiles(sample('theis_cooper_jacob_sample.csv'))
  await expect(page.getByText(/50 data points ready/)).toBeVisible()

  const run = page.getByRole('button', { name: 'Run analysis' })
  await expect(run).toBeEnabled()
  await run.click()

  await expect(page.getByText('Results — Theis (1935)')).toBeVisible()
  // The golden value: the same number the calculator's regression test pins.
  await expect(page.getByText('215.29', { exact: true })).toBeVisible()

  const report = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download report' }).click()
  expect((await report).suggestedFilename()).toBe('theis_report.pdf')

  await page.getByRole('button', { name: 'Save to project' }).click()
  await expect(page.getByRole('button', { name: 'Saved to project' })).toBeDisabled()
  await expect(page.getByText(/Compare saved analyses/)).toBeVisible()

  const projectFile = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  expect((await projectFile).suggestedFilename()).toMatch(/\.hydroassist\.json$/)
})

test('a recovery test reports no storativity', async ({ page }) => {
  await page.goto('/')

  await (await csvInput(page)).setInputFiles(sample('theis_recovery_sample.csv'))
  await expect(page.getByText(/data points ready/)).toBeVisible()

  await page.getByText('Recovery (after the pump stops)').click()
  await expect(page.getByLabel('Pumping duration before shut-down t_p (hours)'))
    .toBeVisible()

  await page.getByRole('button', { name: 'Run analysis' }).click()

  await expect(page.getByText('Results — Theis Recovery (1935)')).toBeVisible()
  await expect(page.getByText('not determined')).toBeVisible()
})

test('the workspace carries on while the consultant is unavailable', async ({ page }) => {
  await page.goto('/')

  await page.getByRole('button', { name: 'Consultant' }).click()
  await expect(page.getByText(/Google may use what you send/)).toBeVisible()

  await page.getByLabel('Message').fill('hello')
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByText(/not available on this server/)).toBeVisible()
})
