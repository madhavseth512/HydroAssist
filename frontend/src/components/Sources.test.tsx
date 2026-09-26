/**
 * What an answer read, shown so a reader can go and check it.
 *
 * The three faults reported from the deployed site, as three groups of tests: a
 * citation that named a folder, a passage clipped with no way to see the rest,
 * and one list shared by every answer in the conversation.
 */
import { MantineProvider } from '@mantine/core'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { RetrievedDoc } from '../api/client'
import { Sources } from './Sources'

function doc(overrides: Partial<RetrievedDoc> = {}): RetrievedDoc {
  return {
    document: 'Walton (1962), Illinois State Water Survey Bulletin 49',
    page: 32,
    url: '',
    method: 'hantush_jacob_1955',
    aquifer: 'leaky',
    score: 0.78,
    preview: 'The leaky artesian formula may be used where…',
    text: 'The leaky artesian formula may be used where the aquifer is overlain '
      + 'by a confining bed through which leakage occurs.',
    ...overrides,
  }
}

function show(sources: RetrievedDoc[]) {
  render(
    <MantineProvider>
      <Sources sources={sources} id="1" />
    </MantineProvider>,
  )
}

/** The list is collapsed until asked for, so the transcript stays readable. */
async function open(sources: RetrievedDoc[]) {
  const user = userEvent.setup()
  show(sources)
  await user.click(screen.getByRole('button', { name: /What this answer read/ }))
  return user
}

describe('the sources under an answer', () => {
  it('names the document and the page', async () => {
    await open([doc()])

    expect(screen.getByText(/Illinois State Water Survey Bulletin 49/))
      .toBeInTheDocument()
    expect(screen.getByText('page 32')).toBeInTheDocument()
  })

  it('says how relevant the passage was', async () => {
    await open([doc()])

    expect(screen.getByText('relevance 78%')).toBeInTheDocument()
  })

  it('offers a link when there is one', async () => {
    await open([doc({ url: 'https://doi.org/10.1016/0022-1694(88)90025-X' })])

    expect(screen.getByRole('link'))
      .toHaveAttribute('href', 'https://doi.org/10.1016/0022-1694(88)90025-X')
  })

  it('says plainly when there is no link, rather than implying one', async () => {
    // Most of this corpus is paywalled. A link that 404s costs the reader their
    // trust in every other citation.
    await open([doc({ url: '' })])

    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText('no public link')).toBeInTheDocument()
  })

  it('opens a cited link away from the analysis in progress', async () => {
    await open([doc({ url: 'https://doi.org/10.1016/x' })])

    expect(screen.getByRole('link')).toHaveAttribute('target', '_blank')
    expect(screen.getByRole('link')).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('shows the opening of the passage, and the whole of it on request', async () => {
    const user = await open([doc()])

    expect(screen.getByText(/The leaky artesian formula may be used where…/))
      .toBeInTheDocument()

    await user.click(screen.getByText(/may be used where…/))

    expect(screen.getByText(/through which leakage occurs/)).toBeInTheDocument()
  })

  it('counts the passages, in the plural only when there are several', () => {
    show([doc(), doc({ document: 'Theis (1935)' })])

    expect(screen.getByText('What this answer read (2 passages)'))
      .toBeInTheDocument()
  })

  it('says passage, singular, for one', () => {
    show([doc()])

    expect(screen.getByText('What this answer read (1 passage)'))
      .toBeInTheDocument()
  })

  it('shows nothing at all for an answer that read nothing', () => {
    // An answer that failed, or one taken from the workspace's own result, has
    // no sources — and citing papers under it would be a small dishonesty.
    show([])

    expect(screen.queryByText(/What this answer read/)).toBeNull()
  })

  it('never shows the folder a paper was filed in', async () => {
    // What was actually on screen before: "Custom_Leaky Confined Aquifer".
    await open([doc()])

    expect(screen.queryByText(/Custom_/)).toBeNull()
  })
})
