/** The consultation drawer: asking, the suggestion card, and honesty about the model. */
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChatDrawer } from './ChatDrawer'
import { renderWithProviders } from '../test/render'
import { chatState, project, stubFetch } from '../test/stubFetch'

afterEach(() => vi.unstubAllGlobals())

const METHODS = {
  test_type: 'constant_rate', gated_out: [], help: '', default_index: 0,
  offered: [{ id: 'theis_1935', label: 'Theis (1935)' },
            { id: 'cooper_jacob_1946', label: 'Cooper-Jacob (1946)' }],
}

function state(overrides: Record<string, unknown> = {}) {
  return { ...chatState, ...overrides }
}

const CONVERSATION = state({
  messages: [{ role: 'user', content: 'When should I use Cooper-Jacob?' },
             { role: 'assistant', content: 'When u < 0.02 at late time.' }],
  context: { ...chatState.context, message_count: 2 },
})

function bodyOf(stub: ReturnType<typeof stubFetch>, fragment: string) {
  const [, init] = stub.mock.mock.calls.find(([url]) => String(url).includes(fragment))!
  return JSON.parse((init as RequestInit).body as string)
}

// Docked rather than overlaid, so there is nothing to open: it is simply
// rendered where the workspace puts it.
function open(onGoTo?: (panel: string) => void) {
  return renderWithProviders(<ChatDrawer onGoTo={onGoTo} />)
}

describe('the consultation', () => {
  it('says before anyone types that the free tier may use what is sent', async () => {
    stubFetch([{ url: '/api/chat', body: state() }])

    open()

    expect(await screen.findByText(/Google may use what you send/)).toBeInTheDocument()
  })

  it('shows the conversation so far', async () => {
    stubFetch([{ url: '/api/chat', body: CONVERSATION }])

    open()

    expect(await screen.findByText('When u < 0.02 at late time.')).toBeInTheDocument()
    expect(screen.getByText('When should I use Cooper-Jacob?')).toBeInTheDocument()
  })

  it('sends a message and shows the reply', async () => {
    const stub = stubFetch([
      { url: '/api/chat/messages/stream', method: 'POST',
        tokens: ['When u < ', '0.02 at ', 'late time.'], body: CONVERSATION },
      { url: '/api/chat', body: state() },
    ])

    open()
    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'When should I use Cooper-Jacob?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText('When u < 0.02 at late time.')).toBeInTheDocument()
    expect(bodyOf(stub, '/api/chat/messages/stream').message)
      .toBe('When should I use Cooper-Jacob?')
  })

  it('offers the suggested method with its reason', async () => {
    const stub = stubFetch([
      { url: '/api/chat/confirm', method: 'POST', body: state() },
      { url: '/api/chat', body: state({ suggestion: {
        method: 'Papadopulos-Cooper (1967)', confidence: 0.85, supported: true,
        reason: 'A well radius of 0.30 m makes wellbore storage significant.' } }) },
      { url: '/api/methods', body: METHODS },
      { url: '/api/project', body: project() },
    ])

    open()

    expect(await screen.findByText('Suggested method: Papadopulos-Cooper (1967)'))
      .toBeInTheDocument()
    expect(screen.getByText(/wellbore storage significant/)).toBeInTheDocument()
    expect(screen.getByText('Confidence 85%')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Use this method' }))

    await waitFor(() =>
      expect(bodyOf(stub, '/api/chat/confirm').method).toBe('Papadopulos-Cooper (1967)'))
  })

  it('says plainly when there is no calculator, with nothing to accept', async () => {
    stubFetch([
      { url: '/api/chat', body: state({ suggestion: {
        method: '', confidence: 0.9, supported: false,
        reason: 'No calculator is implemented for this aquifer type yet.' } }) },
      { url: '/api/methods', body: METHODS },
      { url: '/api/project', body: project() },
    ])

    open()

    expect(await screen.findByText(/No calculator is implemented/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Use this method' })).not.toBeInTheDocument()
  })

  it('explains a consultant that is unavailable', async () => {
    stubFetch([
      { url: '/api/chat/messages', method: 'POST', status: 502,
        body: { detail: 'The consultant is unavailable: quota exhausted' } },
      { url: '/api/chat', body: state() },
    ])

    open()
    fireEvent.change(await screen.findByLabelText('Message'), { target: { value: 'hello' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText(/quota exhausted/)).toBeInTheDocument()
  })

  it('resets the conversation', async () => {
    const stub = stubFetch([
      { url: '/api/chat/reset', method: 'POST', body: state() },
      { url: '/api/chat', body: CONVERSATION },
    ])

    open()
    await screen.findByText('When u < 0.02 at late time.')
    fireEvent.click(screen.getByRole('button', { name: 'Reset conversation' }))

    await waitFor(() =>
      expect(screen.queryByText('When u < 0.02 at late time.')).not.toBeInTheDocument())
    expect(stub.calls.some((c) => c.url.includes('/api/chat/reset'))).toBe(true)
  })
})

describe('how an answer is rendered', () => {
  const MARKDOWN = [
    '### Assumptions of the Theis method',
    '',
    '*   **Confined aquifer** of infinite extent',
    '*   **Constant** pumping rate',
    '',
    'See [Custom_General, p.7] for the derivation.',
  ].join('\n')

  it('renders the model markdown as elements, not as characters', async () => {
    // The consultant writes markdown. Printing it verbatim — ### and ** and all
    // — was most of why the answers looked unprofessional.
    stubFetch([{ url: '/api/chat', body: state({
      messages: [{ role: 'assistant', content: MARKDOWN }],
    }) }])

    open()

    const heading = await screen.findByRole('heading', {
      name: 'Assumptions of the Theis method',
    })
    expect(heading).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    expect(screen.getByText('Confined aquifer').tagName).toBe('STRONG')
    expect(document.body.textContent).not.toContain('###')
    expect(document.body.textContent).not.toContain('**Confined')
  })

  it('leaves what the user typed exactly as typed', async () => {
    // Rendering their asterisks as bold would be rewriting their words.
    stubFetch([{ url: '/api/chat', body: state({
      messages: [{ role: 'user', content: 'is 2*3*4 the rate **or** not?' }],
    }) }])

    open()

    expect(await screen.findByText('is 2*3*4 the rate **or** not?')).toBeInTheDocument()
  })

  it('renders a markdown table, which is how comparisons arrive', async () => {
    stubFetch([{ url: '/api/chat', body: state({
      messages: [{ role: 'assistant', content:
        '| Method | T |\n|---|---|\n| Theis | 215 |\n| Cooper-Jacob | 198 |' }],
    }) }])

    open()

    expect(await screen.findByRole('table')).toBeInTheDocument()
    expect(screen.getAllByRole('row')).toHaveLength(3)
  })

  it('opens a cited link away from the analysis in progress', async () => {
    stubFetch([{ url: '/api/chat', body: state({
      messages: [{ role: 'assistant', content: 'See [USGS](https://usgs.gov).' }],
    }) }])

    open()

    const link = await screen.findByRole('link', { name: 'USGS' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'))
  })
})

describe('what the guide offers', () => {
  const WITH_OFFER = state({
    messages: [{ role: 'user', content: 'Where do I enter the pumping rate?' },
               { role: 'assistant', content: 'The pumping rate Q goes in Step 2.' }],
    actions: [{ kind: 'open_panel', target: 'setup', label: 'Go to Step 2 — Test setup' }],
  })

  it('shows the offer as a button', async () => {
    stubFetch([{ url: '/api/chat', body: WITH_OFFER }])

    open()

    expect(await screen.findByRole('button', { name: /Go to Step 2/ }))
      .toBeInTheDocument()
  })

  it('goes nowhere until the button is pressed', async () => {
    // An assistant that moves the workspace under someone mid-analysis is a
    // worse failure than one that answers badly.
    const goTo = vi.fn()
    stubFetch([{ url: '/api/chat', body: WITH_OFFER }])

    open(goTo)

    await screen.findByRole('button', { name: /Go to Step 2/ })
    expect(goTo).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /Go to Step 2/ }))
    expect(goTo).toHaveBeenCalledWith('setup')
  })

  it('offers nothing when the answer offered nothing', async () => {
    stubFetch([{ url: '/api/chat', body: state({
      messages: [{ role: 'assistant', content: 'The Theis method assumes…' }],
      actions: [],
    }) }])

    open()

    await screen.findByText(/The Theis method assumes/)
    expect(screen.queryByRole('button', { name: /Go to/ })).toBeNull()
  })
})

describe('sending a question', () => {
  /** A send that never answers, so what is on screen meanwhile is the point.
   * The specific route goes first: routes match by substring, and '/api/chat'
   * would otherwise answer the POST to '/api/chat/messages' as well. */
  function hangingSend() {
    return stubFetch([
      { url: '/api/chat/messages', method: 'POST', pending: true },
      { url: '/api/chat', body: state() },
    ])
  }

  it('shows the question at once, without waiting for the answer', async () => {
    hangingSend()
    open()

    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'Why was u filtered?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText('Why was u filtered?')).toBeInTheDocument()
  })

  it('empties the box at once, so nobody asks twice', async () => {
    hangingSend()
    open()

    const box = await screen.findByLabelText('Message')
    fireEvent.change(box, { target: { value: 'Why was u filtered?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(box).toHaveValue('')
  })

  it('leaves the previous answer’s sources with the previous answer', async () => {
    // They used to be one list under the whole conversation, so asking a new
    // question rewrote the citations of the old answer. Each answer owns its own
    // now, and a question in flight has none because it has read nothing.
    stubFetch([
      { url: '/api/chat/messages', method: 'POST', pending: true },
      { url: '/api/chat', body: state({
        messages: [
          { role: 'user', content: 'does Theis assume a confined aquifer?' },
          { role: 'assistant', content: 'Yes — it is one of its assumptions.',
            sources: [{ document: 'Theis (1935)', page: 519, url: '',
                        method: 'theis_1935', aquifer: 'confined', score: 0.9,
                        preview: 'the aquifer is confined…', text: 'full passage' }] },
        ],
      }) },
    ])
    open()

    expect(await screen.findByText(/What this answer read/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Message'),
                     { target: { value: 'and an unconfined one?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    // Still exactly one block of sources: the earlier answer's.
    await waitFor(() => expect(
      screen.getByText('and an unconfined one?')).toBeInTheDocument())
    expect(screen.getAllByText(/What this answer read/)).toHaveLength(1)
  })

  it('gives the question back if it could not be sent', async () => {
    stubFetch([
      { url: '/api/chat/messages', method: 'POST', status: 503,
        body: { detail: 'The consultant is not available on this server.' } },
      { url: '/api/chat', body: state() },
    ])
    open()

    const box = await screen.findByLabelText('Message')
    fireEvent.change(box, { target: { value: 'Why was u filtered?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    // Restored in the box, and taken back out of the transcript: the only thing
    // on screen holding those words is the box itself.
    await waitFor(() => expect(box).toHaveValue('Why was u filtered?'))
    expect(screen.queryAllByText('Why was u filtered?')
      .every((node) => node.tagName === 'TEXTAREA')).toBe(true)
  })
})

describe('an answer as it is written', () => {
  /** Tokens, then the conversation, as the streaming route sends them. */
  function streaming(tokens: string[], final = CONVERSATION) {
    return stubFetch([
      { url: '/api/chat/messages/stream', method: 'POST', tokens, body: final },
      { url: '/api/chat', body: state() },
    ])
  }

  /** Tokens sent, then the stream held open: what is on screen is the
   * half-written answer, which only streaming can produce. */
  function halfWritten(tokens: string[]) {
    return stubFetch([
      { url: '/api/chat/messages/stream', method: 'POST', tokens,
        holdOpen: true },
      { url: '/api/chat', body: state() },
    ])
  }

  it('shows the answer before it is finished', async () => {
    // The point of the feature: the first sentence of an interpretation usually
    // says whether the fit can be trusted, and there is no reason to withhold it
    // until the last sentence exists.
    halfWritten(['The fit is ', 'trustworthy: ', 'R² = 0.96.'])
    open()

    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'is the fit any good?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText(/The fit is trustworthy/)).toBeInTheDocument()
  })

  it('leaves one copy of the answer, not two', async () => {
    // The partial text is replaced by the stored conversation, not added to it.
    streaming(['When u < ', '0.02 at late time.'])
    open()

    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'when?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    await waitFor(() => expect(
      screen.getAllByText('When u < 0.02 at late time.')).toHaveLength(1))
  })

  it('reassembles an event that arrives in pieces', async () => {
    // A chunk can split an event anywhere; the client keeps the tail until its
    // terminator arrives. The stub frames one event per chunk, and the tokens
    // here are deliberately fragments of words.
    halfWritten(['Papado', 'pulos-', 'Cooper ', 'assumes ', 'a large well.'])
    open()

    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'what does it assume?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText(/Papadopulos-Cooper assumes a large well/))
      .toBeInTheDocument()
  })

  it('falls back to the plain route when nothing can be streamed', async () => {
    // A proxy that buffers, or a browser without readable streams: the answer
    // still arrives, it simply arrives all at once.
    const stub = stubFetch([
      { url: '/api/chat/messages/stream', method: 'POST', body: {} },
      { url: '/api/chat/messages', method: 'POST', body: CONVERSATION },
      { url: '/api/chat', body: state() },
    ])
    open()

    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'when?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    // The stream answered with no events at all, so the conversation is read back
    // from the server rather than left half-written.
    await waitFor(() => expect(stub.calls.some(
      (call) => call.url.includes('/api/chat'))).toBe(true))
    expect(screen.queryByText(/undefined/)).toBeNull()
  })
})
