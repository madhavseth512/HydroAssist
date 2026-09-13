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

function open() {
  return renderWithProviders(<ChatDrawer opened onClose={() => {}} />)
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
      { url: '/api/chat/messages', method: 'POST', body: CONVERSATION },
      { url: '/api/chat', body: state() },
    ])

    open()
    fireEvent.change(await screen.findByLabelText('Message'),
                     { target: { value: 'When should I use Cooper-Jacob?' } })
    fireEvent.click(screen.getByRole('button', { name: /Send/ }))

    expect(await screen.findByText('When u < 0.02 at late time.')).toBeInTheDocument()
    expect(bodyOf(stub, '/api/chat/messages').message)
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
