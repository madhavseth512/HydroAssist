/**
 * A fetch stub that answers by URL.
 *
 * Panel tests drive the real client — credentials, JSON parsing, error
 * handling — and only the network is replaced. Routes are matched by substring
 * in insertion order, so a test names just the endpoints it cares about.
 */
import { vi } from 'vitest'

export interface StubRoute {
  /** Matched against the request URL by substring. */
  url: string
  body?: unknown
  status?: number
  /** Only answer this route for a matching method (GET, POST, …). */
  method?: string
  /** Never answer. For what the UI shows while a request is still in flight —
   * an optimistic message, a spinner, a disabled button. */
  pending?: boolean
  /** Answer as server-sent events: these tokens, then `body` as the final
   * state. For the streamed consultation. */
  tokens?: string[]
  /** Send the tokens and leave the stream open, so what is on screen is the
   * half-written answer and nothing else could have put it there. */
  holdOpen?: boolean
}

export interface FetchStub {
  mock: ReturnType<typeof vi.fn>
  /** Every request made, in order. */
  calls: { url: string; method: string }[]
}

export function stubFetch(routes: StubRoute[]): FetchStub {
  const calls: { url: string; method: string }[] = []

  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : String(input)
    const method = (init?.method ?? 'GET').toUpperCase()
    calls.push({ url, method })

    const route = routes.find(
      (candidate) =>
        url.includes(candidate.url) &&
        (!candidate.method || candidate.method.toUpperCase() === method),
    )
    if (!route) {
      return new Response(JSON.stringify({ detail: `No stub for ${method} ${url}` }),
                          { status: 404,
                            headers: { 'content-type': 'application/json' } })
    }
    if (route.pending) {
      // Resolves never. The test asserts on what is on screen meanwhile.
      return new Promise<Response>(() => {})
    }
    if (route.tokens) {
      return sseResponse(route.tokens, route.holdOpen ? undefined : route.body,
                         route.holdOpen ?? false)
    }
    return new Response(
      route.body === undefined ? null : JSON.stringify(route.body),
      {
        status: route.status ?? 200,
        headers: { 'content-type': 'application/json' },
      },
    )
  })

  vi.stubGlobal('fetch', mock)
  return { mock, calls }
}

// ── Fixtures, shaped like the API's responses ────────────────────────────────

export const projectInfo = {
  title: 'Untitled project', client: '', project_number: '', location: '',
  test_date: '', analyst: '', notes: '',
}

export const setup = {
  q_value: 500, q_unit: 'm³/day', r_m: 50, r_w_m: 0.15, r_c_m: 0.1,
  test_type: 'constant_rate', t_p_hours: null, recovery_clock: 'since_stop',
  tracer_injection: 'pulse', tracer_conc_unit: 'mg/L', tracer_background: 0,
}

export function test_(overrides: Record<string, unknown> = {}) {
  return {
    id: 'test-1', name: 'Test 1', pumping_well: '', observation_well: '',
    setup, n_readings: 0, readings_source: '', n_analyses: 0,
    n_stale_analyses: 0, ...overrides,
  }
}

export function project(overrides: Record<string, unknown> = {}) {
  return {
    info: projectInfo,
    tests: [test_()],
    active_test_id: 'test-1',
    filename: 'project.hydroassist.json',
    summary: '1 test · 0 saved analyses',
    analysis_count: 0,
    modified_at: '2026-09-12T00:00:00+00:00',
    ...overrides,
  }
}

export const chatState = {
  messages: [],
  context: { intent: 'unknown', aquifer: 'unknown', method: '',
             suggestion_confirmed: false, has_result: false, message_count: 0 },
  suggestion: null,
  retrieved: [],
}

export function session(overrides: Record<string, unknown> = {}) {
  return { session_id: 'session-1', project: project(), chat: chatState,
           ...overrides }
}

export function prepared(overrides: Record<string, unknown> = {}) {
  return {
    n_points: 50,
    summary: '50 data points ready | t: 10s – 100000s | s: 0.120m – 1.480m',
    preview: [{ time_s: 10, drawdown_m: 0.12 }, { time_s: 20, drawdown_m: 0.21 }],
    conversions: ['time: minutes → seconds'],
    rows_removed: [],
    inspection: {
      time_col: 'time_s', time_unit: 'seconds', drawdown_cols: ['drawdown_m'],
      drawdown_unit: 'meters', confidence: 0.95, issues: [],
    },
    validation: { is_usable: true, errors: [], warnings: [], info: [],
                  suitable_methods: ['Theis (1935)'] },
    diagnosis: {
      regime: 'radial_flow', evidence: 'The derivative is flat at late time.',
      confidence: 0.9, notes: ['Implied T = 215 m²/day'],
      consistent_methods: ['Theis (1935)'],
    },
    diagnosis_error: '',
    derivative_figure: JSON.stringify({ data: [], layout: {} }),
    ...overrides,
  }
}


/** A server-sent-event response: each token as its own frame, then the state.
 *
 * Framed one event per chunk deliberately — the client has to reassemble events
 * that arrive split, and a stub that hands it whole events would never exercise
 * that. */
function sseResponse(tokens: string[], state: unknown,
                     holdOpen = false): Response {
  const encoder = new TextEncoder()
  const frame = (payload: unknown) =>
    encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)

  const stream = new ReadableStream({
    start(controller) {
      tokens.forEach((text) => controller.enqueue(frame({ type: 'token', text })))
      if (state !== undefined) {
        controller.enqueue(frame({ type: 'state', state }))
      }
      if (!holdOpen) controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  })
}
