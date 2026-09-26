/**
 * Server state, in one place.
 *
 * The project, the readings, the result and the conversation all live on the
 * server; these hooks read them back rather than keeping a second copy in the
 * browser that can disagree. Mutations invalidate what they changed, so one
 * action never leaves two panels showing different truths.
 */
import { notifications } from '@mantine/notifications'
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
} from '@tanstack/react-query'
import { useEffect } from 'react'

import {
  ApiError, api, type ChatState, type PreparedData, type Project,
  type ProjectInfo, type Setup,
} from './api/client'
import { dismissRestore, saveProject } from './persistence'

export const keys = {
  session: ['session'] as const,
  project: ['project'] as const,
  testTypes: ['test-types'] as const,
  methods: (testType: string) => ['methods', testType] as const,
  // The row count is part of the key: the preview and the whole table are
  // different answers to the same question, and one must not serve as the other.
  readings: (testId: string, rows?: number) =>
    rows ? (['readings', testId, rows] as const) : (['readings', testId] as const),
  result: ['result'] as const,
  comparison: (testId: string) => ['comparison', testId] as const,
  chat: ['chat'] as const,
}

/** Tell the user what the server said, in its own words. */
export function reportError(error: unknown, title = 'That did not work') {
  const message =
    error instanceof ApiError ? error.message : (error as Error).message
  notifications.show({ color: 'red', title, message })
}

// ── Reads ────────────────────────────────────────────────────────────────────

export function useSession() {
  return useQuery({ queryKey: keys.session, queryFn: api.session })
}

/** The open project.
 *
 * `enabled` exists for callers that render before the session does. Asking for
 * the project without a session cookie creates a second session on the server,
 * and then the page shows one session's project while uploads go to another —
 * the first-visit race that App.test.tsx's session-ordering test was written
 * for. Panels mount only once the session is ready, so they need no flag; the
 * shell itself does.
 */
export function useProject(enabled = true) {
  return useQuery({ queryKey: keys.project, queryFn: api.project, enabled })
}

export function useTestTypes() {
  return useQuery({ queryKey: keys.testTypes, queryFn: api.testTypes,
                    staleTime: Infinity })
}

export function useMethods(testType: string) {
  return useQuery({
    queryKey: keys.methods(testType),
    queryFn: () => api.methods(testType),
  })
}

/** Every method for the test type, gating ignored — what the cards need to be
 * able to show an excluded method at all. */
export function useAllMethods(testType: string) {
  return useQuery({
    queryKey: [...keys.methods(testType), 'all'],
    queryFn: () => api.methods(testType, false),
    staleTime: Infinity,
  })
}

/** The test every panel is showing.
 *
 * Four panels derived this independently from `useProject()`, which is four
 * chances for them to disagree about which test is on screen.
 */
export function useActiveTest() {
  const project = useProject()
  return project.data?.tests.find(
    (test) => test.id === project.data?.active_test_id)
}

/** The readings a test already holds. Absent is a normal state, not an error. */
/** A test's readings. `rows` asks for more of the table than the preview. */
export function useReadings(testId: string | undefined, enabled = true,
                            rows?: number) {
  return useQuery({
    queryKey: keys.readings(testId ?? '', rows),
    queryFn: () => api.readings(testId!, rows),
    enabled: Boolean(testId) && enabled,
    retry: false,
  })
}

export function useCurrentResult(enabled = true) {
  return useQuery({
    queryKey: keys.result,
    queryFn: api.currentResult,
    retry: false,
    // The shell asks for this to decide whether the report button is usable,
    // and it must not ask before the session cookie exists: a cookie-less
    // request mints a second session, which is the race App.test.tsx pins.
    enabled,
  })
}

export function useComparison(testId: string | undefined) {
  return useQuery({
    queryKey: keys.comparison(testId ?? ''),
    queryFn: () => api.comparison(testId!),
    enabled: Boolean(testId),
  })
}

export function useChat() {
  return useQuery({ queryKey: keys.chat, queryFn: api.chat })
}

// ── Writes ───────────────────────────────────────────────────────────────────

type MutationExtras<TData, TVars> = Omit<
  UseMutationOptions<TData, unknown, TVars>,
  'mutationFn'
>

/** A mutation that refreshes the project, and reports failures. */
function useProjectMutation<TData, TVars>(
  mutationFn: (vars: TVars) => Promise<TData>,
  options: MutationExtras<TData, TVars> & { invalidateAll?: boolean } = {},
) {
  const queryClient = useQueryClient()
  const { invalidateAll, onSuccess, onError, ...rest } = options
  return useMutation<TData, unknown, TVars>({
    mutationFn,
    onSuccess: (...args: Parameters<NonNullable<typeof onSuccess>>) => {
      queryClient.invalidateQueries({ queryKey: keys.project })
      if (invalidateAll) {
        // Switching or replacing a test invalidates the analysis too: the old
        // result describes data this workspace is no longer showing.
        // resetQueries, not removeQueries: removing the entry detaches the results
      // panel that is subscribed to it, so the next result would never show.
      queryClient.resetQueries({ queryKey: keys.result })
        queryClient.invalidateQueries({ queryKey: keys.chat })
      }
      onSuccess?.(...args)
    },
    onError: (...args: Parameters<NonNullable<typeof onError>>) => {
      reportError(args[0])
      onError?.(...args)
    },
    ...rest,
  })
}

export function useOpenProject() {
  return useProjectMutation((file: File) => api.openProject(file),
                            { invalidateAll: true })
}

export function useNewProject() {
  return useProjectMutation(() => api.newProject(), { invalidateAll: true })
}

export function useUpdateInfo() {
  return useProjectMutation((info: ProjectInfo) => api.updateInfo(info))
}

export function useAddTest() {
  return useProjectMutation(() => api.addTest(), { invalidateAll: true })
}

export function useActivateTest() {
  return useProjectMutation((testId: string) => api.activateTest(testId),
                            { invalidateAll: true })
}

export function useUpdateTest() {
  return useProjectMutation(
    ({ testId, ...details }: {
      testId: string
      name?: string
      pumping_well?: string
      observation_well?: string
    }) => api.updateTest(testId, details),
  )
}

/** Readings arriving by any route land in the same caches.
 *
 * Uploading a file and pasting columns differ only in how the bytes travel, so
 * the cache work is shared: two copies would drift, and the one that drifted
 * would leave a stale fit on screen beside new data.
 */
function useReadingsMutation<TInput>(
  testId: string | undefined,
  send: (testId: string, input: TInput) => Promise<PreparedData>,
  failureTitle: string,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (input: TInput) => {
      if (!testId) {
        return Promise.reject(new Error('The project is still loading — try again in a moment.'))
      }
      return send(testId, input)
    },
    onSuccess: (prepared) => {
      queryClient.setQueryData(keys.readings(testId ?? ''), prepared)
      queryClient.invalidateQueries({ queryKey: keys.project })
      // By prefix: the gating depends on the data, whatever test type is shown.
      queryClient.invalidateQueries({ queryKey: ['methods'] })
      // New data retires the previous fit, on the server and here.
      // resetQueries, not removeQueries: removing the entry detaches the results
      // panel that is subscribed to it, so the next result would never show.
      queryClient.resetQueries({ queryKey: keys.result })
      queryClient.invalidateQueries({ queryKey: keys.chat })
    },
    onError: (error) => reportError(error, failureTitle),
  })
}

export function useUploadReadings(testId: string | undefined) {
  return useReadingsMutation<File>(
    testId, (id, file) => api.uploadReadings(id, file),
    'That file could not be used')
}

/** Columns pasted from a spreadsheet, which is where the numbers usually are. */
export function usePasteReadings(testId: string | undefined) {
  return useReadingsMutation<string>(
    testId, (id, text) => api.pasteReadings(id, text),
    'Those readings could not be used')
}

/** Read a photograph of a field sheet into a grid to confirm.
 *
 * Deliberately not a `useReadingsMutation`: this stores nothing and invalidates
 * nothing, because nothing has become readings. The corrected grid goes through
 * `usePasteReadings`, so an image can only reach a fit by way of a person.
 */
export function useReadImage(testId: string | undefined) {
  return useMutation({
    mutationFn: (file: File) => {
      if (!testId) {
        return Promise.reject(new Error('The project is still loading — try again in a moment.'))
      }
      return api.readImage(testId, file)
    },
    onError: (error) => reportError(error, 'That photograph could not be read'),
  })
}

/** Run a fit. The server stores the setup on the test before fitting, so the
 * form, the project and the result describe the same test afterwards. */
export function useRunAnalysis() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ method, setup }: { method: string; setup: Setup }) =>
      api.run(method, setup),
    onSuccess: (result) => {
      queryClient.setQueryData(keys.result, result)
      queryClient.invalidateQueries({ queryKey: keys.project })
      // The result is published into the conversation on the server.
      queryClient.invalidateQueries({ queryKey: keys.chat })
    },
    onError: (error) => reportError(error, 'The analysis could not run'),
  })
}

/** Record the current result against the active test. */
export function useSaveAnalysis() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.saveAnalysis(),
    onSuccess: (comparison) => {
      queryClient.setQueryData(keys.comparison(comparison.test_id), comparison)
      queryClient.invalidateQueries({ queryKey: keys.project })
    },
    onError: (error) => reportError(error, 'The analysis was not saved'),
  })
}

/** The hand-matched curve. Called when a slider is released, not on every
 * pixel of the drag, so the free-tier server is not asked for a figure per
 * mouse movement. */
export function useManualMatch() {
  return useMutation({
    mutationFn: ({ T, S }: { T: number; S: number | null }) => api.manualMatch(T, S),
    onError: (error) => reportError(error, 'Manual matching failed'),
  })
}

export function useRemoveAnalysis(testId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (analysisId: string) => api.removeAnalysis(testId!, analysisId),
    onSuccess: (comparison) => {
      queryClient.setQueryData(keys.comparison(comparison.test_id), comparison)
      queryClient.invalidateQueries({ queryKey: keys.project })
    },
    onError: (error) => reportError(error),
  })
}

// ── Conversation ─────────────────────────────────────────────────────────────

function useChatMutation<TVars>(mutationFn: (vars: TVars) => Promise<unknown>,
                                title: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn,
    onSuccess: (state) => {
      queryClient.setQueryData(keys.chat, state)
      // The method list's default follows what the conversation settled on.
      queryClient.invalidateQueries({ queryKey: ['methods'] })
    },
    onError: (error) => reportError(error, title),
  })
}

/** Send a question, and show it in the transcript straight away.
 *
 * The answer takes a few seconds: a classification call, retrieval, then the
 * answer itself. Waiting for all of that before echoing the question left the
 * words sitting in the box, so it read as though nothing had been sent and the
 * same question got asked twice. The message is added to the cached conversation
 * immediately and the server's own copy replaces it on success; a failure puts
 * the transcript back as it was.
 */
export function useSendMessage() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (message: string) => api.sendMessage(message),
    onMutate: (message: string) => {
      const previous = queryClient.getQueryData<ChatState>(keys.chat)
      if (previous) {
        queryClient.setQueryData<ChatState>(keys.chat, {
          ...previous,
          messages: [...(previous.messages ?? []),
                     { role: 'user', content: message }],
          // The sources belong to the previous answer, not to a question that
          // has not been answered yet.
          retrieved: [],
        })
      }
      return { previous }
    },
    onSuccess: (state) => {
      queryClient.setQueryData(keys.chat, state)
      queryClient.invalidateQueries({ queryKey: ['methods'] })
    },
    onError: (error, _message, context) => {
      if (context?.previous) {
        queryClient.setQueryData(keys.chat, context.previous)
      }
      reportError(error, 'The consultant did not answer')
    },
  })
}

export function useConfirmMethod() {
  return useChatMutation((method: string) => api.confirmMethod(method),
                         'The method was not set')
}

export function useResetChat() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.resetChat(),
    onSuccess: (state) => {
      queryClient.setQueryData(keys.chat, state)
      // The server clears the last result along with the conversation.
      // resetQueries, not removeQueries: removing the entry detaches the results
      // panel that is subscribed to it, so the next result would never show.
      queryClient.resetQueries({ queryKey: keys.result })
      queryClient.invalidateQueries({ queryKey: ['methods'] })
    },
    onError: (error) => reportError(error),
  })
}

export function useSaveSetup(testId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (setup: Setup) => api.saveSetup(testId!, setup),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.project }),
    onError: (error) => reportError(error),
  })
}


// ── Keeping the work across a reload ─────────────────────────────────────────

/**
 * Keep a copy of the project file in this browser whenever the project changes.
 *
 * Hung off the project query rather than off each mutation: everything that
 * changes a project already invalidates that query, so there is one place to
 * watch and no way for a new mutation to forget to autosave. The copy is fetched
 * from the server rather than assembled here, so what is stored is exactly the
 * file the Save button writes.
 *
 * A fresh, untouched project is not worth keeping and must not overwrite a good
 * copy from the last visit — the offer to restore it is the whole point.
 */
export function useAutosaveProject(enabled: boolean) {
  const project = useProject(enabled)
  const data = enabled ? project.data : undefined
  // The summary changes whenever anything in the project does, which makes it a
  // sound trigger and a cheap one.
  const signature = data ? `${data.summary}|${data.tests?.length ?? 0}` : ''

  useEffect(() => {
    if (!data || !signature) return
    if (isUntouched(data)) return
    let cancelled = false
    api.projectFile()
      .then((json) => { if (!cancelled) saveProject(json, data.summary) })
      .catch(() => undefined)      // a failed autosave is not worth a notification
    return () => { cancelled = true }
  }, [signature])                  // eslint-disable-line react-hooks/exhaustive-deps
}

/** Whether this is a new project nobody has put anything into yet. */
export function isUntouched(project: Project): boolean {
  const tests = project.tests ?? []
  return tests.every((test) => !test.n_readings && !test.n_analyses)
}

/** Put a kept project back on the server, by the route a reopened file takes. */
export function useRestoreProject() {
  const open = useOpenProject()
  return {
    ...open,
    restore: (json: string) => {
      const file = new File([json], 'restored.hydroassist.json',
                            { type: 'application/json' })
      open.mutate(file, { onSuccess: () => dismissRestore() })
    },
  }
}
