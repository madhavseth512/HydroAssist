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

import {
  ApiError, api, type PreparedData, type ProjectInfo, type Setup,
} from './api/client'

export const keys = {
  session: ['session'] as const,
  project: ['project'] as const,
  testTypes: ['test-types'] as const,
  methods: (testType: string) => ['methods', testType] as const,
  readings: (testId: string) => ['readings', testId] as const,
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

export function useProject() {
  return useQuery({ queryKey: keys.project, queryFn: api.project })
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

/** The readings a test already holds. Absent is a normal state, not an error. */
export function useReadings(testId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: keys.readings(testId ?? ''),
    queryFn: () => api.readings(testId!),
    enabled: Boolean(testId) && enabled,
    retry: false,
  })
}

export function useCurrentResult() {
  return useQuery({
    queryKey: keys.result,
    queryFn: api.currentResult,
    retry: false,
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

export function useSendMessage() {
  return useChatMutation((message: string) => api.sendMessage(message),
                         'The consultant did not answer')
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
