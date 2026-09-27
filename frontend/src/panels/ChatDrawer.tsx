/**
 * The consultation: chat drives, the form confirms.
 *
 * The consultant recommends a method — from the derivative when readings are
 * loaded, otherwise from what the user describes — and the suggestion card asks
 * the user to accept it or pick another. Accepting sets the method the setup
 * form preselects; the fit itself is always run from the form, where every
 * number stays visible.
 *
 * The model runs on Gemini's free tier, under which Google may use what is
 * sent to improve its products. The drawer says so before anyone types.
 */
import {
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Paper,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Text,
  Textarea,
  Tooltip,
} from '@mantine/core'
import { IconArrowRight, IconSend } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'

import { api, type ChatMessage, type Suggestion } from '../api/client'
import { Markdown } from '../components/Markdown'
import { Sources } from '../components/Sources'
import {
  reportError,
  useChat,
  useConfirmMethod,
  useCurrentResult,
  useMethods,
  useProject,
  useResetChat,
  useSendMessage,
} from '../hooks'

export interface ChatDrawerProps {
  /** Scrolls a workspace panel into view when the guide offers to go there.
   * The assistant never navigates by itself — this runs on a click. */
  onGoTo?: (panel: string) => void
}

export function ChatDrawer({ onGoTo }: ChatDrawerProps) {
  const chat = useChat()
  // What the workspace actually ran, which is not the same as what is being
  // talked about. Read from the result rather than from the conversation.
  const result = useCurrentResult()
  const analysing = result.data?.success ? result.data.method : ''
  // The answer as it is being written. Held in a ref as well as in state: the
  // tokens arrive faster than React re-renders, and appending to the ref cannot
  // lose one to a stale closure.
  const partial = useRef('')
  const [writing, setWriting] = useState('')
  const send = useSendMessage((text) => {
    partial.current += text
    setWriting(partial.current)
  })
  const reset = useResetChat()
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  const messages: ChatMessage[] = chat.data?.messages ?? []
  const context = chat.data?.context

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: 'end' })
  }, [messages.length, send.isPending])

  const submit = () => {
    const message = draft.trim()
    if (!message || send.isPending) return
    // Cleared now, not when the answer arrives: the question is already on its
    // way and already in the transcript. If it fails, the draft comes back so
    // nothing anyone typed is lost.
    setDraft('')
    partial.current = ''
    setWriting('')
    send.mutate(message, {
      onError: () => setDraft(message),
      // The finished answer is in the conversation now; the partial copy would
      // otherwise sit underneath it as a duplicate.
      onSettled: () => { partial.current = ''; setWriting('') },
    })
  }

  return (
    <Stack gap="sm" h="100%">
      <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
        <Alert variant="light" color="gray" p="xs">
          <Text size="xs">
            Answers come from Gemini's free tier, under which Google may use what
            you send to improve its products. Do not paste confidential client
            information.
          </Text>
        </Alert>

        {context && (
          <Group gap="xs">
            <Badge variant="light">aquifer: {context.aquifer}</Badge>
            {/* Two different things, which used to be one badge and so read as
                the same thing. What the workspace is analysing is a fact about
                the analysis and only the user changes it; what the conversation
                is about changes with every message. Mentioning a method in here
                is a question about it, not an instruction to switch. */}
            {analysing && (
              <Tooltip label="The method the workspace ran. Only you change this.">
                <Badge variant="filled" color="grape">analysing: {analysing}</Badge>
              </Tooltip>
            )}
            {context.method && context.method !== analysing && (
              <Tooltip label={context.suggestion_confirmed
                ? 'Accepted for the workspace'
                : 'What this conversation is about — the analysis is unchanged'}>
                <Badge variant="light"
                       color={context.suggestion_confirmed ? 'teal' : 'blue'}>
                  discussing: {context.method}
                  {context.suggestion_confirmed ? ' ✓' : ''}
                </Badge>
              </Tooltip>
            )}
          </Group>
        )}

        <ScrollArea style={{ flex: 1 }} offsetScrollbars>
          <Stack gap="xs">
            {messages.length === 0 && !chat.isPending && (
              <Text size="sm" c="dimmed">
                Ask about pumping test methods — when to use Cooper-Jacob instead of
                Theis, what leakage looks like in the data, or what a result means.
                Upload readings first and the recommendation is read off the
                derivative of your own curve.
              </Text>
            )}
            {messages.map((message, index) => (
              <Paper key={index} p="sm" radius="md" withBorder
                     bg={message.role === 'user' ? 'var(--mantine-color-blue-light)' : undefined}
                     ml={message.role === 'user' ? 'xl' : 0}
                     mr={message.role === 'user' ? 0 : 'xl'}>
                <Text size="xs" c="dimmed" mb={4}>
                  {message.role === 'user' ? 'You' : 'Consultant'}
                </Text>
                {message.role === 'user' ? (
                  // What the user typed, shown as typed. Rendering their own
                  // asterisks as bold would be a small act of rewriting.
                  <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{message.content}</Text>
                ) : (
                  <Markdown>{message.content}</Markdown>
                )}
                {/* Under the answer they belong to, not under the conversation:
                    one shared list described the latest answer and was wrong
                    beneath every earlier one. */}
                <Sources sources={message.sources ?? []} id={String(index)} />
              </Paper>
            ))}
            {/* The answer as it arrives. Rendered as markdown like any other,
                so a heading or a list does not appear as characters first and
                reflow into place afterwards. */}
            {writing && (
              <Paper p="sm" radius="md" withBorder mr="xl">
                <Text size="xs" c="dimmed" mb={4}>Consultant</Text>
                <Markdown>{writing}</Markdown>
              </Paper>
            )}
            {send.isPending && !writing && <Loader size="sm" />}

            {chat.data?.suggestion && <SuggestionCard suggestion={chat.data.suggestion} />}

            {(chat.data?.actions ?? []).map((action, index) => (
              <Button key={index} size="xs" variant="light" w="fit-content"
                      leftSection={<IconArrowRight size={14} />}
                      onClick={() => action.target && onGoTo?.(action.target)}>
                {action.label}
              </Button>
            ))}

            <div ref={bottom} />
          </Stack>
        </ScrollArea>

        <Textarea
          placeholder="Ask about pumping test methods…"
          aria-label="Message"
          autosize
          minRows={2}
          maxRows={6}
          value={draft}
          onChange={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
        />
        <Group justify="space-between">
          <Group gap="xs">
            <Button size="xs" variant="subtle" color="gray" loading={reset.isPending}
                    onClick={() => reset.mutate(undefined)}>
              Reset conversation
            </Button>
            {messages.length > 0 && (
              <Button size="xs" variant="subtle" color="gray"
                      onClick={() => api.exportChat().catch((error) => reportError(error))}>
                Export
              </Button>
            )}
          </Group>
          <Button size="xs" rightSection={<IconSend size={14} />}
                  disabled={!draft.trim()} loading={send.isPending} onClick={submit}>
            Send
          </Button>
        </Group>
      </Stack>
    </Stack>
  )
}

function SuggestionCard({ suggestion }: { suggestion: Suggestion }) {
  const confirm = useConfirmMethod()
  const project = useProject()
  const activeTest = project.data?.tests.find((t) => t.id === project.data?.active_test_id)
  const methods = useMethods(activeTest?.setup.test_type ?? 'constant_rate')
  const labels = (methods.data?.offered ?? []).map((m) => m.label)
  const [override, setOverride] = useState<string | null>(null)

  // An aquifer type with no calculator is said plainly, with nothing to accept.
  if (suggestion.supported === false || !suggestion.method) {
    return (
      <Alert color="yellow" variant="light" title="No calculator for this case yet">
        {suggestion.reason}
      </Alert>
    )
  }

  return (
    <Paper withBorder p="sm" radius="md">
      <Stack gap="xs">
        <Text fw={600}>Suggested method: {suggestion.method}</Text>
        <Text size="sm">{suggestion.reason}</Text>
        <Progress value={(suggestion.confidence ?? 0) * 100} size="sm"
                  aria-label="Confidence" />
        <Text size="xs" c="dimmed">
          Confidence {Math.round((suggestion.confidence ?? 0) * 100)}%
        </Text>
        <Group gap="xs" align="flex-end">
          <Button size="xs" loading={confirm.isPending}
                  onClick={() => confirm.mutate(suggestion.method!)}>
            Use this method
          </Button>
          {labels.length > 0 && (
            <>
              <Select size="xs" placeholder="…or choose another" aria-label="Other method"
                      data={labels} value={override} onChange={setOverride} w={220} />
              <Button size="xs" variant="light" disabled={!override}
                      onClick={() => override && confirm.mutate(override)}>
                Use selected
              </Button>
            </>
          )}
        </Group>
      </Stack>
    </Paper>
  )
}
