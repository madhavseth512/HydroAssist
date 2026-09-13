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
  Accordion,
  Alert,
  Badge,
  Button,
  Drawer,
  Group,
  Loader,
  Paper,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Text,
  Textarea,
} from '@mantine/core'
import { IconSend } from '@tabler/icons-react'
import { useEffect, useRef, useState } from 'react'

import { api, type ChatMessage, type Suggestion } from '../api/client'
import {
  reportError,
  useChat,
  useConfirmMethod,
  useMethods,
  useProject,
  useResetChat,
  useSendMessage,
} from '../hooks'

export interface ChatDrawerProps {
  opened: boolean
  onClose: () => void
}

export function ChatDrawer({ opened, onClose }: ChatDrawerProps) {
  const chat = useChat()
  const send = useSendMessage()
  const reset = useResetChat()
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  const messages: ChatMessage[] = chat.data?.messages ?? []
  const context = chat.data?.context
  const retrieved = chat.data?.retrieved ?? []

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: 'end' })
  }, [messages.length, send.isPending])

  const submit = () => {
    const message = draft.trim()
    if (!message || send.isPending) return
    send.mutate(message, { onSuccess: () => setDraft('') })
  }

  return (
    <Drawer opened={opened} onClose={onClose} position="right" size="lg"
            title="Consultant" padding="md">
      <Stack gap="sm" h="calc(100vh - 90px)">
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
            {context.method && (
              <Badge variant="light" color={context.suggestion_confirmed ? 'teal' : 'blue'}>
                {context.method}{context.suggestion_confirmed ? ' ✓' : ''}
              </Badge>
            )}
            {context.has_result && <Badge variant="light" color="grape">result loaded</Badge>}
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
                <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{message.content}</Text>
              </Paper>
            ))}
            {send.isPending && <Loader size="sm" />}

            {chat.data?.suggestion && <SuggestionCard suggestion={chat.data.suggestion} />}

            {retrieved.length > 0 && (
              <Accordion variant="contained" chevronPosition="left">
                <Accordion.Item value="sources">
                  <Accordion.Control>Sources consulted ({retrieved.length})</Accordion.Control>
                  <Accordion.Panel>
                    <Stack gap="xs">
                      {retrieved.map((doc, index) => (
                        <div key={index}>
                          <Text size="xs" fw={500}>
                            {doc.source}
                            {doc.method ? ` · ${doc.method}` : ''}
                            {doc.aquifer ? ` · ${doc.aquifer}` : ''}
                          </Text>
                          <Text size="xs" c="dimmed">{doc.preview}</Text>
                        </div>
                      ))}
                    </Stack>
                  </Accordion.Panel>
                </Accordion.Item>
              </Accordion>
            )}
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
    </Drawer>
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
