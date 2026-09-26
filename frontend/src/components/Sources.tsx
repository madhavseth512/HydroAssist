/**
 * What an answer was built from, under the answer itself.
 *
 * Three faults, all reported from the deployed site, and all of them the same
 * mistake — showing the machinery rather than the citation:
 *
 * - The label named a folder. Every chunk's `source` was one of six directory
 *   names, so an answer was attributed to "Custom_Confined Aquifer", which a
 *   reader cannot look up and cannot judge. The server resolves the document's
 *   real identity now, and this shows it with the page.
 * - The passage was clipped to a couple of lines with no way to read the rest,
 *   so there was no way to check an answer against what was actually retrieved.
 *   Each source expands to the whole passage.
 * - There was one list for the whole conversation, which changed every time a
 *   new question was asked. Each answer carries its own now.
 *
 * Where a link exists it is offered; most of this corpus is paywalled and says
 * so, which is more honest than a link that would not open.
 */
import { Accordion, Anchor, Badge, Group, Stack, Text } from '@mantine/core'
import { IconExternalLink } from '@tabler/icons-react'

import type { RetrievedDoc } from '../api/client'

export interface SourcesProps {
  sources: RetrievedDoc[]
  /** Distinguishes the accordions when several answers are on screen. */
  id: string
}

export function Sources({ sources, id }: SourcesProps) {
  if (sources.length === 0) return null

  return (
    <Accordion variant="contained" chevronPosition="left" mt="xs">
      <Accordion.Item value={`sources-${id}`}>
        <Accordion.Control>
          <Text size="xs">
            What this answer read ({sources.length}
            {sources.length === 1 ? ' passage' : ' passages'})
          </Text>
        </Accordion.Control>
        <Accordion.Panel>
          <Stack gap="sm">
            {sources.map((doc, index) => (
              <Stack key={index} gap={2}>
                <Group gap="xs" wrap="wrap">
                  <Text size="xs" fw={600}>{doc.document}</Text>
                  {doc.page ? (
                    <Badge size="xs" variant="light" color="gray">
                      page {doc.page}
                    </Badge>
                  ) : null}
                  {doc.score ? (
                    <Badge size="xs" variant="light" color="gray">
                      relevance {(doc.score * 100).toFixed(0)}%
                    </Badge>
                  ) : null}
                  {doc.url ? (
                    <Anchor href={doc.url} target="_blank" rel="noopener noreferrer"
                            size="xs">
                      <Group gap={2} wrap="nowrap">
                        open <IconExternalLink size={12} />
                      </Group>
                    </Anchor>
                  ) : (
                    <Text size="xs" c="dimmed">no public link</Text>
                  )}
                </Group>
                {/* The whole passage, in a details element so the list stays
                    readable and checking a quotation is still one click. */}
                <details>
                  <summary style={{ cursor: 'pointer' }}>
                    <Text component="span" size="xs" c="dimmed">
                      {doc.preview}
                    </Text>
                  </summary>
                  <Text size="xs" mt={4}
                        style={{ whiteSpace: 'pre-wrap',
                                 fontFamily: 'var(--mantine-font-family-monospace)' }}>
                    {doc.text || doc.preview}
                  </Text>
                </details>
              </Stack>
            ))}
          </Stack>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  )
}
