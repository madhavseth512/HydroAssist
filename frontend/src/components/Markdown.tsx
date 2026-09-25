/**
 * The consultant's answers, rendered as what they are.
 *
 * The model writes Markdown — headings, lists, tables, bold terms, and
 * citations like [Custom_Fractured Aquifers, p.5]. Until now every one of those
 * reached the reader as literal characters, because the chat printed
 * `message.content` into a <Text> with `pre-wrap`. An answer that opens with
 * "### 1. Assumptions" and lists "*   **Confined aquifer:** ..." reads as
 * broken software, whatever the hydrogeology says.
 *
 * Raw HTML stays disabled, which is react-markdown's default: the text comes
 * from a language model quoting documents we did not write, so it is content,
 * never markup.
 */
import { TypographyStylesProvider } from '@mantine/core'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export interface MarkdownProps {
  children: string
}

export function Markdown({ children }: MarkdownProps) {
  return (
    <TypographyStylesProvider
      // Mantine's typography styles are sized for a document. In a chat bubble
      // the default margins leave a list floating in whitespace, so headings
      // and blocks are pulled in to roughly one line of separation.
      styles={{
        root: {
          fontSize: 'var(--mantine-font-size-sm)',
          lineHeight: 1.55,
        },
      }}
      className="hydro-markdown"
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Sources cited in an answer are usually external; opening them in
          // place would lose the analysis the user is in the middle of.
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </TypographyStylesProvider>
  )
}
