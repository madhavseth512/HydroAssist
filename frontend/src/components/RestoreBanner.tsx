/**
 * The offer to put back what the last session was working on.
 *
 * Shown only when the server's project is empty and this browser kept a copy of
 * one that was not — which is what a reload, or the free host stopping the
 * container after fifteen idle minutes, leaves behind.
 *
 * It is an offer, never automatic. Restoring replaces whatever is open, and a
 * silent replacement is exactly the kind of thing that makes people distrust a
 * tool: someone who deliberately started fresh must not find last week's test
 * loaded because the browser still remembered it.
 */
import { Alert, Button, Group, Text } from '@mantine/core'
import { IconHistory } from '@tabler/icons-react'
import { useState } from 'react'

import type { Project } from '../api/client'
import { isUntouched, useRestoreProject } from '../hooks'
import {
  clearSavedProject, dismissRestore, howLongAgo, readSavedProject,
  restoreWasDismissed,
} from '../persistence'

export interface RestoreBannerProps {
  project?: Project
}

export function RestoreBanner({ project }: RestoreBannerProps) {
  // Read once, on mount: the stored copy is overwritten by this session's own
  // autosave as soon as anything is done, and the offer must not change under
  // the reader while they decide.
  const [saved] = useState(readSavedProject)
  const [answered, setAnswered] = useState(false)
  const { restore, isPending } = useRestoreProject()

  if (!saved || answered || restoreWasDismissed()) return null
  // Nothing to offer over: the session already holds work of its own.
  if (!project || !isUntouched(project)) return null

  return (
    <Alert variant="light" color="blue" icon={<IconHistory size={18} />}
           title="Carry on where you left off?">
      <Text size="sm" mb="xs">
        This browser kept a copy of <strong>{saved.summary}</strong> from{' '}
        {howLongAgo(saved.savedAt)}. The workspace itself keeps nothing once a
        session ends, so this is the copy that survived — restoring it loads the
        readings, the setup and the saved analyses.
      </Text>
      <Group gap="xs">
        <Button size="xs" loading={isPending}
                onClick={() => { restore(saved.json); setAnswered(true) }}>
          Restore it
        </Button>
        <Button size="xs" variant="subtle" color="gray"
                onClick={() => { dismissRestore(); setAnswered(true) }}>
          Not now
        </Button>
        <Button size="xs" variant="subtle" color="red"
                onClick={() => { clearSavedProject(); setAnswered(true) }}>
          Forget it
        </Button>
      </Group>
    </Alert>
  )
}
