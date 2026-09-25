/**
 * Menus and a toolbar, which is what scientific software looks like.
 *
 * The workspace read as a web page: a title bar, and every action buried in the
 * panel it belonged to. AQTESOLV, QGIS and Origin all put actions in menus at
 * the top and the frequent ones on a strip beneath, and a reviewer who knows
 * those finds their footing in seconds.
 *
 * The discipline applied here: if it is on the toolbar it is used every
 * session; if it is in a menu it is used sometimes; if neither, it does not
 * exist yet. Nothing here is a placeholder — every item does something today.
 */
import { Button, Divider, Group, Menu, Text } from '@mantine/core'
import {
  IconChartDots, IconDeviceFloppy, IconFileDownload, IconFilePlus,
  IconFolderOpen, IconHelp, IconMessageCircle, IconPlayerPlay, IconScale,
  IconTable,
} from '@tabler/icons-react'

export interface ToolbarProps {
  onNewProject: () => void
  onOpenProject: () => void
  onSaveProject: () => void
  onDownloadReport: () => void
  onGoTo: (panel: string) => void
  onToggleChat: () => void
  chatOpen: boolean
  /** False until a fit exists, which is what the report and comparison need. */
  hasResult: boolean
}

export function Toolbar({
  onNewProject, onOpenProject, onSaveProject, onDownloadReport, onGoTo,
  onToggleChat, chatOpen, hasResult,
}: ToolbarProps) {
  return (
    <Group gap="xs" wrap="nowrap">
      <Menu shadow="md" width={230} position="bottom-start">
        <Menu.Target>
          <Button size="xs" variant="subtle" color="gray">File</Button>
        </Menu.Target>
        <Menu.Dropdown>
          <Menu.Item leftSection={<IconFilePlus size={14} />} onClick={onNewProject}>
            New project
          </Menu.Item>
          <Menu.Item leftSection={<IconFolderOpen size={14} />} onClick={onOpenProject}>
            Open project…
          </Menu.Item>
          <Menu.Item leftSection={<IconDeviceFloppy size={14} />} onClick={onSaveProject}>
            Save project
          </Menu.Item>
          <Menu.Divider />
          <Menu.Item leftSection={<IconTable size={14} />}
                     onClick={() => onGoTo('data')}>
            Import readings…
          </Menu.Item>
          <Menu.Item leftSection={<IconFileDownload size={14} />}
                     onClick={onDownloadReport} disabled={!hasResult}>
            Download PDF report
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>

      <Menu shadow="md" width={230} position="bottom-start">
        <Menu.Target>
          <Button size="xs" variant="subtle" color="gray">Analysis</Button>
        </Menu.Target>
        <Menu.Dropdown>
          <Menu.Item leftSection={<IconChartDots size={14} />}
                     onClick={() => onGoTo('data')}>
            Field data and diagnostic
          </Menu.Item>
          <Menu.Item leftSection={<IconPlayerPlay size={14} />}
                     onClick={() => onGoTo('setup')}>
            Choose a method and run
          </Menu.Item>
          <Menu.Item leftSection={<IconTable size={14} />}
                     onClick={() => onGoTo('results')} disabled={!hasResult}>
            Results and manual matching
          </Menu.Item>
          <Menu.Item leftSection={<IconScale size={14} />}
                     onClick={() => onGoTo('compare')}>
            Compare saved analyses
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>

      <Menu shadow="md" width={260} position="bottom-start">
        <Menu.Target>
          <Button size="xs" variant="subtle" color="gray">Help</Button>
        </Menu.Target>
        <Menu.Dropdown>
          <Menu.Item leftSection={<IconHelp size={14} />} onClick={onToggleChat}>
            Ask the assistant
          </Menu.Item>
          <Menu.Label>About</Menu.Label>
          <Menu.Item disabled>
            <Text size="xs" c="dimmed">
              Seven methods across four aquifer types, fitted by least squares
              and ranked by AICc.
            </Text>
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>

      <Divider orientation="vertical" />

      {/* The strip: what is reached for every session. */}
      <Button size="xs" variant="subtle" color="gray"
              leftSection={<IconFolderOpen size={14} />} onClick={onOpenProject}>
        Open
      </Button>
      <Button size="xs" variant="subtle" color="gray"
              leftSection={<IconDeviceFloppy size={14} />} onClick={onSaveProject}>
        Save
      </Button>
      <Button size="xs" variant="subtle" color="gray"
              leftSection={<IconFileDownload size={14} />}
              onClick={onDownloadReport} disabled={!hasResult}>
        Report
      </Button>
      <Button size="xs" variant={chatOpen ? 'filled' : 'light'}
              leftSection={<IconMessageCircle size={14} />} onClick={onToggleChat}>
        Assistant
      </Button>
    </Group>
  )
}
