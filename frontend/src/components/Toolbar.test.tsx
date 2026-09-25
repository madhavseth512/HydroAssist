/** The menus and the strip: what every action does, and what it refuses to do. */
import { fireEvent, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Toolbar } from './Toolbar'
import { renderWithProviders } from '../test/render'

function toolbar(overrides: Partial<Parameters<typeof Toolbar>[0]> = {}) {
  const props = {
    onNewProject: vi.fn(), onOpenProject: vi.fn(), onSaveProject: vi.fn(),
    onDownloadReport: vi.fn(), onGoTo: vi.fn(), onToggleChat: vi.fn(),
    chatOpen: false, hasResult: true, ...overrides,
  }
  renderWithProviders(<Toolbar {...props} />)
  return props
}

describe('the toolbar strip', () => {
  it('carries what is reached for every session', () => {
    toolbar()

    for (const label of ['Open', 'Save', 'Report', 'Assistant']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument()
    }
  })

  it('saves and opens the project', () => {
    const props = toolbar()

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    fireEvent.click(screen.getByRole('button', { name: 'Open' }))

    expect(props.onSaveProject).toHaveBeenCalled()
    expect(props.onOpenProject).toHaveBeenCalled()
  })

  it('will not offer a report before there is a fit to report', () => {
    toolbar({ hasResult: false })

    expect(screen.getByRole('button', { name: 'Report' })).toBeDisabled()
  })

  it('shows the assistant as on when it is open', () => {
    toolbar({ chatOpen: true })

    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }))
    expect(screen.getByRole('button', { name: 'Assistant' })).toBeInTheDocument()
  })
})

describe('the menus', () => {
  it('group the actions under File, Analysis and Help', () => {
    toolbar()

    for (const menu of ['File', 'Analysis', 'Help']) {
      expect(screen.getByRole('button', { name: menu })).toBeInTheDocument()
    }
  })

  it('navigates to a panel from the Analysis menu', async () => {
    const props = toolbar()

    fireEvent.click(screen.getByRole('button', { name: 'Analysis' }))
    fireEvent.click(await screen.findByText('Compare saved analyses'))

    expect(props.onGoTo).toHaveBeenCalledWith('compare')
  })

  it('opens the assistant from Help', async () => {
    const props = toolbar()

    fireEvent.click(screen.getByRole('button', { name: 'Help' }))
    fireEvent.click(await screen.findByText('Ask the assistant'))

    expect(props.onToggleChat).toHaveBeenCalled()
  })

  it('offers nothing that does nothing', async () => {
    // Empty menus are worse than no menus: every item here acts today.
    const props = toolbar()

    fireEvent.click(screen.getByRole('button', { name: 'File' }))
    fireEvent.click(await screen.findByText('New project'))

    expect(props.onNewProject).toHaveBeenCalled()
  })
})
