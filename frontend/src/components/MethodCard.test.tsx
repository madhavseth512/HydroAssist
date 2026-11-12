/** The method card: choosing, and being told why you cannot. */
import { fireEvent, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { MethodCard } from './MethodCard'
import { renderWithProviders } from '../test/render'

const THEIS = {
  id: 'theis_1935', label: 'Theis (1935)', year: 1935, authors: ['Theis, C.V.'],
  citation: '', aquifer_types: ['confined'], test_types: ['constant_rate'],
  parameters: ['T', 'S'], data_source: 'observation_well', requires: [],
  ui_help: 'Full-curve type-curve fit.', selection_rationale: '',
  assumptions: ['Confined aquifer of infinite lateral extent',
                'Fully penetrating well; flow is horizontal'],
  needs_well_geometry: false, estimates_storativity: true, uses_pumping_rate: true,
}

describe('choosing a method', () => {
  it('shows what the method is for and how many assumptions it carries', () => {
    renderWithProviders(<MethodCard method={THEIS} />)

    expect(screen.getByText('Theis (1935)')).toBeInTheDocument()
    expect(screen.getByText('Full-curve type-curve fit.')).toBeInTheDocument()
    expect(screen.getByText('2 assumptions')).toBeInTheDocument()
  })

  it('reports what it estimates', () => {
    renderWithProviders(<MethodCard method={THEIS} />)

    expect(screen.getByText('2 parameters')).toBeInTheDocument()
    expect(screen.queryByText('no S')).toBeNull()
  })

  it('says when a method does not estimate storativity', () => {
    renderWithProviders(
      <MethodCard method={{ ...THEIS, estimates_storativity: false }} />)

    expect(screen.getByText('no S')).toBeInTheDocument()
  })

  it('selects on click', () => {
    const onSelect = vi.fn()
    renderWithProviders(<MethodCard method={THEIS} onSelect={onSelect} />)

    fireEvent.click(screen.getByRole('button', { name: /Theis/ }))

    expect(onSelect).toHaveBeenCalledWith('theis_1935')
  })
})

describe('a method this data cannot support', () => {
  const REASON = 'Cooper-Jacob (1946) may not be valid: u < 0.05 is not reached.'

  it('is shown with the reason rather than hidden', () => {
    // The exclusions are the part that teaches. A method that silently
    // disappears tells a practitioner nothing about their data.
    renderWithProviders(
      <MethodCard method={THEIS} unavailableBecause={REASON} />)

    expect(screen.getByText(REASON)).toBeInTheDocument()
  })

  it('cannot be chosen', () => {
    const onSelect = vi.fn()
    renderWithProviders(
      <MethodCard method={THEIS} unavailableBecause={REASON} onSelect={onSelect} />)

    fireEvent.click(screen.getByRole('button', { name: /not available/ }))

    expect(onSelect).not.toHaveBeenCalled()
  })
})

describe('after the fits are in', () => {
  it('carries the result and its standing', () => {
    renderWithProviders(
      <MethodCard method={THEIS} result={{
        rank: 1, transmissivity: '215.3 m²/day', storativity: '1.02e-4',
        rSquared: 0.9991, deltaAicc: 0, weight: 0.74,
        evidence: 'best', notes: 2,
      }} />)

    expect(screen.getByText('best fit')).toBeInTheDocument()
    expect(screen.getByText('T 215.3 m²/day')).toBeInTheDocument()
    expect(screen.getByText(/ΔAICc 0.0/)).toBeInTheDocument()
  })

  it('shows the validity notes beside the ranking, not behind it', () => {
    // "Best fit" and "right model" are different claims: a method that does
    // not physically apply can still win on AICc.
    renderWithProviders(
      <MethodCard method={THEIS} result={{ rank: 1, rSquared: 0.99, notes: 3 }} />)

    expect(screen.getByText('3 validity notes')).toBeInTheDocument()
  })
})
