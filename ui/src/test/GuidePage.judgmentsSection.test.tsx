import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import GuidePage from '../pages/GuidePage'

// Replaces GuidePage.pinsSection.test.tsx. The section it tested
// explained a mechanism that overrode retrieval; leaving that explanation in
// place would have been actively misleading, since the override is now off.

function renderGuide() {
  return render(<MemoryRouter><GuidePage /></MemoryRouter>)
}

describe('GuidePage — judgments section', () => {
  it('explains that a judgment records an observation and changes nothing', () => {
    renderGuide()
    fireEvent.click(screen.getByRole('button', { name: 'Relevance judgments' }))
    expect(screen.getByText(/statement about what is correct/i)).toBeInTheDocument()
    expect(screen.getByText(/A judgment changes nothing about how retrieval works/i)).toBeInTheDocument()
  })

  it('states plainly that the retrieval override is off and never affects a user answer', () => {
    // The reviewer has to be able to tell that recording a verdict will not
    // silently alter what the served system answers.
    renderGuide()
    fireEvent.click(screen.getByRole('button', { name: 'Relevance judgments' }))
    expect(screen.getByText(/off by default/i)).toBeInTheDocument()
    expect(screen.getByText(/never affects a user's answer/i)).toBeInTheDocument()
  })

  it('keeps the feedback triage explanation, now as a section of its own', () => {
    // Split out because two mechanisms with different UI locations were
    // reading as one long topic, and the triage panel sat past everything
    // about judgments.
    renderGuide()
    fireEvent.click(screen.getByRole('button', { name: 'Feedback triage' }))
    expect(screen.getByRole('heading', { name: /How the classification is performed/i })).toBeInTheDocument()
  })
})

describe('GuidePage — deep link', () => {
  it('opens the judgments section from ?section=judgments, not the overview', () => {
    // JudgmentsPage links here. The guide reads a query parameter, so a link
    // written as /guide#judgments lands silently on "Overview" — which is
    // exactly what shipped until this test was added.
    render(
      <MemoryRouter initialEntries={['/guide?section=judgments']}><GuidePage /></MemoryRouter>,
    )
    expect(screen.getByText(/statement about what is correct/i)).toBeInTheDocument()
  })
})
