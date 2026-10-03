import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import AnswerStatements from '../components/AnswerStatements'

// What an answer had to say and whether it says it. The point pinned here is
// the third outcome: a statement the judge gave no verdict on is not counted
// as missing from the answer.

describe('AnswerStatements', () => {
  it('counts the statements the answer makes and marks each one', () => {
    render(<AnswerStatements wholeCorpus={false} verdicts={[
      { statement: 'The filter is replaced every 6 months.', stated: true },
      { statement: 'The lamp is replaced every 12 months.', stated: false },
    ]} />)
    expect(screen.getByText('1 of 2 present')).toBeInTheDocument()
    expect(screen.getByText('in the answer')).toBeInTheDocument()
    expect(screen.getByText('not in the answer')).toBeInTheDocument()
  })

  it('shows a statement without a verdict as unchecked and not as absent', () => {
    render(<AnswerStatements wholeCorpus={false} verdicts={[
      { statement: 'A', stated: true }, { statement: 'B', stated: null },
    ]} />)
    expect(screen.getByText('not checked')).toBeInTheDocument()
    expect(screen.queryByText('not in the answer')).toBeNull()
  })

  it('gives no count when the judge checked nothing', () => {
    render(<AnswerStatements wholeCorpus={false} verdicts={[
      { statement: 'A', stated: null }, { statement: 'B', stated: null },
    ]} />)
    // "0 of 2 present" would say the answer is wrong; nobody looked.
    expect(screen.queryByText(/of 2 present/)).toBeNull()
    expect(screen.getAllByText('not checked')).toHaveLength(3)
  })

  it('says why a whole-corpus question has no retrieval metrics', () => {
    const verdicts = [{ statement: 'A', stated: true }]
    const { rerender } = render(<AnswerStatements wholeCorpus verdicts={verdicts} />)
    expect(screen.getByText(/Retrieval metrics are not computed/)).toBeInTheDocument()
    rerender(<AnswerStatements wholeCorpus={false} verdicts={verdicts} />)
    expect(screen.queryByText(/Retrieval metrics are not computed/)).toBeNull()
  })

  it('renders nothing for a question with no statements', () => {
    const { container } = render(<AnswerStatements wholeCorpus={false} verdicts={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
