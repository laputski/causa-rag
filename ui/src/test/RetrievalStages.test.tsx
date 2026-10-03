import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import RetrievalStages from '../components/RetrievalStages'
import type { QuestionResult, QuestionStages, SourceRefView } from '../api/client'

// Where the needed source was at each stage of one question's retrieval.
//
// The server decides the ranks and the stage that lost the source. What is
// pinned here is that the page shows that decision and adds none of its own:
// the stage the server named is the one marked, a fragment absent from a list
// is shown as absent, and a run whose lists were removed says so instead of
// offering an empty table.

const ref = (id: string, path = id): SourceRefView => ({ chunk_id: id, structural_path: path, score: 1, chunk_text: `text of ${id}` })

const ids = (n: number, prefix: string) => Array.from({ length: n }, (_, i) => `${prefix}${i + 1}`)

// The source (chunk "s3") is third in the sparse half, absent from the dense
// half, and gone after the merge.
const LOST_IN_FUSION: QuestionStages = {
  stages: [
    { stage: 'dense', size: 4, rank: null, same_as: null, ids: ['d1', 'd2', 'd3', 'd4'] },
    { stage: 'sparse', size: 4, rank: 3, same_as: null, ids: ['s1', 's2', 's3', 'd1'] },
    { stage: 'fusion', size: 3, rank: null, same_as: null, ids: ['d1', 'd2', 's1'] },
    { stage: 'context', size: 3, rank: null, same_as: 'fusion', ids: ['d1', 'd2', 's1'] },
  ],
  lost_at: 'fusion', found: true, needed: ['s3'],
}

function question(stages: QuestionStages, extra: Partial<QuestionResult> = {}): QuestionResult {
  return {
    question_id: 'q1', question: 'q', reference_answer: 'r', generated_answer: 'g', metrics: {},
    source_refs: [ref('d1', 'Manual A / 1'), ref('d2'), ref('s1')],
    sparse_source_refs: [ref('s1'), ref('s2'), ref('s3', 'Manual B / 7'), ref('d1')],
    stages, ...extra,
  }
}

const cell = (stage: string) => document.querySelector(`[data-stage="${stage}"]`) as HTMLElement

describe('RetrievalStages', () => {
  it('marks the stage the server named as the one that lost the source, and no other', () => {
    render(<RetrievalStages qr={question(LOST_IN_FUSION)} />)
    expect(cell('fusion')).toHaveClass('lost')
    expect(within(cell('fusion')).getByText('lost')).toBeInTheDocument()
    for (const stage of ['dense', 'sparse', 'context']) {
      expect(cell(stage)).not.toHaveClass('lost')
    }
  })

  it('shows the place of the source on each stage, and a dash where it is absent', () => {
    render(<RetrievalStages qr={question(LOST_IN_FUSION)} />)
    expect(within(cell('sparse')).getByText('3')).toBeInTheDocument()
    expect(within(cell('sparse')).getByText('of 4')).toBeInTheDocument()
    expect(within(cell('dense')).getByText('—')).toBeInTheDocument()
  })

  it('says which half found the source when the merge lost it', () => {
    render(<RetrievalStages qr={question(LOST_IN_FUSION)} />)
    expect(screen.getByText('The source was found by BM25 (place 3), but it did not make the merged list.')).toBeInTheDocument()
  })

  it('notes a stage that holds the same list as the one before it', () => {
    render(<RetrievalStages qr={question(LOST_IN_FUSION)} />)
    expect(within(cell('context')).getByText('same as after fusion')).toBeInTheDocument()
    expect(within(cell('fusion')).queryByText(/same as/)).toBeNull()
  })

  it('lists fragments by stage with the needed one first and a dot where a fragment is absent', () => {
    render(<RetrievalStages qr={question(LOST_IN_FUSION)} />)
    fireEvent.click(screen.getByText(/Fragments by stage/))
    const rows = screen.getAllByRole('row').slice(1)
    // The needed fragment leads although it reached no later stage.
    expect(within(rows[0]).getByText(/Manual B \/ 7/)).toBeInTheDocument()
    expect(within(rows[0]).getByText('needed')).toBeInTheDocument()
    const cells = within(rows[0]).getAllByRole('cell').map(c => c.textContent)
    expect(cells.slice(1)).toEqual(['·', '3', '·', '·'])
    // Then by how far a fragment got: first in the context comes next.
    expect(within(rows[1]).getByText(/Manual A \/ 1/)).toBeInTheDocument()
    expect(within(rows[1]).getAllByRole('cell').map(c => c.textContent).slice(1)).toEqual(['1', '4', '1', '1'])
  })

  it('shows six fragments and offers the rest', () => {
    const many: QuestionStages = {
      stages: [{ stage: 'retrieval', size: 20, rank: 2, same_as: null, ids: ids(20, 'c') },
               { stage: 'context', size: 5, rank: 2, same_as: null, ids: ids(5, 'c') }],
      lost_at: null, found: true, needed: ['c2'],
    }
    render(<RetrievalStages qr={question(many, { source_refs: [], sparse_source_refs: [] })} />)
    fireEvent.click(screen.getByText(/Fragments by stage/))
    expect(screen.getAllByRole('row')).toHaveLength(1 + 6)
    fireEvent.click(screen.getByText('Show 14 more'))
    expect(screen.getAllByRole('row')).toHaveLength(1 + 20)
    expect(screen.getByText('The source reached the context, place 2.')).toBeInTheDocument()
  })

  it('says the lists were removed and offers no table when they were', () => {
    const frozen: QuestionStages = {
      ...LOST_IN_FUSION,
      stages: LOST_IN_FUSION.stages.map(({ ids: _ids, ...stage }) => stage),
      needed: [],
    }
    render(<RetrievalStages qr={question(frozen)} clearedAt="2026-09-30T10:00:00Z" />)
    // The conclusion survives the removal.
    expect(cell('fusion')).toHaveClass('lost')
    expect(screen.getByText(/intermediate lists were removed on 30\.09\.2026/)).toBeInTheDocument()
    expect(screen.queryByText(/Fragments by stage/)).toBeNull()
  })

  it('shows sizes and no places for a question that names no needed sources', () => {
    const unbounded: QuestionStages = {
      stages: [{ stage: 'retrieval', size: 12, rank: null, same_as: null, ids: ids(12, 'c') },
               { stage: 'context', size: 5, rank: null, same_as: null, ids: ids(5, 'c') }],
      lost_at: null, found: null, needed: [],
    }
    render(<RetrievalStages qr={question(unbounded)} />)
    expect(within(cell('retrieval')).getByText('12 fragments')).toBeInTheDocument()
    // A dash would read as "the source is absent", and no source was named.
    expect(within(cell('retrieval')).queryByText('—')).toBeNull()
    expect(document.querySelector('.stage-cell.lost')).toBeNull()
    expect(screen.getByText(/names no needed sources/)).toBeInTheDocument()
  })

  it('renders nothing for a question stored before stages were computed', () => {
    const { container } = render(<RetrievalStages qr={{ ...question(LOST_IN_FUSION), stages: undefined }} />)
    expect(container).toBeEmptyDOMElement()
  })
})
