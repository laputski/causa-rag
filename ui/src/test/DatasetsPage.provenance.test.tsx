import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import DatasetsPage from '../pages/DatasetsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// the bugfix — a reviewer-promoted question used to render in its
// original array position (e.g. 51st of 51, easy to miss) with the exact
// same muted-gray color as every other origin. See the design notes
// "Promoting a reviewed question into a golden dataset" for the fix design.

const listMock = vi.fn().mockResolvedValue([])
const getMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    datasets: {
      list: (realmId?: string | null) => listMock(realmId),
      get: (filename: string) => getMock(filename),
      create: vi.fn(),
      delete: vi.fn(),
      addQuestion: vi.fn(),
      updateQuestion: vi.fn(),
      deleteQuestion: vi.fn(),
      addQuestionsBatch: vi.fn(),
    },
    models: vi.fn().mockResolvedValue([]),
    corpus: { collections: vi.fn().mockResolvedValue([]) },
    generationPresets: { list: vi.fn().mockResolvedValue([]) },
    generate: { questions: vi.fn() },
  },
  referenceAnswerOf: (q: { reference_answer?: string; ground_truth?: string } | undefined) =>
    q?.reference_answer ?? q?.ground_truth ?? '',
}))

const DATASET = {
  id: 'ds1', name: 'cosmos1', version: 'v0', speed: 'full', filename: 'cosmos1.v0.full.jsonl',
  realm_id: 'acme', count: 3,
  // Deliberately NOT in display order — reviewer_feedback last in the
  // array (as it would be if promoted after the other two already existed).
  questions: [
    { id: 'q1', question: 'Manual question?', reference_answer: 'A1', article_refs: [], provenance: { origin: 'manual' as const } },
    { id: 'q2', question: 'Generated question?', reference_answer: 'A2', article_refs: [], provenance: { origin: 'generated' as const, model: 'qwen3:8b', corpus_id: 'handbook_01' } },
    { id: 'q3', question: 'Promoted question?', reference_answer: 'A3', article_refs: [], provenance: { origin: 'reviewer_feedback' as const, source_run_id: 'run1' } },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  listMock.mockResolvedValue([])
})

describe('DatasetsPage — provenance sort and color', () => {
  it('renders reviewer_feedback-origin questions first, regardless of array position', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    stubRealmFetch('acme')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'acme')

    fireEvent.click(await screen.findByText('cosmos1'))
    await waitFor(() => expect(screen.getByText('Promoted question?')).toBeInTheDocument())

    const questionEls = screen.getAllByText(/question\?$/)
    expect(questionEls.map(el => el.textContent)).toEqual([
      'Promoted question?', 'Manual question?', 'Generated question?',
    ])
  })

  it('gives each origin a visually distinct color, not one flat muted gray', async () => {
    listMock.mockResolvedValueOnce([DATASET])
    getMock.mockResolvedValue(DATASET)
    stubRealmFetch('acme')
    renderWithRealm(<DatasetsPage />, '/data/qa', 'acme')

    fireEvent.click(await screen.findByText('cosmos1'))
    await waitFor(() => expect(screen.getByText(/from reviewer feedback/)).toBeInTheDocument())

    const reviewerLabel = screen.getByText(/from reviewer feedback/)
    const generatedLabel = screen.getByText(/^generated \(/)
    const manualLabel = screen.getByText('manual')

    expect(reviewerLabel.style.color).toBe('var(--color-warning)')
    expect(generatedLabel.style.color).toBe('var(--color-primary)')
    expect(manualLabel.style.color).toBe('var(--color-text-muted)')
    // The three must actually differ — not just present, but distinguishable.
    const colors = new Set([reviewerLabel.style.color, generatedLabel.style.color, manualLabel.style.color])
    expect(colors.size).toBe(3)
  })
})
