import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import JudgmentsPage from '../pages/JudgmentsPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { Judgment } from '../api/client'

// Replaces PinsPage.test.tsx. The page records what a reviewer
// observed; it never applies anything, which is why there is no threshold
// field and no "validate blast radius" step to test here.

const listMock = vi.fn()
const createMock = vi.fn()
const promoteMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    judgments: {
      list: (realmId: string, corpusId: string) => listMock(realmId, corpusId),
      create: (body: unknown) => createMock(body),
      update: vi.fn(),
      delete: vi.fn(),
      toGoldenQuestion: (id: string, realmId: string, corpusId: string, body: unknown) =>
        promoteMock(id, realmId, corpusId, body),
    },
    corpus: {
      collections: () => Promise.resolve([{ corpus_id: 'handbook_01' }]),
      chunks: () => Promise.resolve({ items: [] }),
    },
  },
}))

const JUDGMENT: Judgment = {
  id: 'j1', realm_id: 'demo', corpus_id: 'handbook_01',
  question: 'What is the appeal deadline?',
  relevant: [{ chunk_id: 'c1', ref_id: 'S/1', doc_id: 'd1', structural_path: 'Ch 1', text: 'text' }],
  irrelevant: [{ chunk_id: 'c9', ref_id: '', doc_id: '', structural_path: '', text: '' }],
  note: '', author: '', created_at: '2026-07-29T00:00:00Z', status: 'active',
  source_run_id: '', source_feedback_id: '', source_question_id: '',
  preference_pairs: 1, can_become_test: true,
}

function renderPage(path = '/data/judgments') {
  return renderWithRealm(<JudgmentsPage />, path, 'demo')
}

describe('JudgmentsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    stubRealmFetch('demo')
    listMock.mockResolvedValue([JUDGMENT])
    createMock.mockResolvedValue(JUDGMENT)
  })

  it('lists judgments for the selected corpus and shows what each is worth', async () => {
    renderPage('/data/judgments?corpus_id=handbook_01')
    await waitFor(() => expect(document.querySelector('.pick-row-title')).toHaveTextContent('What is the appeal deadline?'))
  })

  it('opens straight into a pre-filled form when deep-linked from feedback triage', async () => {
    // The page it replaces shipped with this broken: `creating` started
    // false, so the prefill was computed and then never shown.
    renderPage('/data/judgments?corpus_id=handbook_01&question=' + encodeURIComponent('What is the deadline?'))
    await waitFor(() => {
      expect(screen.getByDisplayValue('What is the deadline?')).toBeInTheDocument()
    })
  })

  it('scopes a created judgment to the active Realm', async () => {
    // The predecessor page shipped a defect where every record was written
    // with a null realm_id and became invisible to the Realm that made it.
    renderPage('/data/judgments?corpus_id=handbook_01&question=' + encodeURIComponent('What is the deadline?'))
    await waitFor(() => expect(screen.getByDisplayValue('What is the deadline?')).toBeInTheDocument())

    // An exact match, because the irrelevant-chunks label contains the
    // relevant-chunks one as a substring and a loose regex silently matches
    // both fields.
    const relevant = screen.getByLabelText('Relevant chunks')
    fireEvent.change(relevant, { target: { value: 'c1' } })
    fireEvent.click(screen.getByRole('button', { name: /Save judgment/i }))

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1))
    const body = createMock.mock.calls[0][0] as Record<string, unknown>
    expect(body.realm_id).toBe('demo')
    expect(body.relevant).toEqual(['c1'])
  })

  it('refuses to submit a judgment that rules on no chunk at all', async () => {
    renderPage('/data/judgments?corpus_id=handbook_01&question=' + encodeURIComponent('What is the deadline?'))
    await waitFor(() => expect(screen.getByDisplayValue('What is the deadline?')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Save judgment/i })).toBeDisabled()
  })
})
