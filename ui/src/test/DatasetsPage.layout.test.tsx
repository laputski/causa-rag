import { describe, it, expect, vi, beforeEach } from 'vitest'
import { waitFor, fireEvent, screen } from '@testing-library/react'
import DatasetsPage from '../pages/DatasetsPage'
import { renderWithRealm } from './realmTestUtils'

// A set of a hundred and forty-three questions rendered as a ribbon of cards:
// neighbouring rows could not be compared, and how many lacked source
// references was invisible.

const Q = (id: string, over: Record<string, unknown> = {}) => ({
  id, question: `Question ${id}`, reference_answer: `Reference ${id}`,
  question_type: '', article_refs: ['ST/1'], provenance: { origin: 'manual' },
  ...over,
})

const DETAIL = {
  filename: 'handbook.v2.jsonl', name: 'handbook', version: 'v2', speed: 'full', count: 3,
  questions: [
    Q('a'),
    Q('b', { article_refs: [] }),
    Q('c', { reference_answer: '' }),
  ],
}

vi.mock('../api/client', async () => {
  const actual = await vi.importActual<typeof import('../api/client')>('../api/client')
  return {
  ...actual,
  api: {
    datasets: {
      list: () => Promise.resolve([{ filename: 'handbook.v2.jsonl', name: 'handbook', version: 'v2', speed: 'full', count: 3 }]),
      get: () => Promise.resolve(DETAIL),
    },
    corpus: { collections: () => Promise.resolve([]) },
    generationPresets: { list: () => Promise.resolve([]) },
    models: () => Promise.resolve([]),
  },
  }
})

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: [], created_at: '2026-01-01' }],
  })))
})

describe('the control-questions layout', () => {
  it('opens the largest set on its own and shows the questions as a table', async () => {
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    await waitFor(() => expect(document.querySelector('.q-table')).toBeTruthy())
    // Three questions, three rows, and no cards left.
    expect(document.querySelectorAll('.q-table tbody tr')).toHaveLength(3)
    expect(document.querySelectorAll('.q-table .card')).toHaveLength(0)
  })

  it('marks a question with no source references and filters for them by chip', async () => {
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    await waitFor(() => expect(document.querySelector('.q-table')).toBeTruthy())
    // Exactly one question in the set has no references, and it is marked rather
    // than merely empty.
    expect(document.querySelectorAll('.q-noref')).toHaveLength(1)

    const chip = [...document.querySelectorAll('.chip')]
      .find(c => c.textContent?.startsWith('No references')) as HTMLElement
    expect(chip.textContent).toContain('1')
    fireEvent.click(chip)
    expect(document.querySelectorAll('.q-table tbody tr')).toHaveLength(1)
  })

  it('search filters rows by the question text', async () => {
    renderWithRealm(<DatasetsPage />, '/data/qa', 'demo')

    await waitFor(() => expect(document.querySelector('.q-table')).toBeTruthy())
    fireEvent.change(screen.getByLabelText('Search question and reference'), { target: { value: 'Question c' } })
    expect(document.querySelectorAll('.q-table tbody tr')).toHaveLength(1)
  })
})
