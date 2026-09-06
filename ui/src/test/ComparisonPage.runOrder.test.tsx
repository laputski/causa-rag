import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, within } from '@testing-library/react'
import ComparisonPage from '../pages/ComparisonPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { ExperimentItem } from '../api/client'

// The server sorts this list by run_id, which is a random identifier, so the
// two pickers arrived in an order carrying no meaning. The run somebody wants
// to compare is almost always one of the last few.

const listMock = vi.fn()
const compareMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      list: (params: unknown) => listMock(params),
      compare: (ids: string[]) => compareMock(ids),
    },
  },
}))

// Deliberately not in date order, and not in reverse date order either: an
// order the fix has to impose, not one it could preserve.
const EXPERIMENTS: ExperimentItem[] = [
  { run_id: 'r-mid', name: 'Middle', config_hash: 'h', aggregate_metrics: {}, started_at: '2026-03-02T10:00:00Z', finished_at: '', n_questions: 4, dataset_name: 'd.jsonl' },
  { run_id: 'r-old', name: 'Oldest', config_hash: 'h', aggregate_metrics: {}, started_at: '2026-01-05T10:00:00Z', finished_at: '', n_questions: 4, dataset_name: 'd.jsonl' },
  { run_id: 'r-new', name: 'Newest', config_hash: 'h', aggregate_metrics: {}, started_at: '2026-09-01T10:00:00Z', finished_at: '', n_questions: 4, dataset_name: 'd.jsonl' },
]

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  listMock.mockResolvedValue(EXPERIMENTS)
  compareMock.mockResolvedValue(null)
})

describe('Comparison: the order runs are offered in', () => {
  it('offers the newest run first in both pickers', async () => {
    renderWithRealm(<ComparisonPage />, '/compare', 'demo')

    // Both pickers, by role: the labels come from the translation file and
    // this is about the order, not about the wording.
    const selects = await screen.findAllByRole('combobox')
    expect(selects).toHaveLength(2)
    for (const select of selects) {
      const values = within(select)
        .getAllByRole('option')
        .map(option => (option as HTMLOptionElement).value)
        .filter(Boolean)
      expect(values).toEqual(['r-new', 'r-mid', 'r-old'])
    }
  })

  it('puts a run with no start date last instead of first', async () => {
    // An in-flight run the server prepended, or a record written before the
    // field existed: sorting must not float it above every finished run.
    listMock.mockResolvedValue([
      { ...EXPERIMENTS[1] },
      { run_id: 'r-none', name: 'Undated', config_hash: 'h', aggregate_metrics: {}, started_at: '', finished_at: '', n_questions: 4, dataset_name: 'd.jsonl' },
      { ...EXPERIMENTS[2] },
    ])
    renderWithRealm(<ComparisonPage />, '/compare', 'demo')

    const [select] = await screen.findAllByRole('combobox')
    const values = within(select)
      .getAllByRole('option')
      .map(option => (option as HTMLOptionElement).value)
      .filter(Boolean)
    expect(values).toEqual(['r-new', 'r-old', 'r-none'])
  })
})
