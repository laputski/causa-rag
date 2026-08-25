import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import ExperimentsPage from '../pages/ExperimentsPage'
import { renderWithRealm } from './realmTestUtils'

/** A realm that declares its key-metric list.
 *
 *  The shared `stubRealmFetch` returns a realm without one, and then the column
 *  set is derived from the runs themselves, which makes it depend on whether
 *  they have arrived. The test was flaky on that: it passed alone and failed
 *  every other time in a full thirty-two-file run, and raising the timeout did
 *  not help, because speed was not the problem. Declaring the list here tests
 *  exactly what the name claims: whether a legacy metric's column appears for a
 *  run that genuinely carries it. */
function stubRealmWithMetrics(realmId: string, keyMetrics: string[]) {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{
      id: realmId, name: realmId, resources: [], key_metrics: keyMetrics,
      created_at: '2026-01-01',
    }],
  })))
}

const KEY = ['retrieval_recall_at_k', 'answer_similarity', 'context_support', 'correct_refusal']

// The run table's columns named metrics the evaluator does not compute:
// `faithfulness`, `answer_relevancy` and `reference_overlap`, the names of a
// token-overlap evaluator retired in Phase 0.
//
// A dash stood in a hundred percent of rows (checked against a live install: 74
// runs across two realms, not one of the three keys present), and that read as
// "the runs are empty" rather than as "this column names something nobody
// measures".

// `vi.mock` rather than `vi.spyOn(api.…)`: `api` is a single module instance,
// and two test files stubbing the same method on it interfere under parallel
// execution. That showed as a failure every other run, disappearing when the
// file ran alone or under `--no-file-parallelism`. The other twenty-seven files
// already do it this way; this one was the deviation rather than the
// discovery.
const listMock = vi.fn()

vi.mock('../api/client', () => ({
  api: { experiments: { list: (p?: unknown) => listMock(p) } },
}))

const RUN = {
  run_id: 'aaaaaaaa1111', name: 'bge-m3 + rerank', config_hash: 'h',
  started_at: '2026-08-18T11:40:00Z', finished_at: '2026-08-18T11:46:00Z',
  n_questions: 143, dataset_name: 'handbook.v2.full.jsonl', status: 'done' as const,
  realm_id: 'demo',
}

describe('run table columns', () => {
  beforeEach(() => { listMock.mockReset(); localStorage.clear() })

  it('shows the evaluator\'s live keys by default', async () => {
    listMock.mockResolvedValue([{
      ...RUN,
      aggregate_metrics: {
        retrieval_recall_at_k: 0.812, answer_similarity: 0.701,
        context_support: 0.774, correct_refusal: 0.93, retrieval_precision_at_k: 0.318,
      },
    }])
    stubRealmWithMetrics('demo', KEY)
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    await waitFor(() => {
      const headers = [...document.querySelectorAll('th')].map(th => th.textContent ?? '')
      expect(headers.some(h => h.includes('Retrieval Recall@K'))).toBe(true)
      expect(headers.some(h => h.includes('Answer Similarity'))).toBe(true)
    })
    // Scoped to the table row rather than the whole page: the same number now
    // also appears in the band above it ("best recall").
    expect(within(document.querySelector('.run-table') as HTMLElement).getByText('0.812')).toBeInTheDocument()
  })

  it('dead names get no column when no run carries them', async () => {
    listMock.mockResolvedValue([{ ...RUN, aggregate_metrics: { retrieval_recall_at_k: 0.5 } }])
    stubRealmWithMetrics('demo', KEY)
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    await waitFor(() => {
      const headers = [...document.querySelectorAll('th')].map(th => th.textContent ?? '')
      expect(headers.some(h => h.includes('Retrieval Recall@K'))).toBe(true)
      // An empty column asserts that the quantity exists and equals a dash.
      expect(headers.some(h => h.includes('Faithfulness'))).toBe(false)
      expect(headers.some(h => h.includes('Reference Overlap'))).toBe(false)
    })
  })

  it('a legacy metric is shown, labelled, when a run genuinely carries it', async () => {
    listMock.mockResolvedValue([{
      ...RUN,
      aggregate_metrics: { retrieval_recall_at_k: 0.5, faithfulness: 0.04 },
    }])
    stubRealmWithMetrics('demo', KEY)
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    // Old runs exist in the database, and hiding their numbers would lose the
    // history; they need labelling so nobody compares them against the new
    // ones. The whole column header is checked rather than `findByText`: the
    // label is split across nested nodes ("Faithfulness (legacy)" plus a
    // separate "legacy" badge), and a text matcher over that markup matches
    // intermittently. The column itself was present throughout — verified by
    // dumping every `th` at the moment of failure. This asks exactly what the
    // test claims: is the legacy column there, and is it labelled.
    await waitFor(() => {
      const headers = [...document.querySelectorAll('th')].map(th => th.textContent ?? '')
      expect(headers.some(h => h.includes('Faithfulness'))).toBe(true)
      expect(headers.some(h => h.includes('legacy'))).toBe(true)
    })
  })

  it('a running run shows no metrics at all', async () => {
    listMock.mockResolvedValue([{
      ...RUN, status: 'running' as const, aggregate_metrics: {},
      progress_processed: 96, progress_total: 143,
    }])
    stubRealmWithMetrics('demo', KEY)
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    await waitFor(() => expect(screen.getByText(/96\/143/)).toBeInTheDocument())
  })
})
