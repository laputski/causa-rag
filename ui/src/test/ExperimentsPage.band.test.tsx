import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import ExperimentsPage from '../pages/ExperimentsPage'
import { renderWithRealm } from './realmTestUtils'

// The page used to open straight into the list, so "how many are there and
// which is best" had to be extracted by sorting a column.

const listMock = vi.fn()
vi.mock('../api/client', () => ({
  api: { experiments: { list: (p?: unknown) => listMock(p) } },
}))

function realmWithMetrics(keyMetrics: string[]) {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => [{ id: 'demo', name: 'Demo', resources: [], key_metrics: keyMetrics, created_at: '2026-01-01' }],
  })))
}

const RUN = {
  name: 'run', config_hash: 'h', finished_at: '', n_questions: 10,
  dataset_name: 'handbook.v2.full.jsonl', status: 'done' as const, realm_id: 'demo',
}

/** The stat band, once it holds `text`.
 *
 * The band is rendered before the run list arrives, so waiting for the
 * element and then reading it asserts against the empty first paint. Waiting
 * for the content is the same wait done one step later. */
async function bandShowing(text: string) {
  await waitFor(() => {
    const band = document.querySelector('.stat-band')
    expect(band).toBeTruthy()
    expect(within(band as HTMLElement).queryByText(text)).toBeInTheDocument()
  })
  return within(document.querySelector('.stat-band') as HTMLElement)
}

describe('the number band on the runs page', () => {
  beforeEach(() => { listMock.mockReset(); localStorage.clear() })

  it('shows four numbers', async () => {
    listMock.mockResolvedValue([
      { ...RUN, run_id: 'aaaa1111', started_at: '2026-08-18T11:40:00Z', aggregate_metrics: { retrieval_recall_at_k: 0.81 }, is_baseline: true },
      { ...RUN, run_id: 'bbbb2222', started_at: '2026-08-17T09:00:00Z', aggregate_metrics: { retrieval_recall_at_k: 0.58 } },
    ])
    realmWithMetrics(['retrieval_recall_at_k'])
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    await waitFor(() => {
      expect(document.querySelectorAll('.stat-band .stat-cell')).toHaveLength(4)
    })
  })

  it('the baseline cell shows the run carrying is_baseline', async () => {
    listMock.mockResolvedValue([
      { ...RUN, run_id: 'aaaa1111', started_at: '2026-08-18T11:40:00Z', aggregate_metrics: {} },
      { ...RUN, run_id: 'ffff9999', started_at: '2026-08-17T09:00:00Z', aggregate_metrics: {}, is_baseline: true },
    ])
    realmWithMetrics(['retrieval_recall_at_k'])
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    // Scoped to the band rather than the whole page: the same id also appears in
    // a table row, and a document-wide search finds both.
    // Not the newest and not the first in the list: the marked one.
    expect((await bandShowing('ffff9999')).getByText('ffff9999')).toBeInTheDocument()
  })

  it("the best value is taken by the realm's first key metric", async () => {
    listMock.mockResolvedValue([
      { ...RUN, run_id: 'aaaa1111', started_at: '2026-08-18T11:40:00Z', aggregate_metrics: { retrieval_recall_at_k: 0.412, answer_similarity: 0.99 } },
      { ...RUN, run_id: 'bbbb2222', started_at: '2026-08-17T09:00:00Z', aggregate_metrics: { retrieval_recall_at_k: 0.833, answer_similarity: 0.10 } },
    ])
    realmWithMetrics(['retrieval_recall_at_k', 'answer_similarity'])
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    // 0.833 on recall rather than 0.99 on similarity: the metric comes from the
    // realm's list rather than being whichever number is largest.
    expect((await bandShowing('0.833')).getByText('0.833')).toBeInTheDocument()
  })

  it('with no baseline the cell says so rather than showing an arbitrary run', async () => {
    listMock.mockResolvedValue([{ ...RUN, run_id: 'aaaa1111', started_at: '2026-08-18T11:40:00Z', aggregate_metrics: {} }])
    realmWithMetrics(['retrieval_recall_at_k'])
    renderWithRealm(<ExperimentsPage />, '/experiments', 'demo')

    expect(await screen.findByText('not set')).toBeInTheDocument()
  })
})
