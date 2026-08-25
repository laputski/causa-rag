import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { RealmProvider } from '../context/RealmContext'
import RunPage from '../pages/RunPage'
import { stubRealmFetch } from './realmTestUtils'
import type { ExperimentDetail } from '../api/client'

// A chunk injected by an active retrieval pin is
// flagged `pinned` on its SourceRefView; RunPage's RetrievalPanel shows a
// small badge for it so the pin's effect stays auditable directly in the
// run's own trace, not just in the pins registry.

vi.mock('../api/client', () => ({
  api: {
    experiments: { get: (id: string) => getExperimentMock(id), setBaseline: vi.fn() },
    feedback: { get: (runId: string) => getFeedbackMock(runId) },
    prompts: { get: vi.fn() },
  },
}))

const getExperimentMock = vi.fn()
const getFeedbackMock = vi.fn()

const RUN: ExperimentDetail = {
  status: 'done',
  config_hash: 'hash1',
  config_name: 'experiment',
  run_id: 'run1',
  aggregate_metrics: {},
  question_results: [
    {
      question_id: 'q1', question: 'What is the procedure?', reference_answer: 'Yes', generated_answer: 'Yes',
      metrics: {},
      source_refs: [
        { chunk_id: 'c1', structural_path: 'Article 1', score: 0.9, chunk_text: 'an ordinary chunk' },
        { chunk_id: 'c2', structural_path: 'Article 42', score: 0.85, chunk_text: 'a pinned chunk', pinned: true },
      ],
    },
  ],
  config: { pipeline_id: 'naive', corpus_id: 'default' },
}

function renderRunPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/experiments/run1?realm=demo']}>
        <RealmProvider>
          <Routes><Route path="/experiments/:runId" element={<RunPage />} /></Routes>
        </RealmProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  getExperimentMock.mockResolvedValue(RUN)
  getFeedbackMock.mockResolvedValue({})
})

describe('RunPage — pinned chunk badge in the retrieval table', () => {
  it('shows a pin badge only on the source_ref flagged pinned=true', async () => {
    renderRunPage()
    const summary = await screen.findByText(/What is the procedure\?/)
    const details = summary.closest('details') as HTMLDetailsElement
    const withinRow = within(details)

    fireEvent.click(withinRow.getByRole('button', { name: /Retrieval/ }))

    await waitFor(() => expect(withinRow.getByText('Article 42')).toBeInTheDocument())
    const pinnedCell = withinRow.getByText('Article 42').closest('td') as HTMLElement
    const ordinaryCell = withinRow.getByText('Article 1').closest('td') as HTMLElement

    expect(within(pinnedCell).getByText('pinned')).toBeInTheDocument()
    expect(within(ordinaryCell).queryByText('pinned')).not.toBeInTheDocument()
  })
})
