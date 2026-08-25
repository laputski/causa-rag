import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import RunPage from '../pages/RunPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { ExperimentDetail } from '../api/client'

// The cause and the lever reach the reader.
//
// The point of the work is that a funnel verdict of "retrieval" names a layer
// and sends an engineer to tune a ranker even when the expected source is not
// in the index at all. These tests pin that the cause is shown beside the
// verdict, that the lever is named, and that the aggregate ranks causes by
// reach rather than leaving the reader to do it.

const getMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      get: (runId: string, realmId?: string | null) => getMock(runId, realmId),
      listBaselines: () => Promise.resolve([]),
    },
    feedback: { getRun: () => Promise.resolve({}) },
    datasets: { list: () => Promise.resolve([]) },
  },
}))

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return { ...actual, useParams: () => ({ runId: 'run1' }) }
})

function detail(overrides: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    run_id: 'run1',
    status: 'done',
    config: { name: 'run', top_k: 5 },
    aggregate_metrics: {},
    question_results: [],
    ...overrides,
  } as ExperimentDetail
}

const FAILED_QUESTION = {
  question_id: 'q1',
  question: 'What is the appeal deadline?',
  reference_answer: 'r',
  generated_answer: 'g',
  metrics: { retrieval_recall_at_k: 0 },
  funnel: { layer: 'retrieval', detail: 'the required article never reached the context' },
  root_cause: {
    cause: 'data_missing' as const,
    detail: 'The expected source is not in the index (RC/1).',
    lever: 'ingest' as const,
    evidence: { absent_refs: ['RC/1'] },
  },
}

describe('RunPage — root cause and lever', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    stubRealmFetch('acme')
  })

  it('shows the cause beside the funnel verdict rather than only the layer', async () => {
    getMock.mockResolvedValue(detail({ question_results: [FAILED_QUESTION] }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    // getAllByText, because the badge text also matches the <summary> that
    // contains it — the assertion is about the badge existing, not about it
    // being the only node whose combined text mentions the cause.
    await waitFor(() => expect(screen.getAllByText(/not in the index/i).length).toBeGreaterThan(0))
  })

  it('names the lever, not just the cause', async () => {
    // A cause without a lever still leaves "so what do I do" unanswered,
    // which is the question this work exists to close.
    getMock.mockResolvedValue(detail({ question_results: [FAILED_QUESTION] }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const row = await screen.findByText(/What is the appeal deadline/i)
    fireEvent.click(row)
    await waitFor(() =>
      expect(screen.getByText(/Check document ingestion/i)).toBeInTheDocument(),
    )
  })

  it('does not offer the on-demand widened re-query when a cause is already known', async () => {
    // The same answer is computed for every failed question now, so offering
    // the button again would invite a second retrieval for nothing.
    getMock.mockResolvedValue(detail({ question_results: [FAILED_QUESTION] }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const row = await screen.findByText(/What is the appeal deadline/i)
    fireEvent.click(row)
    await waitFor(() => expect(screen.getByText(/Check document ingestion/i)).toBeInTheDocument())
    expect(screen.queryByText(/how close/i)).not.toBeInTheDocument()
  })

  it('ranks the work by how many questions each task would close', async () => {
    // Phase 2 — the panel replaced a per-cause tally, which ranked kinds of
    // work but left the reader to find out which document each concerned.
    getMock.mockResolvedValue(detail({
      question_results: [FAILED_QUESTION],
      fix_tasks: [
        { cause: 'data_missing', lever: 'ingest', entity: 'BIG', questions: 4, question_ids: ['q1', 'q2', 'q3', 'q4'] },
        { cause: 'ranking', lever: 'ranking', entity: 'SMALL', questions: 1, question_ids: ['q5'] },
      ],
      clusterable_share: 0.8,
    }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    fireEvent.click(await screen.findByRole('tab', { name: /diagnostic/i }))
    await waitFor(() => expect(screen.getByText(/What to fix first/i)).toBeInTheDocument())

    // Most-reaching first, and each task names the document someone would
    // actually go and fix.
    const entities = screen.getAllByText(/^(BIG|SMALL)$/).map(n => n.textContent)
    expect(entities).toEqual(['BIG', 'SMALL'])
  })

  it('states how much of the failure actually generalises', async () => {
    // The list looks equally actionable whether failures cluster or every
    // one is its own isolated case, so the share is stated beside it.
    getMock.mockResolvedValue(detail({
      question_results: [FAILED_QUESTION],
      fix_tasks: [
        { cause: 'data_missing', lever: 'ingest', entity: 'BIG', questions: 2, question_ids: ['q1', 'q2'] },
      ],
      clusterable_share: 0.5,
    }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    fireEvent.click(await screen.findByRole('tab', { name: /diagnostic/i }))
    await waitFor(() => expect(screen.getByText(/50%/)).toBeInTheDocument())
  })

  it('says so plainly when a run has no retrieval failures at all', async () => {
    getMock.mockResolvedValue(detail({ fix_tasks: [] }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    fireEvent.click(await screen.findByRole('tab', { name: /diagnostic/i }))
    await waitFor(() => expect(screen.getByText(/No retrieval failures/i)).toBeInTheDocument())
  })
})
