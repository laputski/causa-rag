import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, within } from '@testing-library/react'
import RunPage from '../pages/RunPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { ExperimentDetail } from '../api/client'

// The three blocks the run page gained, where a reader meets them: the
// stages and the statements inside a question's row, and what ran the run on
// the Overview tab. Each block has its own tests; this file pins that the
// page hands them the run's data and not something of its own.

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

const QUESTION = {
  question_id: 'q1', question: 'Which instruments need level 4?', reference_answer: 'r', generated_answer: 'g',
  metrics: { assertion_coverage: 0.5 }, scope: 'global',
  assertion_verdicts: [{ statement: 'The XR-310 needs level 4.', stated: true }, { statement: 'The KL-910 needs level 4.', stated: false }],
  stages: {
    stages: [{ stage: 'retrieval', size: 8, rank: null, same_as: null }, { stage: 'context', size: 5, rank: null, same_as: null }],
    lost_at: null, found: null, needed: [],
  },
}

function detail(overrides: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    run_id: 'run1', status: 'done', config: { name: 'run', top_k: 5 }, aggregate_metrics: { assertion_coverage: 0.5 },
    question_results: [QUESTION],
    served_by: { generator: { provider: 'ollama', address: 'http://localhost:11434', model: 'qwen3:8b', remote: false }, stores: {} },
    ...overrides,
  } as ExperimentDetail
}

describe('RunPage — stages, statements and what ran the run', () => {
  beforeEach(() => { vi.clearAllMocks(); stubRealmFetch('acme') })

  it('shows in a question row what the answer had to contain and where its sources went', async () => {
    getMock.mockResolvedValue(detail())
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const statements = await screen.findByTestId('answer-statements')
    expect(within(statements).getByText('1 of 2 present')).toBeInTheDocument()
    expect(within(statements).getByText(/Retrieval metrics are not computed/)).toBeInTheDocument()
    const stages = screen.getByTestId('retrieval-stages')
    expect(within(stages).getByText('8 fragments')).toBeInTheDocument()
  })

  it('marks a whole-corpus question in the list without opening it', async () => {
    getMock.mockResolvedValue(detail())
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const summary = (await screen.findByText(QUESTION.question)).closest('summary') as HTMLElement
    expect(within(summary).getByText('whole corpus')).toBeInTheDocument()
  })

  it('names the completeness metric in words', async () => {
    getMock.mockResolvedValue(detail())
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    expect((await screen.findAllByText('Answer completeness')).length).toBeGreaterThan(0)
    expect(screen.queryByText('assertion_coverage')).toBeNull()
  })

  it('says on the Overview tab what ran the run', async () => {
    getMock.mockResolvedValue(detail())
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const block = await screen.findByTestId('run-served-by')
    expect(within(block).getByText('qwen3:8b')).toBeInTheDocument()
    expect(within(block).getByText('local')).toBeInTheDocument()
  })

  it('tells a question row that the run lost its intermediate lists', async () => {
    getMock.mockResolvedValue(detail({ debug_cleared_at: '2026-09-30T10:00:00Z' }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    const stages = await screen.findByTestId('retrieval-stages')
    expect(within(stages).getByText(/intermediate lists were removed on 30\.09\.2026/)).toBeInTheDocument()
  })

  it('adds nothing to a question that has neither stages nor statements', async () => {
    const plain = { question_id: 'q2', question: 'Plain?', reference_answer: 'r', generated_answer: 'g', metrics: {} }
    getMock.mockResolvedValue(detail({ question_results: [plain] }))
    renderWithRealm(<RunPage />, '/runs/run1', 'acme')
    await screen.findByText('Plain?')
    expect(screen.queryByTestId('retrieval-stages')).toBeNull()
    expect(screen.queryByTestId('answer-statements')).toBeNull()
  })
})
