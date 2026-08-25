import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import ComparisonPage from '../pages/ComparisonPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { CompareResult, ExperimentItem } from '../api/client'

// per-question paired diff rendered alongside the
// existing aggregate metric_deltas. See core/eval/regression.py:paired_diff
// for the backend side.

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

const EXPERIMENTS: ExperimentItem[] = [
  { run_id: 'run-a', name: 'A', config_hash: 'h1', aggregate_metrics: {}, started_at: '2026-01-01T00:00:00Z', finished_at: '', n_questions: 2, dataset_name: 'handbook.v0.fast.jsonl' },
  { run_id: 'run-b', name: 'B', config_hash: 'h2', aggregate_metrics: {}, started_at: '2026-01-02T00:00:00Z', finished_at: '', n_questions: 2, dataset_name: 'handbook.v0.fast.jsonl' },
]

const COMPARE_RESULT: CompareResult = {
  config_diff: {},
  metric_deltas: [],
  summary: 'no changes',
  paired_diff: {
    fixed: ['q1'], flips: ['q2'], unchanged: [], metric_deltas: {},
    questions: {
      q1: { question: 'Does article 5 apply?', funnel_before: 'retrieval', funnel_after: 'ok' },
      q2: { question: 'Is the equipment rule still in force?', funnel_before: 'ok', funnel_after: 'generation' },
    },
  },
}

function renderComparisonPage() {
  return renderWithRealm(<ComparisonPage />, '/compare?a=run-a&b=run-b', 'demo')
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  listMock.mockResolvedValue(EXPERIMENTS)
  compareMock.mockResolvedValue(COMPARE_RESULT)
})

describe('ComparisonPage — paired diff', () => {
  it('shows question text and funnel transition, not just a bare id', async () => {
    renderComparisonPage()
    await waitFor(() => expect(compareMock).toHaveBeenCalledWith(['run-a', 'run-b']))

    // The questions are split across chips rather than shown as two lists side
    // by side: they get read one kind at a time. The broken ones open first,
    // because that is where reading starts.
    await waitFor(() => expect(screen.getByText('Is the equipment rule still in force?')).toBeInTheDocument())
    expect(screen.getByText(/q2/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Fixed/ }))
    expect(screen.getByText('Does article 5 apply?')).toBeInTheDocument()
    expect(screen.getByText(/q1/)).toBeInTheDocument()
  })

  it('falls back to the bare id when a question has no lookup entry', async () => {
    compareMock.mockResolvedValue({
      config_diff: {}, metric_deltas: [], summary: '',
      paired_diff: { fixed: ['qmissing'], flips: [], unchanged: [], metric_deltas: {}, questions: {} },
    })
    renderComparisonPage()
    await waitFor(() => expect(screen.getAllByText(/qmissing/).length).toBeGreaterThan(0))
  })

  it('shows a warning banner when there are flips', async () => {
    renderComparisonPage()
    await waitFor(() => expect(screen.getByText(/Is the equipment rule still in force/)).toBeInTheDocument())
    expect(screen.getByText(/used to pass diagnosis/)).toBeInTheDocument()
  })

  it('shows a noise-filtered note when resampling reclassified some flips, without affecting the flip count', async () => {
    compareMock.mockResolvedValue({
      config_diff: {}, metric_deltas: [], summary: '',
      paired_diff: {
        fixed: [], flips: ['q2'], unchanged: ['q3'], metric_deltas: {},
        questions: { q2: { question: 'Is the equipment rule still in force?', funnel_before: 'ok', funnel_after: 'generation' } },
        resample_attempted: true, noise_filtered: ['q4', 'q5'],
      },
    })
    renderComparisonPage()
    await waitFor(() => expect(screen.getByText(/Is the equipment rule still in force/)).toBeInTheDocument())

    expect(screen.getByText(/2 flip/)).toBeInTheDocument()
    // The broken count does not shift on reclassification: it appears in the
    // band and on the chip, and both show the same single question.
    expect(screen.getByRole('button', { name: /Broken/ }).textContent).toContain('1')
  })

  it('shows no noise-filtered note when nothing was reclassified', async () => {
    renderComparisonPage()
    await waitFor(() => expect(screen.getByText(/Is the equipment rule still in force/)).toBeInTheDocument())
    expect(screen.queryByText(/did not reproduce on resample/)).not.toBeInTheDocument()
  })

  it('shows the empty state and no warning when nothing changed', async () => {
    compareMock.mockResolvedValue({
      config_diff: {}, metric_deltas: [], summary: '',
      paired_diff: { fixed: [], flips: [], unchanged: ['q1', 'q2'], metric_deltas: {}, questions: {} },
    })
    renderComparisonPage()
    await waitFor(() => expect(compareMock).toHaveBeenCalled())

    await waitFor(() => expect(screen.getByText(/No question moved/)).toBeInTheDocument())
    expect(screen.queryByText(/used to pass diagnosis/)).not.toBeInTheDocument()
  })
})
