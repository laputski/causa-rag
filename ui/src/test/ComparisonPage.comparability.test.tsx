import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import ComparisonPage from '../pages/ComparisonPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import type { Comparability, CompareResult, ExperimentItem } from '../api/client'

// Whether the pair measures the same thing at all, said before the comparison
// (the pre-flight check under the selectors) and again above the report. See
// core/experiment/compare.py:check_comparability for the rules themselves.

const listMock = vi.fn()
const compareMock = vi.fn()
const preflightMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      list: (params: unknown) => listMock(params),
      compare: (ids: string[]) => compareMock(ids),
      comparePreflight: (a: string, b: string) => preflightMock(a, b),
    },
  },
}))

const EXPERIMENTS: ExperimentItem[] = [
  { run_id: 'run-a', name: 'A', config_hash: 'h1', aggregate_metrics: {}, started_at: '2026-01-01T00:00:00Z', finished_at: '', n_questions: 48, dataset_name: 'handbook.v0.fast.jsonl' },
  { run_id: 'run-b', name: 'B', config_hash: 'h2', aggregate_metrics: {}, started_at: '2026-01-02T00:00:00Z', finished_at: '', n_questions: 143, dataset_name: 'handbook.v1.full.jsonl' },
]

const FACTS = {
  before: { run_id: 'run-a', dataset_name: 'handbook.v0.fast.jsonl', n_questions: 48, realm_id: '', corpus_id: 'default', stopped: false },
  after: { run_id: 'run-b', dataset_name: 'handbook.v1.full.jsonl', n_questions: 143, realm_id: '', corpus_id: 'default', stopped: false },
}

const INCOMPATIBLE: Comparability = {
  comparable: false, matched: 0, only_in_before: 48, only_in_after: 143, ...FACTS,
  warnings: [{
    id: 'different_dataset', severity: 'error',
    title: 'The runs were made on different datasets',
    detail: 'server fallback text',
    params: {
      dataset_before: 'handbook.v0.fast.jsonl', count_before: 48,
      dataset_after: 'handbook.v1.full.jsonl', count_after: 143,
    },
  }],
}

const CLEAN: Comparability = {
  comparable: true, matched: 2, only_in_before: 0, only_in_after: 0, ...FACTS, warnings: [],
}

const EMPTY_DIFF = { fixed: [], flips: [], unchanged: [], metric_deltas: {}, questions: {} }

function result(over: Partial<CompareResult> = {}): CompareResult {
  return { config_diff: {}, metric_deltas: [], summary: '', paired_diff: EMPTY_DIFF, ...over }
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  listMock.mockResolvedValue(EXPERIMENTS)
  preflightMock.mockResolvedValue(CLEAN)
  compareMock.mockResolvedValue(result())
})

const render = () => renderWithRealm(<ComparisonPage />, '/compare?a=run-a&b=run-b', 'demo')

/** The pre-flight state: a pair chosen in the picker, nothing compared yet. */
async function pickPair() {
  renderWithRealm(<ComparisonPage />, '/compare', 'demo')
  const a = await screen.findByLabelText('Run A (before)') as HTMLSelectElement
  // The run options arrive from a separate query, and the select renders
  // before it lands, holding only its placeholder. Firing a change to a value
  // that is not an option yet is dropped by a controlled select in silence,
  // so the wait is on the options and never on the element alone. Both
  // selects are fed by the same query, so one wait covers the pair.
  await waitFor(() => expect([...a.options].map(o => o.value)).toContain('run-a'))
  fireEvent.change(a, { target: { value: 'run-a' } })
  fireEvent.change(screen.getByLabelText('Run B (after)'), { target: { value: 'run-b' } })
}

describe('ComparisonPage comparability', () => {
  it('names what is incomparable from the pre-flight check, before the report', async () => {
    preflightMock.mockResolvedValue(INCOMPATIBLE)
    await pickPair()

    await waitFor(() => expect(preflightMock).toHaveBeenCalledWith('run-a', 'run-b'))
    expect(compareMock).not.toHaveBeenCalled()
    // The sentence is rendered from the id and the parameters, so the server's
    // English fallback never reaches the screen.
    await waitFor(() =>
      expect(screen.getAllByText(/were made on different datasets/).length).toBeGreaterThan(0))
    expect(screen.queryByText('server fallback text')).not.toBeInTheDocument()
    expect(screen.getAllByText(/handbook\.v1\.full\.jsonl/).length).toBeGreaterThan(0)
  })

  it('falls back to the server sentence for a warning id this build has no text for', async () => {
    // A gateway newer than the interface can send a rule that shipped after
    // this build. Showing its English text keeps the finding readable.
    preflightMock.mockResolvedValue({
      ...INCOMPATIBLE,
      warnings: [{
        id: 'a_rule_from_the_future', severity: 'warn',
        title: 'Something this build has never heard of',
        detail: 'and the reason for it',
      }],
    })
    await pickPair()

    await waitFor(() =>
      expect(screen.getByText('Something this build has never heard of')).toBeInTheDocument())
    expect(screen.getByText('and the reason for it')).toBeInTheDocument()
  })

  it('leaves the compare button usable on an incomparable pair', async () => {
    preflightMock.mockResolvedValue(INCOMPATIBLE)
    await pickPair()

    await waitFor(() => expect(preflightMock).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: 'Compare' })).toBeEnabled()

  })

  it('does not repeat in the picker what the report above already says', async () => {
    // Opening "pick another pair" over a report of that same pair used to
    // show the identical findings twice on one screen.
    preflightMock.mockResolvedValue(INCOMPATIBLE)
    compareMock.mockResolvedValue(result({ compatibility: INCOMPATIBLE }))
    render()

    await waitFor(() => expect(preflightMock).toHaveBeenCalled())
    await waitFor(() =>
      expect(screen.getAllByText(/were made on different datasets/)).toHaveLength(1))
  })

  it('says nothing could be paired, never that nothing changed', async () => {
    // The two read identically in the counts above, and only one of them
    // means the configuration change had no effect.
    compareMock.mockResolvedValue(result({ compatibility: INCOMPATIBLE }))
    render()

    await waitFor(() => expect(screen.getByText(/nothing to pair/i)).toBeInTheDocument())
    expect(screen.queryByText(/No question moved between/)).not.toBeInTheDocument()
  })

  it('states both datasets in the header when the pair disagrees about them', async () => {
    compareMock.mockResolvedValue(result({ compatibility: INCOMPATIBLE }))
    render()

    // The middle dot is the header's own separator. The warning below states
    // the same two datasets in a sentence, so matching on the name alone
    // would not tell the two places apart.
    await waitFor(() =>
      expect(screen.getByText('A: handbook.v0.fast.jsonl · 48 questions')).toBeInTheDocument())
    expect(screen.getByText('B: handbook.v1.full.jsonl · 143 questions')).toBeInTheDocument()
  })

  it('reports a metric measured in one run only as unmeasured, never as a rise', async () => {
    compareMock.mockResolvedValue(result({
      compatibility: CLEAN,
      metric_deltas: [
        { metric: 'answer_similarity', before: null, after: 0.48, delta: null, delta_pct: null },
      ],
    }))
    render()

    await waitFor(() => expect(screen.getByText('Not measured')).toBeInTheDocument())
    expect(screen.queryByText(/Improved/)).not.toBeInTheDocument()
    // The side named is the one holding the value. Found live: the two were
    // the wrong way round, and the row pointed at the empty column.
    expect(screen.getByText(/Measured in run B only/)).toBeInTheDocument()
  })

  it('counts the questions that could not be paired', async () => {
    compareMock.mockResolvedValue(result({
      compatibility: CLEAN,
      paired_diff: { ...EMPTY_DIFF, unchanged: ['q1'], only_in_before: ['x'], only_in_after: ['y', 'z'] },
    }))
    render()

    await waitFor(() => expect(screen.getByText(/only in A 1/)).toBeInTheDocument())
    expect(screen.getByText(/only in B 2/)).toBeInTheDocument()
  })
})
