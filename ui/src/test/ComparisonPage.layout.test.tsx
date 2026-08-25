import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import ComparisonPage from '../pages/ComparisonPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'
import { classifyDelta, metricLabel } from '../lib/metricMeta'
import type { CompareResult, ExperimentItem } from '../api/client'

// The comparison screen as designed: a band of four numbers rather than two
// warnings in prose, a verdict against each metric, and a "what this means"
// column that did not exist — a number says what changed rather than what it
// means, and every reader drew that conclusion themselves.

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
  { run_id: 'run-a', name: 'A', config_hash: 'h1', aggregate_metrics: {}, started_at: '2026-01-01T00:00:00Z', finished_at: '', n_questions: 4, dataset_name: 'handbook.v0.fast.jsonl' },
  { run_id: 'run-b', name: 'B', config_hash: 'h2', aggregate_metrics: {}, started_at: '2026-01-02T00:00:00Z', finished_at: '', n_questions: 4, dataset_name: 'handbook.v0.fast.jsonl' },
]

// Three metrics, one per verdict: rose, fell, did not move past the noise
// threshold.
const METRIC_DELTAS = [
  { metric: 'retrieval_recall_at_k', before: 0.40, after: 0.80, delta: 0.40, delta_pct: 100 },
  { metric: 'answer_similarity', before: 0.90, after: 0.40, delta: -0.50, delta_pct: -55.6 },
  { metric: 'context_support', before: 0.70, after: 0.705, delta: 0.005, delta_pct: 0.7 },
]

const RESULT: CompareResult = {
  config_diff: {},
  metric_deltas: METRIC_DELTAS,
  summary: 'text summary',
  paired_diff: {
    fixed: ['q1', 'q2'], flips: ['q3'], unchanged: ['q4', 'q5', 'q6'],
    noise_filtered: ['q7'],
    metric_deltas: {},
    questions: {
      q1: { question: 'The first', funnel_before: 'retrieval', funnel_after: 'ok' },
      q2: { question: 'The second', funnel_before: 'rerank', funnel_after: 'ok' },
      q3: { question: 'The third', funnel_before: 'ok', funnel_after: 'generation' },
    },
  },
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('demo')
  listMock.mockResolvedValue(EXPERIMENTS)
  compareMock.mockResolvedValue(RESULT)
})

function render() {
  return renderWithRealm(<ComparisonPage />, '/compare?a=run-a&b=run-b', 'demo')
}

describe('Comparison: the designed layout', () => {
  it('a band of four cells, carrying the same numbers as the paired diff', async () => {
    const { container } = render()
    await screen.findByText('text summary')

    const cells = container.querySelectorAll('.stat-band .stat-cell')
    expect(cells.length).toBe(4)
    // Order and values: fixed, broken, unchanged, filtered out as noise.
    expect([...cells].map(c => c.querySelector('.metric-val')?.textContent))
      .toEqual(['2', '1', '3', '1'])
  })

  it('every metric carries a verdict, the same one classifyDelta returns', async () => {
    const { container } = render()
    await screen.findByText('text summary')

    const table = container.querySelectorAll('table.cmp-table')[0]
    const rows = [...table.querySelectorAll('tbody tr')]
    expect(rows.length).toBe(METRIC_DELTAS.length)

    for (const d of METRIC_DELTAS) {
      const row = rows.find(r => r.querySelector('td')?.textContent === metricLabel(d.metric))
      expect(row, `no row for ${d.metric}`).toBeTruthy()
      // The verdict lives in the fifth column's badge. Checked against the same
      // function that computes it: the test holds the link rather than
      // restating the thresholds.
      const verdict = classifyDelta(d.before, d.after, d.metric)
      const badge = row!.querySelectorAll('td')[4].querySelector('.badge')
      expect(badge, `no verdict on ${d.metric}`).toBeTruthy()
      const expectedClass = { improved: 'badge-success', regressed: 'badge-danger', noise: 'badge-info' }[verdict]
      expect(badge!.className, `${d.metric}: verdict ${verdict}`).toContain(expectedClass)
    }
  })

  it('the "what this means" column is filled, above all for the metric that fell', async () => {
    const { container } = render()
    await screen.findByText('text summary')

    const rows = [...container.querySelectorAll('table.cmp-table')[0].querySelectorAll('tbody tr')]
    for (const row of rows) {
      const means = row.querySelector('.cmp-means')
      expect(means?.textContent?.trim().length, 'empty "what this means" cell').toBeGreaterThan(0)
    }

    const regressed = rows.find(
      r => r.querySelector('td')?.textContent === metricLabel('answer_similarity'),
    )!
    // That is the row the explanation is for: the others get a glance, this one
    // gets returned to.
    expect(within(regressed as HTMLElement).getByText(/.+/, { selector: '.cmp-means' })).toBeTruthy()
  })

  it('choosing another pair stays collapsed while a result exists', async () => {
    const { container } = render()
    await screen.findByText('text summary')
    // Reaching a finished comparison through an expanded picker is one extra
    // step on every visit; the form is wanted only when there is nothing to
    // compare.
    const pick = container.querySelector('details.cmp-pick') as HTMLDetailsElement
    expect(pick).toBeTruthy()
    expect(pick.open).toBe(false)
  })

  it('with no result the picker is expanded, or the screen is blank and mute', async () => {
    compareMock.mockResolvedValue(undefined as never)
    const { container } = renderWithRealm(<ComparisonPage />, '/compare', 'demo')
    await waitFor(() => expect(container.querySelector('details.cmp-pick')).toBeTruthy())
    expect((container.querySelector('details.cmp-pick') as HTMLDetailsElement).open).toBe(true)
  })
})
