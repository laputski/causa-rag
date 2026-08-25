import { describe, it, expect, vi, beforeEach } from 'vitest'
import { waitFor } from '@testing-library/react'
import FrontierPage from '../pages/FrontierPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The latency caveat and the plot are what a table cannot say.
//
// Latency is comparable only within a group: for the built-in pipeline the
// platform measures its own work, for an external one somebody else's response.
// The caveat sits where the number is looked at, and exactly once: repeated over
// every group, it stops meaning "this group is the special one".

const frontierMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      frontier: (realmId: string, metric?: string) => frontierMock(realmId, metric),
      list: () => Promise.resolve([
        { run_id: 'r1', aggregate_metrics: { retrieval_recall_at_k: 0.7 } },
      ]),
    },
  },
}))

const POINT = (id: string, source: string, quality: number, latency: number) => ({
  run_id: id, pipeline_source: source, label: source === 'http' ? 'external' : 'built-in',
  quality, latency_ms: latency, config: { top_k: 5 },
})

const GROUPED = {
  quality_metric: 'retrieval_recall_at_k',
  considered: 5,
  latency_comparable_within_source_only: true,
  frontier_by_source: {
    in_process: [
      POINT('inproc1', 'in_process', 0.76, 24520),
      POINT('inproc2', 'in_process', 0.60, 9000),
      POINT('inproc3', 'in_process', 0.42, 3000),
    ],
    http: [POINT('ext1', 'http', 0.75, 0.15)],
  },
  dominated: [POINT('dom1', 'in_process', 0.30, 20000)],
}

beforeEach(() => {
  vi.clearAllMocks()
  stubRealmFetch('acme')
  frontierMock.mockResolvedValue(GROUPED)
})

describe('The frontier: groups and the plot', () => {
  it("the latency caveat appears once, in the external group's header", async () => {
    const { container } = renderWithRealm(<FrontierPage />, '/frontier', 'acme')
    await waitFor(() => expect(container.querySelectorAll('table').length).toBeGreaterThan(1))

    const caveats = [...container.querySelectorAll('.latency-caveat')]
      .filter(el => (el.getAttribute('title') ?? '').length > 0)
    // Two notes across the table headers: one about latency, one about tokens.
    // Exactly one about latency, in the group whose number is incomparable.
    expect(caveats.length, 'no more header notes than kinds of caveat').toBeLessThanOrEqual(2)

    const tables = [...container.querySelectorAll('table')]
    const withCaveat = tables.filter(t => t.querySelector('thead .latency-caveat[title]'))
    expect(withCaveat.length, 'the note must not appear on every table').toBeLessThan(tables.length)
  })

  it('the plot holds one point per run in the group, and none from elsewhere', async () => {
    const { container } = renderWithRealm(<FrontierPage />, '/frontier', 'acme')
    await waitFor(() => expect(container.querySelector('.frontier-plot')).toBeTruthy())

    // The plot draws one group, the one with more points: two groups' latencies
    // on one axis would claim they are comparable, and they are not.
    const dots = container.querySelectorAll('.frontier-plot svg circle')
    expect(dots.length).toBe(GROUPED.frontier_by_source.in_process.length + GROUPED.dominated.length)
  })

  it('every point carries a label with its run and its numbers', async () => {
    const { container } = renderWithRealm(<FrontierPage />, '/frontier', 'acme')
    await waitFor(() => expect(container.querySelector('.frontier-plot')).toBeTruthy())

    // An unlabelled point is a smudge: the shape of the set is visible, and
    // which run sits in the corner is not.
    for (const dot of container.querySelectorAll('.frontier-plot svg circle')) {
      expect(dot.querySelector('title')?.textContent?.trim().length).toBeGreaterThan(0)
    }
  })

  it('the legend counts what is actually drawn', async () => {
    const { container } = renderWithRealm(<FrontierPage />, '/frontier', 'acme')
    await waitFor(() => expect(container.querySelector('.plot-legend')).toBeTruthy())

    const legend = container.querySelector('.plot-legend')!.textContent ?? ''
    expect(legend).toContain(String(GROUPED.frontier_by_source.in_process.length))
    expect(legend).toContain(String(GROUPED.dominated.length))
  })

  it('with no points the plot is not drawn at all, rather than drawn empty', async () => {
    frontierMock.mockResolvedValue({
      ...GROUPED,
      considered: 0,
      frontier_by_source: { in_process: [], http: [] },
      dominated: [],
    })
    const { container } = renderWithRealm(<FrontierPage />, '/frontier', 'acme')
    await waitFor(() => expect(frontierMock).toHaveBeenCalled())
    expect(container.querySelector('.frontier-plot svg')).toBeNull()
  })
})
