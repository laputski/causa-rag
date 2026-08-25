import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor, fireEvent } from '@testing-library/react'
import FrontierPage from '../pages/FrontierPage'
import { stubRealmFetch, renderWithRealm } from './realmTestUtils'

// The page that shows which configurations are worth choosing
// between. The properties pinned here are the ones a plain table would lose.

const frontierMock = vi.fn()

vi.mock('../api/client', () => ({
  api: {
    experiments: {
      frontier: (realmId: string, metric?: string) => frontierMock(realmId, metric),
      list: () => Promise.resolve([
        { run_id: 'r1', aggregate_metrics: { retrieval_recall_at_k: 0.7, faithfulness: 0.4 } },
      ]),
    },
  },
}))

const GROUPED = {
  quality_metric: 'retrieval_recall_at_k',
  considered: 2,
  latency_comparable_within_source_only: true,
  frontier_by_source: {
    in_process: [{
      run_id: 'inproc1', pipeline_source: 'in_process', label: 'built-in',
      quality: 0.76, latency_ms: 24520, config: { top_k: 5 },
    }],
    http: [{
      run_id: 'ext1', pipeline_source: 'http', label: 'external',
      quality: 0.75, latency_ms: 0.15, config: { top_k: 5 },
    }],
  },
  dominated: [],
}

function renderPage() {
  return renderWithRealm(<FrontierPage />, '/frontier', 'acme')
}

describe('FrontierPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    stubRealmFetch('acme')
    frontierMock.mockResolvedValue(GROUPED)
  })

  // @lat: [[frontier#Choosing a configuration, and checking the metric that chooses#Reviewer-facing surface#Groups stay separate, never merged]]
  it('keeps the two pipeline sources in separate groups', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByText('Built-in pipeline')).toBeInTheDocument())
    expect(screen.getByText('External system')).toBeInTheDocument()
    // Both survive: the external run's 0.15 ms would otherwise wipe out the
    // in-process run at the same quality on an artefact of measurement.
    expect(screen.getByText('external')).toBeInTheDocument()
    expect(screen.getByText('built-in')).toBeInTheDocument()
  })

  // @lat: [[navigation#Navigation and the patterns shared across pages#The latency caveat travels with the number]]
  it('marks the external group as measuring the handover only', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByText('External system')).toBeInTheDocument())
    // Once beside the external group's latency column, and once as the
    // explanation under the table.
    expect(screen.getAllByText(/handover only/).length).toBeGreaterThan(0)
    expect(screen.getByText(/factor of 160,000/)).toBeInTheDocument()
  })

  // @lat: [[frontier#Choosing a configuration, and checking the metric that chooses#Reviewer-facing surface#The quality measure is chosen from what was actually measured]]
  it('offers only quality measures the Realm has actually recorded', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByText('Built-in pipeline')).toBeInTheDocument())
    const select = screen.getByRole('combobox') as HTMLSelectElement
    const options = [...select.options].map(o => o.value)
    expect(options).toContain('faithfulness')
    expect(options).not.toContain('answer_relevancy')
  })

  it('re-asks for the frontier when the quality measure changes', async () => {
    renderPage()
    await waitFor(() => expect(frontierMock).toHaveBeenCalledWith('acme', 'retrieval_recall_at_k'))
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'faithfulness' } })
    await waitFor(() => expect(frontierMock).toHaveBeenCalledWith('acme', 'faithfulness'))
  })

  // A run missing either number is excluded server-side, which leaves this
  // page with zero considered runs — a state that has to explain itself
  // rather than look like "no runs exist".
  // @lat: [[frontier#Choosing a configuration, and checking the metric that chooses#Reviewer-facing surface#An empty frontier explains itself]]
  it('explains why a run can be missing instead of showing an empty table', async () => {
    frontierMock.mockResolvedValue({
      quality_metric: 'retrieval_recall_at_k', considered: 0,
      latency_comparable_within_source_only: true, frontier_by_source: {}, dominated: [],
    })
    renderPage()
    await waitFor(() => expect(screen.getByText('Nothing to compare yet')).toBeInTheDocument())
    expect(screen.getByText(/a measured response time/)).toBeInTheDocument()
  })

  // @lat: [[frontier#Choosing a configuration, and checking the metric that chooses#Reviewer-facing surface#The beaten configurations stay reachable]]
  it('keeps the beaten configurations behind a toggle rather than dropping them', async () => {
    frontierMock.mockResolvedValue({
      ...GROUPED,
      considered: 3,
      dominated: [{
        run_id: 'bad1', pipeline_source: 'in_process', label: 'beaten',
        quality: 0.1, latency_ms: 90000, config: {},
      }],
    })
    renderPage()
    await waitFor(() => expect(screen.getByText(/Show the beaten/)).toBeInTheDocument())
    expect(screen.queryByText('beaten')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText(/Show the beaten/))
    expect(screen.getByText('beaten')).toBeInTheDocument()
  })
})
