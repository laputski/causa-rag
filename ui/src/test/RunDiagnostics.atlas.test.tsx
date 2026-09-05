import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import RunDiagnostics from '../components/RunDiagnostics'
import type { ExperimentDetail } from '../api/client'

// Two things this panel could not do, and both were the answer to "how does a
// reader tell that a classified failure was caught": it printed the server's
// English sentence with no identifier beside it, and what the run made
// impossible to check lived on another tab.

function run(over: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    run_id: 'r1', config: { pipeline_source: 'in_process' },
    aggregate_metrics: {}, question_results: [],
    diagnostics: [
      { id: 'duplicates', severity: 'warn', title: 'Duplicates in the context',
        detail: '86 repeated chunks in the retrieved context.', action: '',
        failure_ids: ['F01'] },
      { id: 'bm25_dominance', severity: 'info', title: 'The semantic half rules the merge',
        detail: '113/127 came from the semantic half alone.', action: '',
        failure_ids: ['F17', 'F21'] },
    ],
    trace_gaps: [
      { field: 'candidate_source_refs',
        unavailable: 'The gain from a different context size cannot be estimated.',
        remedy: 'Return a candidate window wider than the final context.' },
    ],
    diagnosis_depth: 'partial',
    ...over,
  } as unknown as ExperimentDetail
}

function renderPanel(detail: ExperimentDetail) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><RunDiagnostics run={detail} /></MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('run diagnostics and the catalogue', () => {
  it('a finding names the catalogue entries it is evidence for', () => {
    // Before this, a reader saw a sentence and had no way to tell whether it
    // named a failure the platform knows.
    renderPanel(run())
    expect(screen.getByRole('link', { name: 'F01' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'F17' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'F21' })).toBeTruthy()
  })

  it('the link opens that entry and not the catalogue at large', () => {
    renderPanel(run())
    const link = screen.getByRole('link', { name: 'F01' }) as HTMLAnchorElement
    expect(link.getAttribute('href')).toContain('/atlas')
    expect(link.getAttribute('href')).toContain('entry=F01')
  })

  it('a finding is titled from its identifier and not from the server sentence', () => {
    // Distinguishable on purpose: in English the translation happens to read
    // the same as the server's own sentence, so a fixture carrying the real
    // wording would pass whether or not the lookup happens at all.
    renderPanel(run({
      diagnostics: [{ id: 'duplicates', severity: 'warn',
                      title: 'SERVER PROSE THAT MUST NOT BE SHOWN',
                      detail: 'x', action: '', failure_ids: ['F01'] }],
    } as Partial<ExperimentDetail>))
    expect(screen.queryByText('SERVER PROSE THAT MUST NOT BE SHOWN')).toBeNull()
    expect(screen.getByText(/Duplicates in the context|Дубликаты в контексте/)).toBeTruthy()
  })

  it('a finding the interface has no wording for still shows the server sentence', () => {
    // Honest degradation: an id this file does not know yet must not render an
    // empty row.
    renderPanel(run({
      diagnostics: [{ id: 'a_signal_added_later', severity: 'info',
                      title: 'Something the interface has never heard of',
                      detail: 'x', action: '', failure_ids: [] }],
    } as Partial<ExperimentDetail>))
    expect(screen.getByText('Something the interface has never heard of')).toBeTruthy()
  })

  it('what the run made impossible to check is shown beside what it found', () => {
    // It used to live on the prescription tab. A check that could not run and a
    // check that ran and found nothing are the same absence unless one is named.
    renderPanel(run())
    expect(screen.getByText('candidate_source_refs')).toBeTruthy()
    expect(screen.getByText(/cannot be estimated/)).toBeTruthy()
  })

  it('a run that recorded everything says so instead of showing an empty panel', () => {
    renderPanel(run({ trace_gaps: [], diagnosis_depth: 'full' } as Partial<ExperimentDetail>))
    expect(screen.getByText(/записал всё|recorded everything/)).toBeTruthy()
  })

  it('a finding no entry names carries no link and is still shown', () => {
    renderPanel(run({
      diagnostics: [{ id: 'header_only', severity: 'warn', title: 'Chunks carry headings only',
                      detail: 'x', action: '', failure_ids: [] }],
    } as Partial<ExperimentDetail>))
    expect(screen.queryByRole('link', { name: /^F\d+$/ })).toBeNull()
    // The platform is saying something the catalogue has no place for, and the
    // finding must not vanish for it.
    expect(screen.getByText(/заголовк|headings/i)).toBeTruthy()
  })
})
