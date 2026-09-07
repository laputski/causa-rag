import { describe, it, expect, beforeEach, afterAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import i18n from '../i18n'
import RunDiagnostics from '../components/RunDiagnostics'
import type { ExperimentDetail } from '../api/client'

/**
 * The line of a finding that used to arrive in English.
 *
 * A title in Russian above two English sentences was the reported symptom, and
 * the title and the action were fixed first because both are static prose on
 * the server. The detail is composed there out of what the detector measured,
 * so it stayed, until the values started travelling beside it.
 *
 * What is checked here is the render and not the bundle: a key can exist in
 * both languages and still never be reached, and a sentence can be reached
 * with its placeholders unfilled, which renders the placeholder's own name to
 * a reader who cannot know that is what happened.
 */
function run(diagnostics: unknown[]): ExperimentDetail {
  return {
    run_id: 'r1', config: { pipeline_source: 'in_process' },
    aggregate_metrics: {}, question_results: [], diagnostics,
    trace_gaps: [], diagnosis_depth: 'full',
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

const DUPLICATES = {
  id: 'duplicates', severity: 'warn', title: 'Duplicates in the context',
  detail: '86 repeated chunks inside a single question’s context, across 12 question(s).',
  action: '', failure_ids: [], detail_key: 'duplicates',
  params: { repeats: 86, questions: 12 },
}

beforeEach(async () => { await i18n.changeLanguage('ru') })
afterAll(async () => { await i18n.changeLanguage('en') })

describe('a finding’s detail reaches the reader in their language', () => {
  it('renders the Russian sentence and keeps the numbers in it', () => {
    renderPanel(run([DUPLICATES]))
    const detail = screen.getByText(/86/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('12')
    expect(detail.textContent).not.toContain('repeated chunks')
    expect(detail.textContent).not.toMatch(/\{\{/)
  })

  it('falls back to the server’s English for a finding it has no sentence for', () => {
    // Every run stored before the values travelled is this case, and so is a
    // detector added on the server before the interface catches up.
    renderPanel(run([{
      id: 'a_finding_from_the_future', severity: 'warn', title: 'Something new',
      detail: 'A sentence this side has never seen.', action: '', failure_ids: [],
    }]))
    expect(screen.getByText('A sentence this side has never seen.')).toBeTruthy()
  })

  it('says which half of a merge in Russian, and never leaves the code in the sentence', () => {
    // The half travels as a code because a label composed on the server is an
    // English word arriving inside a Russian sentence.
    renderPanel(run([{
      id: 'bm25_dominance', severity: 'info', title: 'The semantic half rules the merge',
      detail: '113/127 came from the semantic half alone (89%).', action: '', failure_ids: [],
      detail_key: 'bm25_dominance',
      params: { count: 113, chunks: 127, half: 'semantic', other: 'keyword', share: '89%' },
    }]))
    const detail = screen.getByText(/113/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('смысловая')
    expect(detail.textContent).not.toContain('semantic')
    expect(detail.textContent).not.toContain('keyword')
  })

  it('tells the two cases of one identifier apart', () => {
    // Both are "the index and the query do not use the same model", on
    // different evidence, under one identifier because the catalogue entry
    // they evidence is one entry.
    renderPanel(run([{
      id: 'index_and_query_models_differ', severity: 'error',
      title: 'The index and the query do not use the same model',
      detail: 'The corpus was indexed by bge_m3 1.0 and this run queried with bge_m3 2.0.',
      action: '', failure_ids: [], detail_key: 'index_and_query_models_differ.by_version',
      params: { indexing: 'bge_m3', indexed_version: '1.0',
                querying: 'bge_m3', querying_version: '2.0' },
    }]))
    const detail = screen.getByText(/1\.0/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('2.0')
    expect(detail.textContent).not.toContain('The corpus was indexed')
  })
})
