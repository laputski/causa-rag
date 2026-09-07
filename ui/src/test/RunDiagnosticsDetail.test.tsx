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

  it('writes the clause as well as the frame, when the value arrives as parts', () => {
    // The last of the four values that reached a reader in English whatever
    // their language. The frame said "three metrics were recorded against
    // questions that cannot support them" in theirs, and then named the three
    // in the server's words.
    renderPanel(run([{
      id: 'metric_without_grounds', severity: 'error',
      title: 'A metric was computed where it has no grounds',
      detail: '2 metric(s) of this run were recorded against questions that cannot support '
        + 'them: answer_similarity on 20 question(s) where the run reached the generator does '
        + 'not hold. Averaged into the run’s numbers, these are confident values about '
        + 'something nobody measured.',
      action: '', failure_ids: [], detail_key: 'metric_without_grounds',
      params: {
        metrics: 2,
        grounds: [
          { metric: 'answer_similarity', count: 20, precondition: 'reached_the_generator' },
          { metric: 'retrieval_recall_at_k', count: 7, precondition: 'answerable' },
        ],
      },
    }]))
    const detail = screen.getByText(/answer_similarity/, { selector: '.find-detail' })
    // Both clauses, joined, each carrying its own numbers.
    expect(detail.textContent).toContain('answer_similarity')
    expect(detail.textContent).toContain('retrieval_recall_at_k')
    expect(detail.textContent).toContain('20')
    expect(detail.textContent).toContain('7')
    // The precondition in the reader's words and never as the code it travels as.
    expect(detail.textContent).toContain('прогон дошёл до генератора')
    expect(detail.textContent).toContain('корпус покрывает')
    expect(detail.textContent).not.toContain('reached_the_generator')
    expect(detail.textContent).not.toContain('does not hold')
    expect(detail.textContent).not.toContain('[object Object]')
  })

  it('writes both shapes of a clause, when one finding says two things', () => {
    renderPanel(run([{
      id: 'aggregate_disagrees', severity: 'error',
      title: 'The run’s numbers are not its questions’ numbers',
      detail: 'server English', action: '', failure_ids: [],
      detail_key: 'aggregate_disagrees',
      params: {
        disagreeing: 2, metrics: 9,
        named: [
          { shape: 'against_the_mean', metric: 'recall_at_k', recorded: '0.9000',
            mean: '0.1000', questions: 20 },
          { shape: 'on_no_question_at_all', metric: 'faithfulness', recorded: '0.5000' },
        ],
      },
    }]))
    const detail = screen.getByText(/recall_at_k/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('0.9000')
    expect(detail.textContent).toContain('ни на одном вопросе')
    expect(detail.textContent).toContain('вопросам, которые её несут')
    expect(detail.textContent).not.toContain('in the run')
    expect(detail.textContent).not.toContain('[object Object]')
  })

  it('says which promise a segmentation broke, in the reader’s words', () => {
    renderPanel(run([{
      id: 'segmentation_broke_its_promise', severity: 'error',
      title: 'The segmentation did not do what its name says',
      detail: 'server English', action: '', failure_ids: [],
      detail_key: 'segmentation_broke_its_promise',
      params: { strategy: 'structure_aware', unmet: [{ promise: 'structure_aware' }] },
    }]))
    const detail = screen.getByText(/structure_aware/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('заголовки прочитать не удалось')
    expect(detail.textContent).not.toContain('the headings could not be read')
  })

  it('names why coverage was not checked, and shows a library’s own words as they came', () => {
    renderPanel(run([{
      id: 'unverified_coverage', severity: 'warn',
      title: 'Corpus coverage was not verified',
      detail: 'server English', action: '', failure_ids: [],
      detail_key: 'unverified_coverage',
      params: { reason: 'the_index_is_unreachable', note: ' (Connection refused)' },
    }]))
    const detail = screen.getByText(/Connection refused/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('до указателя не удалось достучаться')
    expect(detail.textContent).not.toContain('the_index_is_unreachable')
  })

  it('shows a reason from a run stored before reasons were named', () => {
    // The identifier arrives beside the sentence and never instead of it, so
    // a stored run that has only the sentence still reads.
    renderPanel(run([{
      id: 'unverified_coverage', severity: 'warn',
      title: 'Corpus coverage was not verified',
      detail: 'server English', action: '', failure_ids: [],
      detail_key: 'unverified_coverage',
      params: { reason: 'index unreachable: boom', note: '' },
    }]))
    const detail = screen.getByText(/index unreachable: boom/, { selector: '.find-detail' })
    expect(detail.textContent).toContain('Классы отвечаемости')
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
